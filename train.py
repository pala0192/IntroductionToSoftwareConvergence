"""
train.py - XGBoost 통합(pooled) 모델 학습 + 결합 보정식 적합 (walk-forward 검증)

사용:
    python train.py --snapshot v1                 # 서빙 25종목 스냅샷
    python train.py --snapshot v2-wide            # KOSPI200+KOSDAQ150 학습 유니버스 스냅샷
    python train.py                               # 스냅샷 없이 universe.json 10년치를 네트워크에서 (재현 불가, 비권장)
    python train.py --snapshot v1 --calibrate-only   # 저장된 OOF 예측으로 보정식만 다시 적합

- 데이터: 스냅샷(data/snapshots/{tag})의 고정 기간 시세 → 매번 같은 데이터셋
- 유니버스: 스냅샷에 들어 있는 유니버스 사본. 편입 구간(member)이 있으면 그 날짜의 행만 학습에 사용
  (피처·라벨은 종목 전체 이력으로 계산 → 워밍업 확보, 편입 전후 정보 누설 없음: 피처는 과거, 라벨은 t..t+N)
- 라벨 (--label)
    target   : N거래일 후 수익률 >= 목표 수익률 (config.TARGET_RETURNS)
    relative : 같은 날짜 (편입) 종목들 중 위험 조정 수익률(로그수익률 / σ₆₀√N) 상위 30% → 1   [기본]
- 피처 (--features): base 23개 / extended 36개 [기본]
- 검증: walk-forward (확장 창)
    fold k: [train_core][purge N][inner_val 252일][purge N][test 252일]   (마지막 fold 의 test 는 남은 날짜 전부)
    train_core 로 학습, inner_val 로 early stopping, test 예측 = OOF
- 최종 서빙 모델: 라벨 있는 전체 데이터로 재학습, 트리 수 = fold best_iteration(+1) 의 중앙값 (early stopping 없음)
- 결합 보정식: OOF 예측으로 적합 (적합 데이터의 모든 예측이 그 예측을 만든 모델에게 표본 밖)
    logit(실현가능성) = a + b·logit(HV₀) + c·(logit(p_model) − logit(base_rate))   대상: 고정 목표 달성 여부
- 산출물: models/xgb_{period}.json, metadata_{period}.json, cv_report_{period}.json, oof_{period}.parquet
"""
import argparse
import json
import os
import sys
from datetime import datetime

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from xgboost import XGBClassifier

from config import HOLDING_PERIODS, TARGET_RETURNS
from data_loader import MarketDataLoader
from indicators import IndicatorEngine
from ml_model import (MODEL_DIR, PARAMS_FILE, MARKET_TICKER, BASE_FEATURES, EXTENDED_FEATURES,
                      build_features, build_labels, risk_adjusted_forward, apply_calibration, _logit,
                      load_universe, model_path, metadata_path, cv_report_path, oof_path)
from options_analyzer import hv_series

DATA_PERIOD = '10y'          # 스냅샷 미사용 시에만
RELATIVE_TOP = 0.30          # 상대 라벨: 상위 30%
FEATURE_SETS = {'base': BASE_FEATURES, 'extended': EXTENDED_FEATURES}

# walk-forward (거래일 수)
WF_TEST_DAYS = 252
WF_MIN_TRAIN_DAYS = 756
WF_INNER_VAL_DAYS = 252


def min_tickers_per_date(n_tickers: int) -> int:
    """상대 라벨 계산에 필요한 그날 최소 종목 수 (25종목: 10, 확장 유니버스: 100)"""
    return 10 if n_tickers < 100 else 100


# ── 데이터 로드 ──────────────────────────────────────────────────────────
def compute_indicators(raw_frames: dict) -> dict:
    frames = {}
    for i, (t, raw) in enumerate(raw_frames.items(), 1):
        try:
            frames[t] = IndicatorEngine.calculate_all(raw)
        except Exception as e:
            print(f"  ✗ {t} 지표 계산 실패: {e}")
        if i % 100 == 0:
            print(f"  지표 계산 {i}/{len(raw_frames)}")
    return frames


def load_data(snapshot: str = None) -> tuple:
    """→ (frames {ticker: 지표 DataFrame}, market, universe {ticker: {name, member}}, source 정보)"""
    if snapshot:
        from build_snapshot import load_snapshot
        raw, market, universe, manifest = load_snapshot(snapshot)
        src = {'snapshot': snapshot, 'start': manifest['start'], 'end': manifest['end'],
               'universe_mode': manifest['universe_meta'].get('mode'),
               'survivorship_biased': manifest['universe_meta'].get('survivorship_biased'),
               'universe_size': len(universe), 'loaded': len(raw), 'failed': len(manifest['failed'])}
    else:
        universe = {t: {'name': v['name'], 'member': None} for t, v in load_universe().items()}
        raw = {}
        for t in universe:
            try:
                raw[t] = MarketDataLoader(t).fetch_data(period=DATA_PERIOD, use_cache=False)
            except Exception as e:
                print(f"  ✗ {t} 로드 실패: {e}")
        market = MarketDataLoader(MARKET_TICKER).fetch_data(period=DATA_PERIOD, use_cache=False)
        src = {'snapshot': None, 'data_period': DATA_PERIOD, 'universe_mode': 'static',
               'survivorship_biased': True, 'universe_size': len(universe), 'loaded': len(raw)}
    return compute_indicators(raw), market, universe, src


def member_mask(index: pd.DatetimeIndex, intervals) -> np.ndarray:
    """편입 구간 [[from, to], ...] 안의 날짜 → True. intervals=None 이면 전부 True"""
    if not intervals:
        return np.ones(len(index), dtype=bool)
    m = np.zeros(len(index), dtype=bool)
    for a, b in intervals:
        m |= (index >= pd.Timestamp(a)) & (index <= pd.Timestamp(b))
    return m


# ── 데이터셋 ─────────────────────────────────────────────────────────────
def build_dataset(frames: dict, days: int, target_pct: float, features: list,
                  label_mode: str, market: pd.DataFrame = None, universe: dict = None) -> tuple:
    """
    → (data, stats)
    data 컬럼: features + label + target_label + fwd_ret + hv0 + ticker + date
    stats: 단계별 행 수 (유효 행 → 편입 마스크 → 최소 종목 수 필터)
    """
    parts, n_valid, n_member = [], 0, 0
    for ticker, df in frames.items():
        X = build_features(df, market)[features].astype('float32')
        X['target_label'] = build_labels(df, days, target_pct)
        X['fwd_ret'] = df['Close'].shift(-days) / df['Close'] - 1.0
        X['z'] = risk_adjusted_forward(df, days)
        X['hv0'] = hv_series(df['Close'], target_pct, days)['hv0']
        X['ticker'] = ticker
        X = X.dropna(subset=features + ['target_label', 'z'])
        n_valid += len(X)
        intervals = (universe or {}).get(ticker, {}).get('member')
        X = X[member_mask(X.index, intervals)]
        n_member += len(X)
        parts.append(X)
    data = pd.concat(parts)
    data.index.name = 'date'
    data = data.reset_index()

    min_t = min_tickers_per_date(len(frames))
    counts = data.groupby('date')['ticker'].transform('count')
    data = data[counts >= min_t].copy()
    if label_mode == 'relative':
        pct = data.groupby('date')['z'].rank(pct=True)
        data['label'] = (pct > 1 - RELATIVE_TOP).astype(float)
    else:
        data['label'] = data['target_label']
    data = data.drop(columns=['z']).sort_values(['date', 'ticker']).reset_index(drop=True)
    stats = {'rows_valid': int(n_valid), 'rows_in_membership': int(n_member),
             'rows_final': int(len(data)), 'min_tickers_per_date': min_t}
    return data, stats


# ── walk-forward 분할 ────────────────────────────────────────────────────
def walk_forward_folds(n_dates: int, purge: int, test_days: int = WF_TEST_DAYS,
                       min_train_days: int = WF_MIN_TRAIN_DAYS, inner_val_days: int = WF_INNER_VAL_DAYS) -> list:
    """
    고유 날짜 인덱스(0..n_dates-1) 기준 fold 경계. 각 구간은 [start, end) 반열린 구간.
    fold: train_core [0, v0) | purge | inner_val [v0+purge, t0-purge) | purge | test [t0, t1)
    """
    first_test = min_train_days + purge + inner_val_days + purge
    k = (n_dates - first_test) // test_days
    if k < 1:
        raise ValueError(f"날짜 {n_dates}개로는 walk-forward fold 를 만들 수 없음 (최소 {first_test + test_days}개 필요)")
    folds = []
    for i in range(k):
        t0 = first_test + i * test_days
        t1 = n_dates if i == k - 1 else t0 + test_days          # 마지막 fold 는 남은 날짜 전부
        val = (t0 - purge - inner_val_days, t0 - purge)
        train = (0, val[0] - purge)
        # 무결성: 각 구간의 라벨 창(t..t+N)이 다음 구간 날짜에 닿지 않아야 함
        assert train[1] - 1 + purge < val[0], "train/val purge 위반"
        assert val[1] - 1 + purge < t0, "val/test purge 위반"
        assert train[1] - train[0] >= min_train_days
        folds.append({'fold': i + 1, 'train': train, 'val': val, 'test': (t0, t1)})
    # test 구간끼리 겹치지 않고 연속
    for a, b in zip(folds, folds[1:]):
        assert a['test'][1] == b['test'][0], "test 구간 겹침/공백"
    return folds


def fold_frames(data: pd.DataFrame, dates: np.ndarray, fold: dict) -> dict:
    di = data['_di'].to_numpy()
    out = {}
    for seg in ('train', 'val', 'test'):
        a, b = fold[seg]
        out[seg] = data[(di >= a) & (di < b)]
    return out


def segment_summary(d: pd.DataFrame) -> dict:
    return {
        'rows': int(len(d)),
        'trading_days': int(d['date'].nunique()),
        'date_from': str(pd.Timestamp(d['date'].min()).date()) if len(d) else None,
        'date_to': str(pd.Timestamp(d['date'].max()).date()) if len(d) else None,
        'tickers': int(d['ticker'].nunique()),
        'positive_ratio': round(float(d['label'].mean()), 4) if len(d) else None,
        'target_positive_ratio': round(float(d['target_label'].mean()), 4) if len(d) else None,
    }


def evaluate(y, p) -> dict:
    y = np.asarray(y)
    return {
        'auc': round(float(roc_auc_score(y, p)), 4) if len(np.unique(y)) > 1 else None,
        'logloss': round(float(log_loss(y, p, labels=[0, 1])), 4),
        'brier': round(float(brier_score_loss(y, p)), 4),
    }


# ── 보정식 ───────────────────────────────────────────────────────────────
def fit_calibration(p, hv0, y, base_rate: float, fit_split: str) -> dict:
    """고정 목표 달성 여부 ~ logit(HV₀) + 모델 우위. hv0 결측 행은 제외."""
    p, hv0, y = np.asarray(p, float), np.asarray(hv0, float), np.asarray(y, float)
    ok = ~np.isnan(hv0)
    p, hv0, y = p[ok], hv0[ok], y[ok]
    X = np.column_stack([_logit(hv0), _logit(p) - _logit(base_rate)])
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(X, y)
    cal = {'a': float(lr.intercept_[0]), 'b': float(lr.coef_[0][0]), 'c': float(lr.coef_[0][1]),
           'base_rate': float(base_rate), 'fit_rows': int(len(y)), 'fit_split': fit_split}
    cal['fit_logloss_hv0'] = round(float(log_loss(y, hv0, labels=[0, 1])), 4)
    cal['fit_logloss_calibrated'] = round(float(log_loss(y, apply_calibration(cal, p, hv0), labels=[0, 1])), 4)
    return cal


def print_calibration(cal: dict):
    print(f"  🎯 보정식({cal['fit_split']} {cal['fit_rows']:,}행): a={cal['a']:+.3f}, b={cal['b']:.3f}, c={cal['c']:+.3f}")


# ── 학습 ─────────────────────────────────────────────────────────────────
TUNING_DIR = os.path.join(MODEL_DIR, 'tuning')


def fold_model(params: dict) -> XGBClassifier:
    """
    fold 학습용 모델. params 의 es_metric(logloss|auc, 기본 auc)을 eval_metric 의 마지막에 둠
    → XGBoost 는 여러 지표 중 마지막 지표로 early stopping 을 판단
    """
    p = dict(params)
    es_metric = p.pop('es_metric', 'auc')
    metrics = [m for m in ('logloss', 'auc') if m != es_metric] + [es_metric]
    return XGBClassifier(objective='binary:logistic', eval_metric=metrics, tree_method='hist', **p)


def final_params(params: dict, n_trees: int) -> dict:
    """최종 모델: early stopping 관련 키 제외, 트리 수 고정"""
    p = {k: v for k, v in params.items() if k not in ('early_stopping_rounds', 'es_metric')}
    p['n_estimators'] = n_trees
    return p


def tuning_info(period: str, params: dict):
    """tune.py 결과가 현재 파라미터와 일치하면 튜닝 정보 반환 (아니면 None)"""
    path = os.path.join(TUNING_DIR, f'{period}_best.json')
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        best = json.load(f)
    if best.get('params') != params:
        return None
    return {k: best.get(k) for k in ('study', 'snapshot', 'tune_folds', 'holdout_folds', 'best_value',
                                     'baseline_value', 'n_trials', 'n_pruned', 'finished_at')}


def run_walk_forward(data: pd.DataFrame, features: list, params: dict, purge: int, verbose: bool = True) -> tuple:
    """→ (fold 리포트 리스트, OOF DataFrame, train 양성비율 가중평균)"""
    dates = np.sort(data['date'].unique())
    data['_di'] = np.searchsorted(dates, data['date'].to_numpy())
    folds = walk_forward_folds(len(dates), purge)
    reports, oof_parts, pos_w = [], [], []
    for f in folds:
        seg = fold_frames(data, dates, f)
        tr, va, te = seg['train'], seg['val'], seg['test']
        model = fold_model(params)
        model.fit(tr[features], tr['label'], eval_set=[(va[features], va['label'])], verbose=False)
        p = model.predict_proba(te[features])[:, 1]
        oof_parts.append(te[['date', 'ticker', 'label', 'target_label', 'fwd_ret', 'hv0']].assign(fold=f['fold'], prob=p))
        pos_w.append((len(tr), float(tr['label'].mean())))
        rep = {'fold': f['fold'], 'best_iteration': int(model.best_iteration),
               **{s: segment_summary(seg[s]) for s in ('train', 'val', 'test')},
               'test_metrics': evaluate(te['label'], p)}
        reports.append(rep)
        if verbose:
            print(f"  fold {f['fold']}: train {rep['train']['rows']:>8,}행 {rep['train']['date_from']}~{rep['train']['date_to']} | "
                  f"val {rep['val']['rows']:>7,}행 {rep['val']['date_from']}~{rep['val']['date_to']} | "
                  f"test {rep['test']['rows']:>7,}행 {rep['test']['date_from']}~{rep['test']['date_to']} | best_iter {rep['best_iteration']}")
    data.drop(columns=['_di'], inplace=True)
    oof = pd.concat(oof_parts, ignore_index=True)
    base_rate = sum(n * r for n, r in pos_w) / sum(n for n, _ in pos_w)
    return reports, oof, base_rate


def train_period(period: str, frames: dict, market, universe: dict, src: dict, params: dict,
                 label_mode: str, feature_set: str):
    days = int(HOLDING_PERIODS[period])
    target = float(TARGET_RETURNS[period])
    features = FEATURE_SETS[feature_set]
    print(f"\n{'=' * 72}\n🚀 [{period}] {days}거래일 · 목표 {target}% · 라벨 {label_mode} · 피처 {feature_set}({len(features)}개)\n{'=' * 72}")

    data, dstats = build_dataset(frames, days, target, features, label_mode, market, universe)
    n_dates = int(data['date'].nunique())
    print(f"  📊 데이터셋 {len(data):,}행 · 거래일 {n_dates:,} · 종목 {data['ticker'].nunique()} · "
          f"{pd.Timestamp(data['date'].min()).date()} ~ {pd.Timestamp(data['date'].max()).date()} "
          f"(유효 {dstats['rows_valid']:,} → 편입 {dstats['rows_in_membership']:,} → 최종 {dstats['rows_final']:,})")

    reports, oof, base_rate = run_walk_forward(data, features, dict(params), purge=days)
    n_trees = int(np.median([r['best_iteration'] + 1 for r in reports]))

    # 최종 모델: 전체 데이터, 트리 수 고정
    model = XGBClassifier(objective='binary:logistic', tree_method='hist', **final_params(params, n_trees))
    model.fit(data[features], data['label'], verbose=False)
    print(f"  🌲 최종 모델: 전체 {len(data):,}행, 트리 {n_trees}개 (fold best_iteration+1 중앙값)")

    cal = fit_calibration(oof['prob'], oof['hv0'], oof['target_label'], base_rate, 'oof')
    print_calibration(cal)

    importance = dict(sorted(((f, round(float(v), 4)) for f, v in zip(features, model.feature_importances_)),
                             key=lambda kv: kv[1], reverse=True))
    tm = [r['test_metrics'] for r in reports]
    cv_summary = {k: {'mean': round(float(np.mean([m[k] for m in tm if m[k] is not None])), 4),
                      'std': round(float(np.std([m[k] for m in tm if m[k] is not None])), 4)}
                  for k in ('auc', 'logloss', 'brier')}
    in_test = oof[['date']].drop_duplicates()
    cv_report = {
        'period': period, 'holding_days': days, 'purge_days': days,
        'scheme': {'type': 'walk_forward_expanding', 'test_days': WF_TEST_DAYS,
                   'min_train_days': WF_MIN_TRAIN_DAYS, 'inner_val_days': WF_INNER_VAL_DAYS},
        'source': src, 'dataset': {**dstats, 'trading_days': n_dates, 'tickers': int(data['ticker'].nunique()),
                                   'date_from': str(pd.Timestamp(data['date'].min()).date()),
                                   'date_to': str(pd.Timestamp(data['date'].max()).date()),
                                   'positive_ratio': round(float(data['label'].mean()), 4)},
        'oof': {'rows': int(len(oof)), 'trading_days': int(len(in_test)),
                'date_from': str(pd.Timestamp(oof['date'].min()).date()),
                'date_to': str(pd.Timestamp(oof['date'].max()).date())},
        'final_model': {'rows': int(len(data)), 'n_estimators': n_trees},
        'folds': reports, 'cv_summary': cv_summary,
    }

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_model(model_path(period))
    oof.to_parquet(oof_path(period), index=False)
    meta = {
        'period': period, 'holding_days': days, 'target_return_pct': target,
        'label_mode': label_mode, 'feature_set': feature_set,
        'trained_at': datetime.now().isoformat(timespec='seconds'),
        'validation': 'walk_forward', 'n_folds': len(reports), 'final_n_estimators': n_trees,
        'snapshot': src.get('snapshot'), 'train_universe': {k: src.get(k) for k in ('universe_mode', 'survivorship_biased', 'universe_size')},
        'features': features, 'base_rate': round(base_rate, 4),
        'hyperparameters': params, 'tuning': tuning_info(period, params), 'cv_summary': cv_summary,
        'calibration': cal, 'feature_importance': importance,
        'dataset_rows': int(len(data)), 'n_tickers': int(data['ticker'].nunique()),
    }
    with open(metadata_path(period), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    with open(cv_report_path(period), 'w', encoding='utf-8') as f:
        json.dump(cv_report, f, indent=2, ensure_ascii=False)
    print(f"  💾 저장: {model_path(period)}, {metadata_path(period)}, {cv_report_path(period)}, {oof_path(period)}")


def calibrate_only(period: str):
    """저장된 모델·OOF 예측 유지, 보정식만 다시 적합 (데이터 재로딩 불필요)"""
    with open(metadata_path(period), encoding='utf-8') as f:
        meta = json.load(f)
    if not os.path.exists(oof_path(period)):
        raise SystemExit(f"{oof_path(period)} 없음 → walk-forward 로 다시 학습(python train.py --snapshot ...) 필요")
    oof = pd.read_parquet(oof_path(period))
    cal = fit_calibration(oof['prob'], oof['hv0'], oof['target_label'], meta['base_rate'], 'oof')
    print(f"\n[{period}]")
    print_calibration(cal)
    meta['calibration'] = cal
    with open(metadata_path(period), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--snapshot', default=None, help='data/snapshots/{tag} (build_snapshot.py 로 생성)')
    ap.add_argument('--label', choices=['target', 'relative'], default='relative')
    ap.add_argument('--features', choices=list(FEATURE_SETS), default='extended')
    ap.add_argument('--calibrate-only', action='store_true')
    args = ap.parse_args()

    if args.calibrate_only:
        for period in HOLDING_PERIODS:
            calibrate_only(period)
        sys.exit(0)

    print("=" * 72)
    print(f" 🧠 XGBoost 통합 모델 학습 (walk-forward) — 스냅샷 {args.snapshot or '없음(네트워크 10y)'}")
    print("=" * 72)
    frames, market, universe, src = load_data(args.snapshot)
    print(f"\n✓ {len(frames)}개 종목 + KOSPI 로드 완료 (유니버스 {src['universe_size']}, 모드 {src['universe_mode']}, "
          f"생존편향 {'있음' if src['survivorship_biased'] else '완화'})")

    with open(PARAMS_FILE, 'r', encoding='utf-8') as f:
        all_params = json.load(f)
    for period in HOLDING_PERIODS:
        train_period(period, frames, market, universe, src, all_params[period], args.label, args.features)

    print("\n✅ 완료. 서버(server.py)를 재시작하면 반영됩니다.")
