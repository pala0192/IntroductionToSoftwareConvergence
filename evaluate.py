"""
evaluate.py - 보조지표 모델 성능 / 총 성능 평가 (XGBoost vs 로지스틱 회귀, walk-forward OOF = 표본 밖)

사용: python evaluate.py --snapshot v1 --tag wf_v1      → reports/evaluation_wf_v1.json

- XGBoost: train.py 가 저장한 walk-forward OOF 예측(models/oof_{period}.parquet)과 메타데이터(보정식)를 그대로 사용
- 로지스틱 회귀: 같은 데이터·라벨·피처·fold 로 학습 (표준화 + L2, C 는 fold inner_val log loss 로 선택)
  + 같은 방식의 OOF 보정식
- 평가 세트: 학습 유니버스 전체 OOF / 서빙 25종목(universe.json) OOF / 섹터 대표 5종목 OOF
- fold 별 보조지표 AUC 평균±표준편차 (보정 전 확률 → 1차 판단 기준)
- 주의: 보정식을 OOF 로 적합하고 같은 OOF 로 총 성능을 평가하므로 총 성능 지표는 약간 낙관적
- 보조지표 성능: 모델 확률 vs 모델 자신의 학습 라벨
    AUC / 날짜 내 AUC(같은 날 종목 고르기) / 종목 내 AUC(같은 종목의 시점 고르기) / Brier skill / 수익률 순위 상관
- 총 성능: 서버와 같은 ProbabilityEngine 결과 vs 고정 목표 달성 여부 (사용자가 보는 '실현가능성'의 정답)
    실현가능성 AUC / Brier skill / 종목 내 AUC / 점수–수익률 Spearman / 등급별 달성률·수익률
- 뉴스는 과거 시점 재현 불가 → '일반'(보정 없음)
"""
import argparse
import json
import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import HOLDING_PERIODS, TARGET_RETURNS, PERIOD_NAMES
import ml_model
from ml_model import ModelStore, load_universe
from probability_engine import ProbabilityEngine, GRADE_ORDER
from train import build_dataset, walk_forward_folds, fold_frames, fit_calibration, load_data

EVAL_TICKERS = ['373220.KS', '034020.KS', '068270.KS', '138040.KS', '003230.KS']   # 섹터별 첫 종목
MODELS = ['xgb', 'logreg']
MODEL_NAMES = {'xgb': 'XGBoost', 'logreg': '로지스틱'}
LOGREG_C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]


# ── 지표 ─────────────────────────────────────────────────────────────────
def _auc(y, p):
    y = np.asarray(y)
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None


def grouped_auc(df: pd.DataFrame, y: str, p: str, by: str):
    vals = [a for _, g in df.groupby(by) if (a := _auc(g[y], g[p])) is not None]
    return round(float(np.mean(vals)), 4) if vals else None


def prob_metrics(df: pd.DataFrame, y: str, p: str) -> dict:
    df = df[df[p].notna()]       # hv0: 상장 직후 등 변동성 이력 부족 행은 결측
    yy, pp = df[y].to_numpy(), df[p].to_numpy()
    base = float(np.mean((yy - yy.mean()) ** 2))
    brier = float(brier_score_loss(yy, pp))
    auc = _auc(yy, pp)
    return {
        'auc': None if auc is None else round(auc, 4),
        'within_date_auc': grouped_auc(df, y, p, 'date'),
        'within_ticker_auc': grouped_auc(df, y, p, 'ticker'),
        'brier': round(brier, 4),
        'brier_skill': round(1 - brier / base, 4) if base > 0 else None,
        'mean_pred_pct': round(float(pp.mean()) * 100, 1),
        'actual_rate_pct': round(float(yy.mean()) * 100, 1),
    }


def grade_table(df: pd.DataFrame, col: str) -> list:
    out = []
    for g in GRADE_ORDER:
        d = df[df[col] == g]
        out.append({'grade': g, 'n': int(len(d)), 'share_pct': round(len(d) / len(df) * 100, 1),
                    'hit_rate_pct': round(float(d['target_label'].mean()) * 100, 1) if len(d) else None,
                    'mean_fwd_return_pct': round(float(d['fwd_ret'].mean()) * 100, 2) if len(d) else None})
    return out


def set_results(df: pd.DataFrame) -> dict:
    out = {'n': int(len(df)), 'tickers': int(df['ticker'].nunique()),
           'date_from': str(pd.Timestamp(df['date'].min()).date()), 'date_to': str(pd.Timestamp(df['date'].max()).date()),
           'label_positive_pct': round(float(df['label'].mean()) * 100, 1),
           'target_positive_pct': round(float(df['target_label'].mean()) * 100, 1),
           'hv0_only': prob_metrics(df, 'target_label', 'hv0'), 'models': {}}
    for k in MODELS:
        good = df[f'{k}_grade'].isin(['매수', '관심'])
        out['models'][k] = {
            'indicator': {**prob_metrics(df, 'label', f'{k}_prob'),
                          'fold_auc': fold_auc(df, f'{k}_prob'),
                          # 확률 압축 정도: 10~90% 분위 폭 (좁을수록 기준선 근처에 몰림)
                          'prob_p10_p90': [round(float(df[f'{k}_prob'].quantile(q)), 4) for q in (0.1, 0.9)],
                          'spearman_vs_fwd_return': round(float(spearmanr(df[f'{k}_prob'], df['fwd_ret']).statistic), 4)},
            'total': {
                **prob_metrics(df, 'target_label', f'{k}_feas'),
                'score_spearman_vs_fwd_return': round(float(spearmanr(df[f'{k}_score'], df['fwd_ret']).statistic), 4),
                'grades': grade_table(df, f'{k}_grade'),
                'buy_interest_n': int(good.sum()),
                'buy_interest_hit_pct': round(float(df.loc[good, 'target_label'].mean()) * 100, 1) if good.any() else None,
                'buy_interest_ret_pct': round(float(df.loc[good, 'fwd_ret'].mean()) * 100, 2) if good.any() else None,
                'others_ret_pct': round(float(df.loc[~good, 'fwd_ret'].mean()) * 100, 2) if (~good).any() else None,
            },
        }
    return out


# ── 모델 ─────────────────────────────────────────────────────────────────
def fit_logreg(tr: pd.DataFrame, va: pd.DataFrame, features: list):
    best = None
    for c in LOGREG_C_GRID:
        m = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000))
        m.fit(tr[features], tr['label'])
        loss = log_loss(va['label'], m.predict_proba(va[features])[:, 1], labels=[0, 1])
        if best is None or loss < best[0]:
            best = (loss, c, m)
    return best[2], best[1]


def logreg_oof(data: pd.DataFrame, features: list, purge: int) -> tuple:
    """XGBoost 와 같은 walk-forward fold 로 로지스틱 회귀 OOF 예측 → (oof DataFrame, train 양성비율 가중평균, fold별 C)"""
    dates = np.sort(data['date'].unique())
    data = data.assign(_di=np.searchsorted(dates, data['date'].to_numpy()))
    parts, pos_w, cs = [], [], []
    for f in walk_forward_folds(len(dates), purge):
        seg = fold_frames(data, dates, f)
        m, c = fit_logreg(seg['train'], seg['val'], features)
        parts.append(seg['test'][['date', 'ticker']].assign(fold=f['fold'], prob=m.predict_proba(seg['test'][features])[:, 1]))
        pos_w.append((len(seg['train']), float(seg['train']['label'].mean())))
        cs.append(c)
    base = sum(n * r for n, r in pos_w) / sum(n for n, _ in pos_w)
    return pd.concat(parts, ignore_index=True), base, cs


def fold_auc(df: pd.DataFrame, p: str) -> dict:
    vals = [a for _, g in df.groupby('fold') if (a := _auc(g['label'], g[p])) is not None]
    return {'per_fold': [round(v, 4) for v in vals], 'mean': round(float(np.mean(vals)), 4),
            'std': round(float(np.std(vals)), 4)}


def score_rows(te: pd.DataFrame, k: str, cal: dict, label_mode: str, period: str) -> pd.DataFrame:
    days, target = int(HOLDING_PERIODS[period]), float(TARGET_RETURNS[period])
    feas, score, grade = [], [], []
    for p, hv0 in zip(te[f'{k}_prob'], te['hv0']):
        r = ProbabilityEngine.compute(
            ml={'probability': float(p), 'calibration': cal, 'label_mode': label_mode},
            hv0=None if pd.isna(hv0) else float(hv0), news=None, target_return_pct=target,
            holding_days=days, period_name=PERIOD_NAMES[period])
        feas.append(r['feasibility_pct'] / 100.0)
        score.append(r['recommendation']['score'])
        grade.append(r['recommendation']['grade'])
    te[f'{k}_feas'], te[f'{k}_score'], te[f'{k}_grade'] = feas, score, grade
    return te


# ── 출력 ─────────────────────────────────────────────────────────────────
def fmt(v, s=''):
    return 'n/a' if v is None else f"{v}{s}"


def print_set(title: str, r: dict):
    print(f"\n  {title} ({r['date_from']} ~ {r['date_to']}, {r['tickers']}종목, 표본 {r['n']:,}, "
          f"라벨 양성 {r['label_positive_pct']}%, 목표 달성 {r['target_positive_pct']}%)")
    print(f"  {'':<16}{'AUC':>8}{'날짜내':>8}{'종목내':>8}{'BSS':>9}{'예측':>8}{'실제':>8}{'Spearman':>10}")
    line = lambda lab, m, sp: print(f"  {lab:<16}{fmt(m['auc']):>8}{fmt(m['within_date_auc']):>8}{fmt(m['within_ticker_auc']):>8}"
                                    f"{fmt(m['brier_skill']):>9}{m['mean_pred_pct']:>7}%{m['actual_rate_pct']:>7}%{fmt(sp):>10}")
    for k in MODELS:
        m = r['models'][k]['indicator']
        line(f"{MODEL_NAMES[k]} 보조지표", m, m['spearman_vs_fwd_return'])
    for k in MODELS:
        fa = r['models'][k]['indicator']['fold_auc']
        pw = r['models'][k]['indicator']['prob_p10_p90']
        print(f"  {MODEL_NAMES[k]} fold별 AUC: {fa['per_fold']} → {fa['mean']} ± {fa['std']} · 확률 p10~p90 {pw[0]}~{pw[1]}")
    for k in MODELS:
        m = r['models'][k]['total']
        line(f"{MODEL_NAMES[k]} 총성능", m, m['score_spearman_vs_fwd_return'])
    line('참고: HV₀ 단독', r['hv0_only'], None)
    for k in MODELS:
        t = r['models'][k]['total']
        g = ' / '.join(f"{x['grade']} {x['n']}건·{fmt(x['hit_rate_pct'], '%')}·{fmt(x['mean_fwd_return_pct'], '%')}"
                       for x in t['grades'] if x['n'])
        print(f"  {MODEL_NAMES[k]} 등급(건수·달성률·평균수익률): {g}")


# ── main ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--snapshot', required=True, help='학습에 쓴 스냅샷 태그 (train.py --snapshot 과 같아야 함)')
    ap.add_argument('--tag', default='latest')
    ap.add_argument('--models-dir', default=None, help='평가할 모델 폴더 (예: models_archive/stage1)')
    args = ap.parse_args()
    if args.models_dir:
        ml_model.MODEL_DIR = args.models_dir

    serving = load_universe()
    store = ModelStore(HOLDING_PERIODS)
    frames, market, universe, src = load_data(args.snapshot)
    print(f"✓ {len(frames)}개 종목 + KOSPI 로드 (스냅샷 {args.snapshot}) · 서빙 {len(serving)}종목 · "
          f"대표 5종목: {', '.join(serving[t]['name'] for t in EVAL_TICKERS)}")

    results = {}
    for period in HOLDING_PERIODS:
        days, target = int(HOLDING_PERIODS[period]), float(TARGET_RETURNS[period])
        meta = store.meta[period]
        if meta.get('snapshot') != args.snapshot:
            raise SystemExit(f"모델({period})은 스냅샷 {meta.get('snapshot')} 로 학습됨 ≠ --snapshot {args.snapshot}")
        features, label_mode = meta['features'], meta.get('label_mode', 'target')

        data, _ = build_dataset(frames, days, target, features, label_mode, market, universe)
        xgb_oof = pd.read_parquet(ml_model.oof_path(period))[['date', 'ticker', 'fold', 'prob']]
        lr_oof, lr_base, lr_cs = logreg_oof(data, features, purge=days)

        te = data.merge(xgb_oof.rename(columns={'prob': 'xgb_prob'}), on=['date', 'ticker'], how='inner')
        te = te.merge(lr_oof.rename(columns={'prob': 'logreg_prob'}).drop(columns='fold'), on=['date', 'ticker'], how='inner')
        if len(te) != len(xgb_oof):
            raise SystemExit(f"OOF 행 불일치 (xgb {len(xgb_oof)}, 병합 {len(te)}) → 같은 스냅샷으로 train.py 재실행 필요")

        cals = {'xgb': meta['calibration'],
                'logreg': fit_calibration(te['logreg_prob'], te['hv0'], te['target_label'], lr_base, 'oof')}
        for k in MODELS:
            te = score_rows(te, k, cals[k], label_mode, period)

        results[period] = {
            'label_mode': label_mode, 'feature_set': meta.get('feature_set', 'base'), 'n_features': len(features),
            'calibration': {k: {x: round(cals[k][x], 4) for x in ('a', 'b', 'c')} for k in MODELS},
            'logreg_C_per_fold': lr_cs,
            'full_oof': set_results(te),
            'serving_oof': set_results(te[te['ticker'].isin(serving)]),
            'eval_set': set_results(te[te['ticker'].isin(EVAL_TICKERS)]),
        }
        tuning = meta.get('tuning')
        if tuning:   # 튜닝에 쓰지 않은 fold 만 → 튜닝 효과의 공정한 비교 기준
            ho = te[te['fold'].isin(tuning['holdout_folds'])]
            results[period]['tuning'] = tuning
            results[period]['holdout_full'] = set_results(ho)
            results[period]['holdout_serving'] = set_results(ho[ho['ticker'].isin(serving)])
        r = results[period]
        print(f"\n{'=' * 80}\n📊 [{PERIOD_NAMES[period]}] {days}거래일 · 목표 +{target}% · 라벨 {label_mode} · "
              f"피처 {len(features)}개 · 보정 c: XGB {r['calibration']['xgb']['c']:+.3f}, 로지스틱 {r['calibration']['logreg']['c']:+.3f}\n{'=' * 80}")
        print_set(f"① 학습 유니버스 전체 OOF", r['full_oof'])
        print_set(f"② 서빙 {len(serving)}종목 OOF", r['serving_oof'])
        print_set('③ 대표 5종목 OOF', r['eval_set'])
        if 'holdout_full' in r:
            print_set(f"④ holdout fold {r['tuning']['holdout_folds']} 전체 (튜닝 미사용)", r['holdout_full'])
            print_set(f"⑤ holdout fold {r['tuning']['holdout_folds']} 서빙 {len(serving)}종목", r['holdout_serving'])

    os.makedirs('reports', exist_ok=True)
    out = os.path.join('reports', f'evaluation_{args.tag}.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'tag': args.tag, 'snapshot': args.snapshot, 'validation': 'walk_forward_oof',
                   'eval_tickers': EVAL_TICKERS, 'news': 'excluded (neutral)',
                   'note': '총 성능은 OOF 로 적합한 보정식을 같은 OOF 로 평가 → 약간 낙관적. fold별 보조지표 AUC 를 1차 기준으로.',
                   'results': results},
                  f, indent=2, ensure_ascii=False)
    print(f"\n💾 저장: {out}")
