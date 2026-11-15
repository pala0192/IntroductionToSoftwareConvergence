"""
tune.py - XGBoost 하이퍼파라미터 베이지안 탐색 (Optuna TPE) — early stopping 설정 포함

사용:
    python tune.py --snapshot v2-wide --period short long --timeout 1800 --apply

설계
- fold 구조는 train.walk_forward_folds 그대로. 마지막 HOLDOUT_FOLDS(2)개 fold 는 튜닝에서 사용하지 않음
  (단기: 튜닝 fold 1~3 / holdout 4~5, 장기: 튜닝 1~2 / holdout 3~4) → 튜닝 전후 비교는 holdout 으로만
- 목적값: 튜닝 fold 들의 test AUC 평균 (최대화). test 는 해당 fold 의 전체 행
- 속도: 학습·early stopping 은 행 25% 표본 (seed 고정, 모든 trial 동일)
- trial 단위 조기 중단(pruning): fold 마다 누적 평균 AUC 를 보고 → MedianPruner
- 탐색 공간: 트리 구조·정규화·학습률 + early_stopping_rounds + es_metric(early stopping 기준 지표)
  n_estimators 는 2000 고정 (실제 트리 수는 early stopping 이 결정)
- 첫 trial = 현재 xgb_params.json 설정 (기존 설정 기준점)
- 산출물: models/tuning/optuna.db (sqlite, 중단 후 재개 가능), models/tuning/{period}_best.json
- --apply: 최적 파라미터를 models/xgb_params.json 에 기록 (최초 1회 xgb_params.prev.json 백업)
"""
import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
import optuna
from sklearn.metrics import roc_auc_score

from config import HOLDING_PERIODS, TARGET_RETURNS
from ml_model import PARAMS_FILE
from train import (FEATURE_SETS, TUNING_DIR, build_dataset, fold_model, load_data, walk_forward_folds)

HOLDOUT_FOLDS = 2
TRAIN_SAMPLE = 0.25
N_ESTIMATORS_CAP = 2000
SEED = 42


def search_space(trial: optuna.Trial) -> dict:
    return {
        'n_estimators': N_ESTIMATORS_CAP,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 2, 8),
        'min_child_weight': trial.suggest_float('min_child_weight', 1.0, 300.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 100.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'early_stopping_rounds': trial.suggest_int('early_stopping_rounds', 20, 300),
        'es_metric': trial.suggest_categorical('es_metric', ['logloss', 'auc']),
        'random_state': SEED,
    }


def baseline_trial_values(p: dict) -> dict:
    """현재 xgb_params 를 탐색 공간 값으로 (범위 밖이면 경계로 맞춤)"""
    clip = lambda v, lo, hi: float(min(max(v, lo), hi))
    return {
        'learning_rate': clip(p['learning_rate'], 0.01, 0.2), 'max_depth': int(min(max(p['max_depth'], 2), 8)),
        'min_child_weight': clip(p['min_child_weight'], 1.0, 300.0), 'subsample': clip(p['subsample'], 0.5, 1.0),
        'colsample_bytree': clip(p['colsample_bytree'], 0.3, 1.0), 'reg_lambda': clip(p['reg_lambda'], 1e-3, 100.0),
        'reg_alpha': clip(p['reg_alpha'], 1e-3, 10.0), 'gamma': clip(p['gamma'], 0.0, 5.0),
        'early_stopping_rounds': int(min(max(p.get('early_stopping_rounds', 50), 20), 300)),
        'es_metric': p.get('es_metric', 'auc'),
    }


def prepare_folds(data, features: list, days: int) -> tuple:
    """튜닝 fold 별 numpy 배열 (train/val 은 표본, test 는 전체) → (folds, tune_ids, holdout_ids)"""
    dates = np.sort(data['date'].unique())
    di = np.searchsorted(dates, data['date'].to_numpy())
    folds = walk_forward_folds(len(dates), purge=days)
    tune = folds[:len(folds) - HOLDOUT_FOLDS]
    if not tune:
        raise SystemExit("튜닝에 쓸 fold 가 없음")
    sample = np.random.default_rng(SEED).random(len(data)) < TRAIN_SAMPLE
    X = data[features].to_numpy(np.float32)
    y = data['label'].to_numpy(np.float32)
    out = []
    for f in tune:
        seg = {s: (di >= f[s][0]) & (di < f[s][1]) for s in ('train', 'val', 'test')}
        tr, va, te = seg['train'] & sample, seg['val'] & sample, seg['test']
        out.append({'fold': f['fold'], 'Xtr': X[tr], 'ytr': y[tr], 'Xva': X[va], 'yva': y[va],
                    'Xte': X[te], 'yte': y[te]})
        print(f"  fold {f['fold']}: train {tr.sum():,} / val {va.sum():,} (표본 {TRAIN_SAMPLE:.0%}) / test {te.sum():,} (전체)")
    return out, [f['fold'] for f in tune], [f['fold'] for f in folds[len(tune):]]


def make_objective(folds: list):
    def objective(trial: optuna.Trial) -> float:
        params = search_space(trial)
        aucs, iters = [], []
        for step, f in enumerate(folds):
            model = fold_model(params)
            model.fit(f['Xtr'], f['ytr'], eval_set=[(f['Xva'], f['yva'])], verbose=False)
            aucs.append(float(roc_auc_score(f['yte'], model.predict_proba(f['Xte'])[:, 1])))
            iters.append(int(model.best_iteration))
            trial.set_user_attr('fold_auc', [round(a, 4) for a in aucs])
            trial.set_user_attr('best_iteration', iters)
            trial.report(float(np.mean(aucs)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr('auc_std', round(float(np.std(aucs)), 4))
        return float(np.mean(aucs))
    return objective


def log_trial(study: optuna.Study, trial: optuna.trial.FrozenTrial):
    state = trial.state.name
    val = f"{trial.value:.4f}" if trial.value is not None else '  -   '
    best = f"{study.best_value:.4f}" if any(t.value is not None for t in study.trials) else '-'
    dur = (trial.datetime_complete - trial.datetime_start).total_seconds() if trial.datetime_complete else 0
    p = trial.params
    print(f"  #{trial.number:<3} {state:<8} AUC {val} (best {best}) {dur:5.0f}s | "
          f"lr {p.get('learning_rate', 0):.3f} d {p.get('max_depth')} mcw {p.get('min_child_weight', 0):.1f} "
          f"es {p.get('early_stopping_rounds')}/{p.get('es_metric')} iters {trial.user_attrs.get('best_iteration')}")


def tune_period(period: str, frames, market, universe, snapshot: str, base_params: dict, args) -> dict:
    days, target = int(HOLDING_PERIODS[period]), float(TARGET_RETURNS[period])
    features = FEATURE_SETS['extended']
    print(f"\n{'=' * 72}\n🔎 [{period}] {days}거래일 · TPE · timeout {args.timeout}s · 최대 {args.n_trials} trial\n{'=' * 72}")
    data, _ = build_dataset(frames, days, target, features, 'relative', market, universe)
    folds, tune_ids, holdout_ids = prepare_folds(data, features, days)
    print(f"  튜닝 fold {tune_ids} / holdout fold {holdout_ids} (튜닝에 사용 안 함)")

    study = optuna.create_study(
        study_name=f'{snapshot}_{period}', storage=args.storage, load_if_exists=True, direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=SEED, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0))
    if not study.trials:
        study.enqueue_trial(baseline_trial_values(base_params), user_attrs={'baseline': True})
    t0 = time.time()
    study.optimize(make_objective(folds), n_trials=args.n_trials, timeout=args.timeout, callbacks=[log_trial])

    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    base = next((t for t in study.trials if t.user_attrs.get('baseline')), None)
    best = study.best_trial
    best_params = search_space(optuna.trial.FixedTrial(best.params))
    result = {
        'study': study.study_name, 'snapshot': snapshot, 'period': period,
        'tune_folds': tune_ids, 'holdout_folds': holdout_ids, 'train_sample': TRAIN_SAMPLE,
        'best_value': round(best.value, 4), 'best_fold_auc': best.user_attrs.get('fold_auc'),
        'best_iteration': best.user_attrs.get('best_iteration'),
        'baseline_value': round(base.value, 4) if base and base.value is not None else None,
        'baseline_fold_auc': base.user_attrs.get('fold_auc') if base else None,
        'n_trials': len(study.trials), 'n_complete': len(done), 'n_pruned': len(pruned),
        'elapsed_sec_this_run': round(time.time() - t0),
        'finished_at': datetime.now().isoformat(timespec='seconds'),
        'params': best_params,
    }
    os.makedirs(TUNING_DIR, exist_ok=True)
    with open(os.path.join(TUNING_DIR, f'{period}_best.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n  ✓ best #{best.number}: 튜닝 fold AUC {result['best_value']} (기존 설정 {result['baseline_value']}) · "
          f"trial {len(study.trials)} (완료 {len(done)}, prune {len(pruned)})")
    print(f"    {best_params}")
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--snapshot', required=True)
    ap.add_argument('--period', nargs='+', choices=list(HOLDING_PERIODS), default=list(HOLDING_PERIODS))
    ap.add_argument('--n-trials', type=int, default=40)
    ap.add_argument('--timeout', type=int, default=1800, help='기간당 탐색 시간(초)')
    ap.add_argument('--storage', default=f"sqlite:///{os.path.join(TUNING_DIR, 'optuna.db')}")
    ap.add_argument('--apply', action='store_true', help='최적 파라미터를 xgb_params.json 에 기록')
    args = ap.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    os.makedirs(TUNING_DIR, exist_ok=True)
    with open(PARAMS_FILE, encoding='utf-8') as f:
        all_params = json.load(f)

    frames, market, universe, src = load_data(args.snapshot)
    print(f"✓ {len(frames)}개 종목 로드 (스냅샷 {args.snapshot})")

    results = {p: tune_period(p, frames, market, universe, args.snapshot, all_params[p], args) for p in args.period}

    if args.apply:
        prev = PARAMS_FILE.replace('.json', '.prev.json')
        if not os.path.exists(prev):
            shutil.copyfile(PARAMS_FILE, prev)
        for p, r in results.items():
            all_params[p] = r['params']
        with open(PARAMS_FILE, 'w', encoding='utf-8') as f:
            json.dump(all_params, f, indent=2, ensure_ascii=False)
        print(f"\n💾 {PARAMS_FILE} 갱신 ({', '.join(results)}) · 백업 {prev}")
