"""
ml_model.py - XGBoost 학습/서빙 공통 모듈
- build_features:   가격 수준과 무관한(scale-free) 피처. 기본 23개 + 확장(자기 이력 정규화·52주 위치·시장) 13개 = 36개
                    모델이 실제로 쓰는 피처 목록은 metadata['features'] 에 저장 → 서빙 시 그 목록만 사용
- build_labels:     N거래일 후 수익률 >= 목표 → 1 (고정 목표 라벨, 총 성능 평가 기준)
- risk_adjusted_forward: N거래일 후 로그수익률 / (σ₆₀·√N)  — '상대 라벨'(train.py) 계산용
- ModelStore:       저장된 모델(JSON) + 메타데이터(보정식 포함) 로드 및 확률 예측
"""
import json
import os

import numpy as np
import pandas as pd

from analyzer import StrategyAnalyzer, SIGNAL_FEATURES

MODEL_DIR = 'models'
PARAMS_FILE = os.path.join(MODEL_DIR, 'xgb_params.json')
UNIVERSE_FILE = 'universe.json'
MARKET_TICKER = '^KS11'     # KOSPI 지수 (확장 피처용)
Z_WINDOW = 252              # 자기 이력 정규화 창 (1년)
Z_MIN = 120


def model_path(period: str) -> str:
    return os.path.join(MODEL_DIR, f'xgb_{period}.json')


def metadata_path(period: str) -> str:
    return os.path.join(MODEL_DIR, f'metadata_{period}.json')


def cv_report_path(period: str) -> str:
    return os.path.join(MODEL_DIR, f'cv_report_{period}.json')


def oof_path(period: str) -> str:
    """walk-forward fold test 예측 (out-of-fold) — 보정식 적합·평가용"""
    return os.path.join(MODEL_DIR, f'oof_{period}.parquet')


def load_universe(filepath: str = UNIVERSE_FILE) -> dict:
    """{ticker: {'name': ..., 'sector': ...}}"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_train_universe(filepath: str) -> tuple:
    """
    학습 유니버스 → ({ticker: {'name', 'member': [[from, to], ...] | None}}, meta)
    - universe_train.json 형식 ({'meta', 'tickers'}): 편입 구간 포함
    - universe.json 형식 ({ticker: {name, sector}}): member=None (전 기간)
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    if 'tickers' in raw and 'meta' in raw:
        return raw['tickers'], raw['meta']
    return ({t: {'name': v['name'], 'member': None} for t, v in raw.items()},
            {'mode': 'static', 'survivorship_biased': True})


def _zscore(s: pd.Series) -> pd.Series:
    r = s.rolling(Z_WINDOW, min_periods=Z_MIN)
    return (s - r.mean()) / r.std().replace(0, np.nan)


def build_features(df: pd.DataFrame, market: pd.DataFrame = None) -> pd.DataFrame:
    """
    IndicatorEngine.calculate_all 결과(df)로부터 모델 입력 피처를 생성.
    market: KOSPI 일봉 (없으면 시장 피처는 NaN — XGBoost 는 결측 분기로 처리)
    """
    close = df['Close']
    feats = pd.DataFrame(index=df.index)

    # ── 기본 23개 ──────────────────────────────────────────────
    # 1) 신호 점수 (-1~+1)
    scores = StrategyAnalyzer.compute_signal_scores(df)
    for f in SIGNAL_FEATURES:
        feats[f'sig_{f}'] = scores[f]

    # 2) 원시 지표 (정규화)
    feats['rsi'] = df['RSI_14']
    feats['stoch_k'] = df['STOCHk_14_3_3']
    feats['stoch_d'] = df['STOCHd_14_3_3']
    feats['macd_hist_pct'] = df['MACDh_12_26_9'] / close
    band = (df['BBU_20_2.0'] - df['BBL_20_2.0']).replace(0, np.nan)
    feats['bb_pctb'] = (close - df['BBL_20_2.0']) / band
    feats['bb_width'] = band / close
    for w in (20, 60, 120):
        feats[f'dist_ema{w}'] = close / df[f'EMA_{w}'] - 1.0
    feats['atr_pct'] = df['ATRr_14'] / close

    # 3) 가격/거래량
    for w in (5, 20, 60):
        feats[f'ret_{w}'] = close.pct_change(w)
    log_ret = np.log(close / close.shift(1))
    feats['vol_20'] = log_ret.rolling(20).std() * np.sqrt(252)
    vol_ma = df['Volume'].rolling(20).mean().replace(0, np.nan)
    feats['volume_ratio'] = df['Volume'] / vol_ma

    # ── 확장 13개 ──────────────────────────────────────────────
    # 4) 자기 이력 대비 정규화: "이 종목치고" 높은가 낮은가
    feats['rsi_z'] = _zscore(feats['rsi'])
    feats['vol_ratio_1y'] = feats['vol_20'] / feats['vol_20'].rolling(Z_WINDOW, min_periods=Z_MIN).median()
    feats['bb_width_z'] = _zscore(feats['bb_width'])
    feats['volume_ratio_z'] = _zscore(np.log(feats['volume_ratio'].replace(0, np.nan)))

    # 5) 52주 위치
    feats['dist_52w_high'] = close / close.rolling(252, min_periods=Z_MIN).max() - 1.0
    feats['dist_52w_low'] = close / close.rolling(252, min_periods=Z_MIN).min() - 1.0

    # 6) 시장 상황 / 시장 대비
    if market is not None and len(market):
        m = market['Close'].reindex(df.index).ffill()
        m_ret = np.log(m / m.shift(1))
        feats['kospi_ret_20'] = m.pct_change(20)
        feats['kospi_ret_60'] = m.pct_change(60)
        feats['kospi_vol_20'] = m_ret.rolling(20).std() * np.sqrt(252)
        feats['kospi_dist_ema60'] = m / m.ewm(span=60, adjust=False).mean() - 1.0
        feats['rel_ret_20'] = feats['ret_20'] - feats['kospi_ret_20']
        feats['rel_ret_60'] = feats['ret_60'] - feats['kospi_ret_60']
        feats['beta_60'] = log_ret.rolling(60).cov(m_ret) / m_ret.rolling(60).var().replace(0, np.nan)
    else:
        for c in ('kospi_ret_20', 'kospi_ret_60', 'kospi_vol_20', 'kospi_dist_ema60',
                  'rel_ret_20', 'rel_ret_60', 'beta_60'):
            feats[c] = np.nan

    return feats.replace([np.inf, -np.inf], np.nan)


BASE_FEATURES = [
    *[f'sig_{f}' for f in SIGNAL_FEATURES],
    'rsi', 'stoch_k', 'stoch_d', 'macd_hist_pct', 'bb_pctb', 'bb_width',
    'dist_ema20', 'dist_ema60', 'dist_ema120', 'atr_pct', 'ret_5', 'ret_20', 'ret_60', 'vol_20', 'volume_ratio',
]
EXTENDED_FEATURES = BASE_FEATURES + [
    'rsi_z', 'vol_ratio_1y', 'bb_width_z', 'volume_ratio_z', 'dist_52w_high', 'dist_52w_low',
    'kospi_ret_20', 'kospi_ret_60', 'kospi_vol_20', 'kospi_dist_ema60', 'rel_ret_20', 'rel_ret_60', 'beta_60',
]


def build_labels(df: pd.DataFrame, days: int, target_pct: float) -> pd.Series:
    """N거래일 후 수익률 >= 목표(%) → 1. 미래 데이터가 없는 마지막 N행은 NaN."""
    fwd = df['Close'].shift(-days) / df['Close'] - 1.0
    label = (fwd >= target_pct / 100.0).astype(float)
    label[fwd.isna()] = np.nan
    return label


def risk_adjusted_forward(df: pd.DataFrame, days: int) -> pd.Series:
    """N거래일 후 로그수익률 / (t 시점까지 60일 일간 변동성 × √N). 미래 없는 마지막 N행은 NaN."""
    log_ret = np.log(df['Close'] / df['Close'].shift(1))
    sigma = log_ret.rolling(60).std().replace(0, np.nan)
    fwd_log = np.log(df['Close'].shift(-days) / df['Close'])
    return fwd_log / (sigma * np.sqrt(days))


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def apply_calibration(cal: dict, model_prob, hv0):
    """보정식: logit(실현가능성) = a + b·logit(HV₀) + c·(logit(p_model) − logit(base_rate))"""
    z = cal['a'] + cal['b'] * _logit(hv0) + cal['c'] * (_logit(model_prob) - _logit(cal['base_rate']))
    return 1.0 / (1.0 + np.exp(-z))


class ModelStore:
    """기간별(short/long) XGBoost 모델 로더. 모델 파일·보정식이 없으면 available=False."""

    def __init__(self, periods):
        self.models = {}
        self.meta = {}
        for p in periods:
            self._load(p)

    def _load(self, period: str):
        mp, md = model_path(period), metadata_path(period)
        if not (os.path.exists(mp) and os.path.exists(md)):
            print(f"[ModelStore] ⚠️ '{period}' 모델 없음 ({mp}) → train.py 실행 필요. 기준 확률(HV₀)만 사용")
            return
        try:
            from xgboost import XGBClassifier
            with open(md, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            if 'calibration' not in meta:
                print(f"[ModelStore] ⚠️ '{period}' 보정식 없음 → `python train.py --calibrate-only` 실행 필요")
                return
            model = XGBClassifier()
            model.load_model(mp)
            self.meta[period], self.models[period] = meta, model
            cal = meta['calibration']
            print(f"[ModelStore] '{period}' 모델 로드 ✓ (라벨 {meta.get('label_mode', 'target')}, "
                  f"피처 {len(meta['features'])}개, 보정 c={cal['c']:.3f})")
        except Exception as e:
            print(f"[ModelStore] ⚠️ '{period}' 모델 로드 실패: {e}")

    def available(self, period: str) -> bool:
        return period in self.models

    def predict(self, period: str, df: pd.DataFrame, market: pd.DataFrame = None):
        """
        df(IndicatorEngine 결과)의 마지막 행 예측. 모델이 없으면 None.
        반환: {'probability': 모델 확률(0~1), 'calibration': 보정식, 'label_mode': 라벨 정의}
        """
        if not self.available(period):
            return None
        meta = self.meta[period]
        x = build_features(df, market)[meta['features']].iloc[[-1]]
        prob = float(self.models[period].predict_proba(x)[0, 1])
        return {'probability': prob, 'calibration': meta['calibration'],
                'label_mode': meta.get('label_mode', 'target')}
