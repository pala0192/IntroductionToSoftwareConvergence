"""
options_analyzer.py - 역사적 변동성(HV) 기반 '기준 확률' HV₀

P(N거래일 후 로그수익률 >= log(1 + 목표))  — 수익률 ~ 정규분포(평균 0, σ√T) 가정
- 방향(추세)은 예측하지 않음: "이 종목의 변동성이라면 목표가 얼마나 어려운가"만 표현
  (과거 3년 추세를 반영하던 방식은 평가에서 역효과 → 제거, 보고서 15장)
- 방향 판단은 보조지표 모델이 HV₀ 를 조정하는 방식으로 담당 (probability_engine)
- 학습·평가(hv_series, 날짜별 벡터 계산)와 서버(get_historical_vol_probability, 마지막 날짜)가 같은 함수 사용
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

VOL_WINDOW = 504     # σ 추정 창: 최근 2년(504거래일)
MIN_OBS = 250        # 최소 관측치 (상장 1년 미만이면 계산하지 않음)


def hv_series(close: pd.Series, target_return_pct: float, holding_days: int) -> pd.DataFrame:
    """날짜별 연 변동성(σ)과 기준 확률 HV₀ (그 날짜까지의 데이터만 사용)"""
    log_ret = np.log(close / close.shift(1))
    sigma = log_ret.rolling(VOL_WINDOW, min_periods=MIN_OBS).std() * np.sqrt(252)
    sigma_t = sigma * np.sqrt(holding_days / 252)
    k = np.log(1 + target_return_pct / 100.0)
    prob = pd.Series(norm.cdf(-k / sigma_t), index=close.index).where(sigma_t > 0)
    return pd.DataFrame({'sigma': sigma, 'hv0': prob.clip(0.01, 0.99)})


class OptionsAnalyzer:

    @staticmethod
    def get_historical_vol_probability(df: pd.DataFrame, target_return_pct: float, holding_days: int) -> dict:
        """마지막 날짜 기준 HV₀ (데이터 부족 시 probability=None)"""
        last = hv_series(df['Close'], target_return_pct, holding_days).iloc[-1]
        if pd.isna(last['hv0']):
            return {'probability': None, 'historical_vol': None, 'data_source': '데이터 부족 (1년 미만)'}
        return {
            'probability': float(last['hv0']),
            'historical_vol': round(float(last['sigma']) * 100, 2),
            'data_source': f"역사적 변동성 기반 (최근 2년, 연 {last['sigma'] * 100:.1f}%)",
        }
