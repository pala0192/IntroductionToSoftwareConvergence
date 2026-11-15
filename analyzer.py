"""
analyzer.py - 기술 신호 점수(모델 피처) + 자연어 해석
"""
import numpy as np
import pandas as pd

# 신호 그룹: 기술(추세/모멘텀) vs 지표(오실레이터)
# ⚠️ 학습된 모델의 피처(sig_*)이므로 정의를 바꾸면 train.py 재학습 필요
TREND_FEATURES = ['macd', 'ema20', 'ema60', 'ema120', 'atr']
OSCILLATOR_FEATURES = ['rsi', 'stoch', 'bollinger']
SIGNAL_FEATURES = OSCILLATOR_FEATURES + TREND_FEATURES

STOP_LOSS_ATR = 2.0   # 손절가 = 현재가 − ATR × 2


class StrategyAnalyzer:
    @staticmethod
    def _normalize_rsi(rsi_series):
        # RSI 30 이하 과매도(매수시그널 +1), 70 이상 과매수(매도시그널 -1)
        score = (50 - rsi_series) / 20.0
        return np.clip(score, -1.0, 1.0)

    @staticmethod
    def _normalize_stoch(k_series, d_series):
        # K가 D를 상향 돌파하면 매수, 하향 돌파하면 매도
        diff = k_series - d_series
        return np.clip(diff / 10.0, -1.0, 1.0)

    @staticmethod
    def _normalize_macd(macd_series, signal_series):
        diff = macd_series - signal_series
        return np.clip(diff / (macd_series.abs() + 0.001), -1.0, 1.0)

    @staticmethod
    def _normalize_bollinger(price_series, lower_series, upper_series):
        # 하단 터치 시 +1, 상단 터치 시 -1
        mid = (lower_series + upper_series) / 2
        score = (mid - price_series) / (upper_series - lower_series) * 2
        return np.clip(score, -1.0, 1.0)

    @staticmethod
    def _normalize_ema(price_series, ema_series):
        diff_pct = (price_series - ema_series) / ema_series * 100
        return np.clip(diff_pct / 5.0, -1.0, 1.0)

    @classmethod
    def compute_signal_scores(cls, df: pd.DataFrame) -> pd.DataFrame:
        """신호별 -1~+1 점수 (학습/서빙 공통 단일 소스)"""
        scores = pd.DataFrame(index=df.index)
        scores['rsi'] = cls._normalize_rsi(df['RSI_14'])
        scores['stoch'] = cls._normalize_stoch(df['STOCHk_14_3_3'], df['STOCHd_14_3_3'])
        scores['macd'] = cls._normalize_macd(df['MACD_12_26_9'], df['MACDs_12_26_9'])
        scores['bollinger'] = cls._normalize_bollinger(df['Close'], df['BBL_20_2.0'], df['BBU_20_2.0'])
        scores['ema20'] = cls._normalize_ema(df['Close'], df['EMA_20'])
        scores['ema60'] = cls._normalize_ema(df['Close'], df['EMA_60'])
        scores['ema120'] = cls._normalize_ema(df['Close'], df['EMA_120'])

        # 가격의 20일 변동 방향성을 ATR로 나누어 방향과 변동성을 동시 고려
        price_momentum = df['Close'] - df['Close'].shift(20)
        scores['atr'] = np.clip(price_momentum / (df['ATRr_14'] + 0.001), -1.0, 1.0).fillna(0)
        return scores


def _valid(*vals) -> bool:
    return all(v is not None and not np.isnan(v) for v in vals)


def technical_summary(row) -> str:
    """RSI / MACD 기준 한 줄 요약 (추천 이유용)"""
    if row is None:
        return None
    parts = []
    rsi = row.get('RSI_14')
    if _valid(rsi):
        if rsi >= 70:
            parts.append(f"RSI {rsi:.0f} 과매수(조정 가능성)")
        elif rsi <= 30:
            parts.append(f"RSI {rsi:.0f} 과매도(반등 가능성)")
        else:
            parts.append(f"RSI {rsi:.0f} 중립")
    macd, sig = row.get('MACD_12_26_9'), row.get('MACDs_12_26_9')
    if _valid(macd, sig):
        parts.append("MACD 시그널선 위(상승 흐름)" if macd > sig else "MACD 시그널선 아래(하락 흐름)")
    return "📈 기술 신호: " + ", ".join(parts) if parts else None


class ConsultingInterpreter:
    @staticmethod
    def interpret(row, target_return_pct=5.0):
        """특정 시점의 데이터를 바탕으로 자연어 해석과 목표가/손절가를 산출합니다."""
        feedback = []

        # 1. RSI 분석
        rsi = row['RSI_14']
        if rsi >= 70:
            feedback.append("⚠️ [RSI(14D)] 현재 70 이상으로 과매수 구간입니다. 단기 하락 조정 가능성이 높습니다.")
        elif rsi <= 30:
            feedback.append("✅ [RSI(14D)] 현재 30 이하로 과매도 구간입니다. 단기적인 기술적 반등이 기대됩니다.")
        else:
            feedback.append(f"ℹ️ [RSI(14D)] 현재 {rsi:.1f}로 중립적인 모멘텀입니다.")

        # 2. 볼린저 밴드 분석
        price = float(row['Close'])
        if price >= row['BBU_20_2.0']:
            feedback.append("⚠️ [볼린저 밴드(20D, 2.0σ)] 주가가 상단 밴드를 터치/돌파했습니다. 강한 저항 및 매도 압력이 예상됩니다.")
        elif price <= row['BBL_20_2.0']:
            feedback.append("✅ [볼린저 밴드(20D, 2.0σ)] 주가가 하단 밴드를 터치/돌파했습니다. 지지선 역할을 하여 반등할 확률이 높습니다.")

        # 3. MACD 분석
        macd = row['MACD_12_26_9']
        signal = row['MACDs_12_26_9']
        if macd > signal and macd > 0:
            feedback.append("📈 [MACD(12D-26D-9D)] MACD가 시그널 선 위에 위치하여 상승 추세가 견고합니다.")
        elif macd < signal and macd < 0:
            feedback.append("📉 [MACD(12D-26D-9D)] MACD가 시그널 선 아래 위치하여 하락 추세가 진행 중입니다.")

        # 4. 목표가(기간별 고정 목표 수익률) / 손절가(ATR 기반 — 종목 변동성 반영)
        target_price = price * (1 + target_return_pct / 100.0)
        atr = row.get('ATRr_14')
        if _valid(atr) and atr > 0:
            stop_loss = max(price - STOP_LOSS_ATR * float(atr), price * 0.01)
            feedback.append(f"🛑 [손절가] 현재가에서 ATR(14D)의 {STOP_LOSS_ATR:g}배 아래로 설정했습니다.")
        else:
            stop_loss = price * (1 - target_return_pct / 100.0 * 0.5)

        return {
            'text': "\n".join(feedback),
            'current_price': price,
            'target_price': target_price,
            'stop_loss': stop_loss,
            'expected_return_pct': (target_price - price) / price * 100,
            'risk_pct': (price - stop_loss) / price * 100,
        }
