"""
config.py - 홀딩 기간 / 목표 수익률 설정
(신호 가중치는 XGBoost 모델로 대체됨 → models/ 참고)
"""

# 고정 홀딩 기간 (레이블 → 거래일 수)
# 일수는 모두 '거래일' 기준 (OptionsAnalyzer의 T=days/252, train.py의 shift(-days)와 일치)
HOLDING_PERIODS = {
    'short': 20,    # 단기: 약 1개월
    'long':  100,   # 장기: 약 5개월
}

# 기간별 고정 목표 수익률 (%) — 모델 라벨 정의에도 사용
TARGET_RETURNS = {
    'short': 5.0,
    'long':  13.0,
}

PERIOD_NAMES = {
    'short': '단기',
    'long':  '장기',
}

DEFAULT_PERIOD = 'short'


def period_label_to_days(label: str) -> int:
    """'short' → 20, 'long' → 100 (거래일)"""
    return HOLDING_PERIODS.get(label, HOLDING_PERIODS[DEFAULT_PERIOD])


def target_return_for(label: str) -> float:
    """기간 레이블별 고정 목표 수익률(%)"""
    return TARGET_RETURNS.get(label, TARGET_RETURNS[DEFAULT_PERIOD])
