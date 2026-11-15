"""
probability_engine.py - 기준 확률(HV₀) + 보조지표 모델 조정 + 뉴스 → 실현가능성% / 추천 점수·등급·이유

1) 기준 확률 HV₀: 이 종목의 변동성으로 목표에 도달할 확률 (방향 예측 없음, options_analyzer)
2) 모델 조정 (val 구간 실제 결과로 적합한 보정식, train.py):
     logit(조정 확률) = a + b·logit(HV₀) + c·(logit(p_model) − logit(base_rate))
   모델이 쓸모없으면 c ≈ 0 → 조정 확률 ≈ HV₀ (가중치를 손으로 정하지 않음)
3) 추천 점수(0~100) = 50 + 50 × 우위 + 뉴스 보정
     우위 = (조정 확률 − 중립 확률) / max(중립, 100 − 중립)   ← '같은 변동성 종목 대비' 모델 우위
     중립 확률 = 같은 보정식에 p_model = base_rate(모델 중립)를 넣은 값 → HV₀ 재보정분은 우위에서 제외
   변동성이 크다는 이유만으로 점수가 오르지 않음. 근거가 없으면 50점(관망) 근처.
4) 뉴스 (Gemini 기사 평균 점수 s)
     호재 s ≥ +0.5: 실현가능성 ×1.10, +10점 / 위험 s ≤ −0.5: ×0.85, −15점, 등급 최대 '관망'
"""
import numpy as np

from analyzer import technical_summary
from ml_model import apply_calibration

NEWS_THRESHOLD = 0.5
NEWS_EFFECTS = {
    #           확률 배수, 추천 점수 보정
    'positive': {'multiplier': 1.10, 'score': 10, 'label': '호재', 'icon': '🟢'},
    'neutral':  {'multiplier': 1.00, 'score': 0,  'label': '일반', 'icon': '⚪'},
    'risk':     {'multiplier': 0.85, 'score': -15, 'label': '위험', 'icon': '🔴'},
}
GRADES = [(70, '매수'), (55, '관심'), (40, '관망'), (0, '회피')]
GRADE_ORDER = [g for _, g in GRADES]
RISK_MAX_GRADE = '관망'


def classify_news(score: float, analyzed: bool) -> str:
    if not analyzed:
        return 'neutral'
    if score >= NEWS_THRESHOLD:
        return 'positive'
    if score <= -NEWS_THRESHOLD:
        return 'risk'
    return 'neutral'


def grade_for(score: float) -> str:
    for cut, name in GRADES:
        if score >= cut:
            return name
    return GRADES[-1][1]


class ProbabilityEngine:

    @classmethod
    def compute(
        cls,
        ml: dict,                  # ModelStore.predict 결과 (없으면 None)
        hv0,                       # 기준 확률 HV₀ (0~1, 없으면 None)
        news: dict,                # NewsAnalyzer.get_news_score 결과
        target_return_pct: float,
        holding_days: int,
        period_name: str,
        latest_row=None,
        hist_vol_pct=None,         # 연 변동성(%) — 이유 문구용
    ) -> dict:
        news = news or {}
        news_score = float(news.get('score', 0.0) or 0.0)
        analyzed = bool(news.get('analyzed', False))
        category = classify_news(news_score, analyzed)
        effect = NEWS_EFFECTS[category]
        mult = effect['multiplier']

        reasons = []
        base = 50.0 if hv0 is None else float(hv0) * 100.0
        if hv0 is None:
            reasons.append("🎲 변동성 데이터 부족(상장 1년 미만) → 기준 확률 50%로 가정")
        else:
            vol_txt = f"연 변동성 {hist_vol_pct:.1f}%, " if hist_vol_pct else ""
            reasons.append(f"🎲 기준 확률 {base:.1f}% ({vol_txt}방향 가정 없음): 이 정도 변동성이면 목표 도달이 이만큼 가능")

        # ── 모델 조정 ────────────────────────────────────────────────
        model_prob = adjusted = None
        neutral = base     # 모델이 중립(p = base_rate)일 때의 확률 → 우위 비교 기준
        if ml is not None and hv0 is not None:
            model_prob = ml['probability'] * 100.0
            cal = ml['calibration']
            adjusted = float(apply_calibration(cal, ml['probability'], hv0)) * 100.0
            neutral = float(apply_calibration(cal, cal['base_rate'], hv0)) * 100.0
            what = '동료 종목 대비 위험 조정 상위 30%' if ml.get('label_mode') == 'relative' else '목표 달성'
            reasons.append(f"📊 보조지표 모델: '{what}' 확률 {model_prob:.1f}% (학습 평균 {cal['base_rate'] * 100:.1f}%) "
                           f"→ 보정식(c={cal['c']:+.2f}) 적용 후 {adjusted:.1f}%")
        else:
            adjusted = base
            reasons.append("📊 보조지표 모델 없음 → 기준 확률 그대로 사용")

        feasibility = float(np.clip(adjusted * mult, 1.0, 99.0))

        # ── 추천 점수: 같은 변동성 대비 우위 (모델 기여분만) ───────────
        denom = max(neutral, 100.0 - neutral) or 1.0
        edge = float(np.clip((adjusted - neutral) / denom, -1.0, 1.0))
        score = 50.0 + 50.0 * edge
        reasons.append(f"📏 같은 변동성 대비 우위 {adjusted - neutral:+.1f}%p (모델 중립 시 {neutral:.1f}%) "
                       f"→ 기본 점수 {score:.0f}점")

        score += effect['score']
        if category == 'neutral':
            reasons.append("⚪ 큰 뉴스 없음 → 보정 없음" if analyzed else "⚪ 뉴스 분석 불가 → 보정 없음")
        else:
            articles = news.get('article_reasons') or []
            top = max(articles, key=lambda a: abs(a.get('score', 0))) if articles else None
            why = f": {top['reason']}" if top and top.get('reason') else ""
            reasons.append(f"{effect['icon']} {effect['label']} 뉴스({news_score:+.2f}){why} "
                           f"→ 실현가능성 ×{mult:.2f}, {effect['score']:+d}점")

        tech = technical_summary(latest_row)
        if tech:
            reasons.append(tech)

        score = float(np.clip(score, 0.0, 100.0))
        grade = grade_for(score)
        if category == 'risk' and GRADE_ORDER.index(grade) < GRADE_ORDER.index(RISK_MAX_GRADE):
            grade = RISK_MAX_GRADE
            reasons.append(f"🔴 위험 뉴스로 등급을 '{RISK_MAX_GRADE}'까지 제한")

        r1 = lambda v: None if v is None else round(v, 1)
        return {
            'feasibility_pct': round(feasibility, 1),
            'recommendation': {'score': int(round(score)), 'grade': grade, 'reasons': reasons},
            'news_category': {
                'key': category, 'label': effect['label'], 'icon': effect['icon'],
                'score': round(news_score, 3), 'analyzed': analyzed, 'multiplier': mult,
            },
            'components': {
                'hv0_pct': r1(base),
                'model_prob_pct': r1(model_prob),
                'model_base_rate_pct': r1(ml['calibration']['base_rate'] * 100) if ml else None,
                'adjusted_pct': r1(adjusted),
                'neutral_pct': r1(neutral),
                'edge_pct': r1(adjusted - neutral),
                'calibration_c': round(ml['calibration']['c'], 3) if ml else None,
                'label_mode': ml.get('label_mode') if ml else None,
            },
            'holding_period': f"{period_name} ({int(holding_days)}거래일 · 목표 {target_return_pct:+g}%)",
        }
