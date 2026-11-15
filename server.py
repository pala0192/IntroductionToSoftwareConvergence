"""
server.py - 퀀트 트레이딩 분석 Flask REST API 서버
실행: python server.py   (환경변수 FLASK_DEBUG=1 이면 디버그 모드, HOST/PORT 로 바인딩 변경)
접속: http://localhost:5000
"""
import sys
# 로컬 Windows 콘솔 한글 출력용 (Vercel 등 reconfigure 미지원 환경에서는 건너뜀)
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import os
import traceback
from datetime import timedelta

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify, request, render_template

from config import (HOLDING_PERIODS, TARGET_RETURNS, PERIOD_NAMES, DEFAULT_PERIOD,
                    period_label_to_days, target_return_for)
from data_loader import MarketDataLoader
from indicators import IndicatorEngine
from analyzer import ConsultingInterpreter
from news_analyzer import NewsAnalyzer
from options_analyzer import OptionsAnalyzer
from probability_engine import ProbabilityEngine
from ml_model import ModelStore, load_universe, MARKET_TICKER

load_dotenv()

# ── 앱 초기화 ──────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder='templates')

# XGBoost 모델 로드 (train.py 산출물: models/xgb_{period}.json)
model_store = ModelStore(HOLDING_PERIODS)

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
news_analyzer = NewsAnalyzer(api_key=GEMINI_API_KEY)

# ── 분석 대상 종목 (universe.json 단일 소스) ───────────────────────────────
# 서빙 대상 종목 (모델은 universe_train.json 의 넓은 유니버스로 학습). 임의 티커 허용 시 Gemini 비용 남용 가능 → 목록 내 종목만 허용
_UNIVERSE = load_universe()
WATCHLIST = {t: v['name'] for t, v in _UNIVERSE.items()}
SECTORS = {t: v['sector'] for t, v in _UNIVERSE.items()}
DEFAULT_TICKER = next(iter(WATCHLIST))
CHART_DAYS = 180


class BadRequest(Exception):
    pass


# ── 헬퍼 함수 ─────────────────────────────────────────────────────────────
def _validate_ticker(raw) -> str:
    ticker = str(raw or '').strip().upper()
    if ticker not in WATCHLIST:
        raise BadRequest('지원하지 않는 종목입니다')
    return ticker


def _error(e: Exception):
    """사용자에게 보여도 되는 오류만 그대로 전달, 나머지는 서버 로그에만 남김"""
    if isinstance(e, BadRequest):
        return jsonify({'success': False, 'error': str(e)}), 400
    traceback.print_exc()
    if isinstance(e, ValueError):   # 데이터 없음 등 (data_loader 메시지)
        return jsonify({'success': False, 'error': str(e)}), 502
    return jsonify({'success': False, 'error': '분석 중 서버 오류가 발생했습니다'}), 500


def _load_indicators(ticker: str) -> pd.DataFrame:
    return IndicatorEngine.calculate_all(MarketDataLoader(ticker).fetch_data())


def _load_market():
    """KOSPI 일봉 (모델의 시장 피처용, 10분 캐시). 실패하면 None → 시장 피처는 결측 처리"""
    try:
        return MarketDataLoader(MARKET_TICKER).fetch_data()
    except Exception as e:
        print(f"[Server] KOSPI 로드 실패 (시장 피처 결측 처리): {e}")
        return None


def _check_macd_alert(df: pd.DataFrame) -> dict:
    """마지막 봉 기준 MACD 데드크로스/골든크로스 감지"""
    last = df.iloc[-1]
    macd, signal = float(last['MACD_12_26_9']), float(last['MACDs_12_26_9'])
    alert = {'type': 'none', 'triggered': False,
             'message': 'MACD > Signal (상승세)' if macd > signal else 'MACD < Signal (하락세)'}
    if len(df) >= 2:
        prev = df.iloc[-2]
        prev_macd, prev_signal = float(prev['MACD_12_26_9']), float(prev['MACDs_12_26_9'])
        if prev_macd >= prev_signal and macd < signal:
            alert = {'type': 'sell', 'triggered': True, 'message': '⚠️ MACD 데드크로스 발생! 매도 시그널'}
        elif prev_macd <= prev_signal and macd > signal:
            alert = {'type': 'buy', 'triggered': True, 'message': '✅ MACD 골든크로스 발생! 매수 시그널'}
    alert.update(macd=round(macd, 4), signal=round(signal, 4))
    return alert


def _safe_float(val):
    """NaN/None 안전 float 변환"""
    if val is None:
        return None
    try:
        v = float(val)
        return None if np.isnan(v) or np.isinf(v) else v
    except Exception:
        return None


def _build_chart_data(df: pd.DataFrame) -> dict:
    """Plotly용 차트 데이터 직렬화"""
    cols = {
        'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close',
        'ema20': 'EMA_20', 'ema60': 'EMA_60', 'bbu': 'BBU_20_2.0', 'bbl': 'BBL_20_2.0',
        'macd': 'MACD_12_26_9', 'macd_signal': 'MACDs_12_26_9', 'macd_hist': 'MACDh_12_26_9',
    }
    out = {'dates': [d.strftime('%Y-%m-%d') for d in df.index]}
    for key, col in cols.items():
        out[key] = [_safe_float(v) for v in df[col]]
    return out


# ── 라우트 ─────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/watchlist')
def watchlist():
    """분석 가능 종목 목록"""
    return jsonify({'success': True, 'watchlist': WATCHLIST, 'sectors': SECTORS})


@app.route('/api/periods')
def get_periods():
    """고정 홀딩 기간(단기/장기)과 기간별 목표 수익률"""
    periods = [
        {'label': k, 'name': PERIOD_NAMES[k], 'days': v, 'target_return_pct': TARGET_RETURNS[k]}
        for k, v in HOLDING_PERIODS.items()
    ]
    return jsonify({'success': True, 'periods': periods})


@app.route('/api/model-info')
def model_info():
    """학습된 XGBoost 모델 메타데이터 (하이퍼파라미터, 분할 통계, 성능)"""
    return jsonify({'success': True, 'models': model_store.meta})


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """
    메인 분석 엔드포인트
    Input (JSON): { ticker, period_label: 'short' | 'long' }
    Output: 실현가능성%, 추천 점수·등급·이유, 뉴스/변동성 세부내역, 차트 데이터
    """
    try:
        data = request.get_json(silent=True) or {}
        ticker = _validate_ticker(data.get('ticker', DEFAULT_TICKER))
        period_label = str(data.get('period_label', DEFAULT_PERIOD))
        if period_label not in HOLDING_PERIODS:
            raise BadRequest(f"기간은 {list(HOLDING_PERIODS)} 중 하나여야 합니다")

        holding_days = period_label_to_days(period_label)     # 거래일
        target_return_pct = target_return_for(period_label)
        stock_name = WATCHLIST[ticker]
        print(f"\n[Server] 분석 요청: {ticker} | {PERIOD_NAMES[period_label]}({holding_days}거래일) | 목표수익 {target_return_pct}%")

        # 1. 데이터 + 지표 (3년, 캐시)
        full_df = _load_indicators(ticker)
        latest = full_df.iloc[-1]
        chart_df = full_df[full_df.index >= full_df.index[-1] - timedelta(days=CHART_DAYS)]

        # 2. 뉴스 (Gemini, 종목별 캐시)
        news_result = news_analyzer.get_news_score(ticker, stock_name)

        # 3. 기준 확률 HV₀ (변동성 기반, 방향 가정 없음)
        hist_result = OptionsAnalyzer.get_historical_vol_probability(full_df, target_return_pct, holding_days)

        # 4. 보조지표 모델 (모델 없으면 None → 기준 확률만 사용)
        ml_result = model_store.predict(period_label, full_df, _load_market())

        # 5. 확률 종합
        result = ProbabilityEngine.compute(
            ml=ml_result,
            hv0=hist_result.get('probability'),
            news=news_result,
            target_return_pct=target_return_pct,
            holding_days=holding_days,
            period_name=PERIOD_NAMES[period_label],
            latest_row=latest,
            hist_vol_pct=hist_result.get('historical_vol'),
        )

        interpretation = ConsultingInterpreter.interpret(latest, target_return_pct=target_return_pct)

        return jsonify({
            'success': True,
            'ticker': ticker,
            'stock_name': stock_name,
            'current_price': round(float(latest['Close']), 2),
            'analysis_date': latest.name.strftime('%Y-%m-%d'),
            # 핵심 결과
            'feasibility_pct': result['feasibility_pct'],
            'recommendation': result['recommendation'],
            'news_category': result['news_category'],
            'period_label': period_label,
            'holding_days': holding_days,
            'target_return_pct': target_return_pct,
            'holding_period': result['holding_period'],
            'components': result['components'],
            'news': {
                'score': round(news_result.get('score', 0.0), 3),
                'reason': news_result.get('reason', ''),
                'key_topics': news_result.get('key_topics', []),
                'sentiment': news_result.get('sentiment', '중립'),
                'headlines': news_result.get('recent_headlines', []),
                'count': news_result.get('news_count', 0),
            },
            'volatility': {
                'hist_vol_pct': hist_result.get('historical_vol'),
                'data_source': hist_result.get('data_source', ''),
            },
            'macd_alert': _check_macd_alert(full_df),
            'interpretation': {
                'text': interpretation['text'],
                'target_price': round(interpretation['target_price'], 2),
                'stop_loss': round(interpretation['stop_loss'], 2),
                'expected_return_pct': round(interpretation['expected_return_pct'], 2),
                'risk_pct': round(interpretation['risk_pct'], 2),
            },
            'chart_data': _build_chart_data(chart_df),
        })
    except Exception as e:
        return _error(e)


@app.route('/api/macd-alert/<path:ticker>')
def macd_alert_check(ticker):
    """MACD 알림 폴링용 경량 엔드포인트 (프론트에서 5분마다 호출, 시세는 캐시 사용)"""
    try:
        ticker = _validate_ticker(ticker)
        return jsonify({'success': True, 'ticker': ticker, 'macd_alert': _check_macd_alert(_load_indicators(ticker))})
    except Exception as e:
        return _error(e)


if __name__ == '__main__':
    debug = os.environ.get('FLASK_DEBUG', '').lower() in ('1', 'true')
    host = os.environ.get('HOST', '127.0.0.1')   # 외부 공개가 필요할 때만 HOST=0.0.0.0
    port = int(os.environ.get('PORT', 5000))
    print("=" * 55)
    print("  📈 퀀트 트레이딩 분석 서버 시작")
    print(f"  접속 URL: http://localhost:{port}  (bind {host}, debug={debug})")
    print("  GEMINI_API_KEY:", "✓ 설정됨" if GEMINI_API_KEY else "✗ 미설정 (뉴스 분석 비활성화)")
    for k, v in HOLDING_PERIODS.items():
        print(f"  {PERIOD_NAMES[k]}: {v}거래일 / 목표 수익률 {TARGET_RETURNS[k]}%")
    print("=" * 55)
    app.run(host=host, port=port, debug=debug)
