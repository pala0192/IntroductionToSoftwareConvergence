"""
data_loader.py - yfinance 일봉 수집 (+ 서버용 메모리 캐시)
"""
import os
import shutil
import tempfile
import threading
import time
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

KST = ZoneInfo('Asia/Seoul')
MARKET_SETTLED = dtime(16, 0)   # KRX 15:30 마감 → 16:00 이후 당일 봉 확정으로 간주
CACHE_TTL_SEC = 600             # 같은 종목·기간 재요청 시 10분간 재사용
OHLCV = ['Open', 'High', 'Low', 'Close', 'Volume']

_cache = {}
_cache_lock = threading.Lock()


def _ascii_ca_bundle():
    """
    yfinance 내부 curl_cffi 는 CA 파일 경로에 한글 등 비ASCII 문자가 있으면 읽지 못함 (curl error 77).
    (예: C:\\Users\\...\\문서\\...\\.venv) → ASCII 경로로 복사한 사본을 사용. 문제 없으면 None.
    """
    import certifi
    src = certifi.where()
    if src.isascii():
        return None
    for d in (tempfile.gettempdir(), os.environ.get('PUBLIC', r'C:\Users\Public')):
        if d and d.isascii() and os.path.isdir(d):
            dst = os.path.join(d, 'qaunt_cacert.pem')
            if not os.path.exists(dst) or os.path.getmtime(dst) < os.path.getmtime(src):
                shutil.copyfile(src, dst)
            return dst
    return None


_CA_BUNDLE = _ascii_ca_bundle()


def _session():
    if _CA_BUNDLE is None:
        return None   # yfinance 기본 세션
    from curl_cffi import requests as curl_requests
    return curl_requests.Session(impersonate='chrome', verify=_CA_BUNDLE)


def _drop_incomplete_bar(df: pd.DataFrame) -> pd.DataFrame:
    """장중(또는 마감 직후) 미완성 당일 봉 제거 — 학습 데이터와 분포를 맞추기 위함"""
    now = datetime.now(KST)
    if len(df) and df.index[-1].date() == now.date() and now.time() < MARKET_SETTLED:
        return df.iloc[:-1]
    return df


class MarketDataLoader:
    def __init__(self, ticker):
        self.ticker = ticker

    def fetch_data(self, period: str = "3y", use_cache: bool = True) -> pd.DataFrame:
        """지정 기간의 일봉 데이터 (서버: 3y + 캐시, 학습: 10y)"""
        key = (self.ticker, period)
        if use_cache:
            with _cache_lock:
                hit = _cache.get(key)
            if hit and time.time() - hit[0] < CACHE_TTL_SEC:
                return hit[1].copy()

        print(f"[{self.ticker}] Data fetch: {period}")
        # yf.download 는 내부 공유 상태를 써서 스레드(동시 요청)에 안전하지 않음 → Ticker.history 사용
        df = yf.Ticker(self.ticker, session=_session()).history(period=period, auto_adjust=True)
        if df.empty:
            raise ValueError(f"{self.ticker} 시세 데이터를 찾을 수 없습니다")

        df = df[OHLCV].copy()
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.index.name = 'Date'
        df = _drop_incomplete_bar(df.dropna())

        if use_cache:
            with _cache_lock:
                now = time.time()
                for k in [k for k, (t, _) in _cache.items() if now - t >= CACHE_TTL_SEC]:
                    del _cache[k]
                _cache[key] = (now, df)
        return df.copy()

    def fetch_range(self, start: str, end: str) -> pd.DataFrame:
        """
        고정 기간 일봉 (학습 스냅샷용, 캐시 없음). start/end 는 'YYYY-MM-DD', end 포함.
        데이터가 없으면 빈 DataFrame (상장폐지·티커 불일치 등 → 호출측에서 판단)
        """
        end_excl = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')   # yfinance end 는 미포함
        df = yf.Ticker(self.ticker, session=_session()).history(start=start, end=end_excl, auto_adjust=True)
        if df.empty:
            return df
        df = df[OHLCV].copy()
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.index.name = 'Date'
        return _drop_incomplete_bar(df.dropna())
