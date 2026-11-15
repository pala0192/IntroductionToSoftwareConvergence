"""
build_universe.py - 학습용 종목 유니버스 생성 (KOSPI200 + KOSDAQ150) → universe_train.json

사용:
    python build_universe.py                      # auto: KRX 계정 있으면 pit, 없으면 current
    python build_universe.py --mode pit           # 시점별 구성종목 (pykrx, KRX_ID/KRX_PW 환경변수 필요)
    python build_universe.py --mode current       # 현재 구성종목 (네이버 금융, 생존 편향 있음)

모드
- pit (point-in-time, 권장):
    매년 1월·7월 첫 거래일(6월·12월 정기변경 직후)의 KOSPI200(1028)·KOSDAQ150(2203) 구성종목을 수집.
    스냅샷 k 의 구성은 [date_k, date_{k+1}) 동안 유효한 것으로 보고, 연속 편입 구간을 하나로 합침.
    (정기변경 사이의 수시 변경은 반기 단위로 근사)
- current (대체 경로):
    KOSPI200 = 네이버 금융 지수 편입종목, KOSDAQ150 ≈ KOSDAQ 시가총액 상위 150 보통주 (네이버에 KOSDAQ150 편입 목록 없음).
    편입 구간 = 전 기간. 과거 상장폐지·지수 제외 종목이 빠지므로 생존 편향이 있음 → meta.survivorship_biased = true
- 서빙 종목(universe.json)은 항상 포함 (source='serving', 지수에 없으면 전 기간 편입으로 간주)

출력 형식:
{
  "meta": {"mode", "survivorship_biased", "start", "end", "built_at", "sources", "counts"},
  "tickers": {"005930.KS": {"name", "code", "market", "member": [["2016-01-04", "2026-11-13"], ...], "source"}}
}
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import pandas as pd
import requests

from ml_model import load_universe

OUT_FILE = 'universe_train.json'
NAVER = 'https://m.stock.naver.com/api'
HEADERS = {'User-Agent': 'Mozilla/5.0'}
KOSDAQ_TOP = 150
INDEXES = {'1028': ('KOSPI200', 'KOSPI', '.KS'), '2203': ('KOSDAQ150', 'KOSDAQ', '.KQ')}


# ── current: 네이버 금융 ─────────────────────────────────────────────────
def _naver_json(url: str):
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=10)
            if r.status_code == 200 and r.text.strip():
                return r.json()
            return None
        except Exception:
            time.sleep(1 + attempt)
    return None


def naver_kospi200() -> list:
    out, page = [], 1
    while page < 30:
        j = _naver_json(f'{NAVER}/index/KPI200/enrollStocks?page={page}&pageSize=50')
        if not j:
            break
        out += j
        page += 1
    return [(s['itemCode'], s['stockName']) for s in out if s.get('stockEndType') == 'stock']


def naver_kosdaq_top(n: int) -> list:
    out, page = [], 1
    while len(out) < n and page < 20:
        j = _naver_json(f'{NAVER}/stocks/marketValue/KOSDAQ?page={page}&pageSize=100')
        if not j or not j.get('stocks'):
            break
        for s in j['stocks']:
            # 보통주만 (ETF·ETN·스팩 제외)
            if s.get('stockEndType') == 'stock' and '스팩' not in s['stockName']:
                out.append((s['itemCode'], s['stockName']))
        page += 1
    return out[:n]


def build_current(start: str, end: str) -> tuple:
    k200 = naver_kospi200()
    q150 = naver_kosdaq_top(KOSDAQ_TOP)
    if len(k200) < 150 or len(q150) < 100:
        raise RuntimeError(f"네이버 구성종목 수집 실패 (KOSPI200 {len(k200)}개, KOSDAQ {len(q150)}개)")
    tickers = {}
    for code, name in k200:
        tickers[f'{code}.KS'] = {'name': name, 'code': code, 'market': 'KOSPI',
                                 'member': [[start, end]], 'source': 'KOSPI200(current)'}
    for code, name in q150:
        tickers.setdefault(f'{code}.KQ', {'name': name, 'code': code, 'market': 'KOSDAQ',
                                          'member': [[start, end]], 'source': 'KOSDAQ_top150(current)'})
    meta = {'mode': 'current', 'survivorship_biased': True,
            'sources': {'KOSPI200': 'naver enrollStocks (현재 편입)',
                        'KOSDAQ150': f'naver KOSDAQ 시가총액 상위 {KOSDAQ_TOP} 보통주 (KOSDAQ150 근사)'},
            'counts': {'KOSPI200': len(k200), 'KOSDAQ150_approx': len(q150)}}
    return tickers, meta


# ── pit: pykrx (KRX 로그인 필요) ───────────────────────────────────────────
def rebalance_dates(start: str, end: str) -> list:
    """1월·7월 첫 영업일 후보 (실제 영업일은 pykrx 로 보정)"""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    out = []
    for y in range(s.year, e.year + 1):
        for m in (1, 7):
            d = pd.Timestamp(year=y, month=m, day=2)
            if d <= e and d >= s - pd.DateOffset(months=6):
                out.append(d)
    return out


def build_pit(start: str, end: str) -> tuple:
    if not (os.environ.get('KRX_ID') and os.environ.get('KRX_PW')):
        raise RuntimeError("KRX_ID / KRX_PW 환경변수가 없습니다 (data.krx.co.kr 계정 필요)")
    from pykrx import stock

    snaps = []   # (date, {yf_ticker: (code, name, market, index)})
    for d in rebalance_dates(start, end):
        day = stock.get_nearest_business_day_in_a_week(d.strftime('%Y%m%d'), prev=False)
        members = {}
        for idx, (idx_name, market, suffix) in INDEXES.items():
            codes = stock.get_index_portfolio_deposit_file(idx, day)
            if not len(codes):
                raise RuntimeError(f"{idx_name} {day} 구성종목이 비어 있음 (로그인/권한 확인)")
            for code in codes:
                members[f'{code}{suffix}'] = (code, market, idx_name)
            time.sleep(0.3)
        snaps.append((pd.Timestamp(day), members))
        print(f"  {day}: KOSPI200+KOSDAQ150 {len(members)}종목")

    tickers = {}
    for k, (d, members) in enumerate(snaps):
        valid_from = max(d, pd.Timestamp(start))
        valid_to = (snaps[k + 1][0] - pd.Timedelta(days=1)) if k + 1 < len(snaps) else pd.Timestamp(end)
        if valid_to < pd.Timestamp(start):
            continue
        for t, (code, market, idx_name) in members.items():
            info = tickers.setdefault(t, {'name': None, 'code': code, 'market': market,
                                          'member': [], 'source': f'{idx_name}(pit)'})
            iv = [valid_from.strftime('%Y-%m-%d'), valid_to.strftime('%Y-%m-%d')]
            # 직전 구간과 이어지면 병합
            if info['member'] and pd.Timestamp(info['member'][-1][1]) + pd.Timedelta(days=1) >= valid_from:
                info['member'][-1][1] = iv[1]
            else:
                info['member'].append(iv)
    for t, info in tickers.items():
        try:
            info['name'] = stock.get_market_ticker_name(info['code'])
        except Exception:
            info['name'] = info['code']
    meta = {'mode': 'pit', 'survivorship_biased': False,
            'sources': {'KOSPI200': 'pykrx 1028 (반기 스냅샷)', 'KOSDAQ150': 'pykrx 2203 (반기 스냅샷)'},
            'counts': {'snapshots': len(snaps), 'unique_tickers': len(tickers)},
            'note': '상장폐지 종목 중 yfinance 에 시세가 없는 종목은 build_snapshot 단계에서 빠짐 (manifest 의 failed 참고)'}
    return tickers, meta


# ── main ─────────────────────────────────────────────────────────────────
def add_serving(tickers: dict, start: str, end: str) -> int:
    added = 0
    for t, info in load_universe().items():
        if t not in tickers:
            tickers[t] = {'name': info['name'], 'code': t.split('.')[0],
                          'market': 'KOSDAQ' if t.endswith('.KQ') else 'KOSPI',
                          'member': [[start, end]], 'source': 'serving'}
            added += 1
    return added


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['auto', 'pit', 'current'], default='auto')
    ap.add_argument('--start', default='2016-01-01')
    ap.add_argument('--end', default='2026-11-13')
    ap.add_argument('--out', default=OUT_FILE)
    args = ap.parse_args()

    mode = args.mode
    if mode == 'auto':
        mode = 'pit' if (os.environ.get('KRX_ID') and os.environ.get('KRX_PW')) else 'current'
        if mode == 'current':
            print("⚠️ KRX_ID/KRX_PW 없음 → 현재 구성종목(current) 모드로 대체합니다. 생존 편향이 남습니다.")

    tickers, meta = (build_pit if mode == 'pit' else build_current)(args.start, args.end)
    n_serving = add_serving(tickers, args.start, args.end)
    meta.update({'start': args.start, 'end': args.end, 'built_at': datetime.now().isoformat(timespec='seconds'),
                 'n_tickers': len(tickers), 'serving_added_outside_index': n_serving})

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump({'meta': meta, 'tickers': tickers}, f, indent=1, ensure_ascii=False)
    by_src = pd.Series([v['source'] for v in tickers.values()]).value_counts().to_dict()
    print(f"✓ {args.out}: {len(tickers)}종목 (mode={mode}) {by_src}")
