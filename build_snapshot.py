"""
build_snapshot.py - 학습용 시세 스냅샷 (고정 기간, 재현 가능)

사용:
    python build_snapshot.py --universe universe.json --tag v1
    python build_snapshot.py --universe universe_train.json --tag v2-wide

산출물: data/snapshots/{tag}/
    prices/{ticker}.parquet   종목별 수정주가 일봉 (OHLCV)
    prices/^KS11.parquet      KOSPI 지수
    universe.json             사용한 유니버스 사본 (train/evaluate 는 이 사본을 읽음)
    manifest.json             기간·수집 시각·성공/실패 종목·유니버스 해시

- 실패 시 3회 재시도. 시장 이전 종목(.KS ↔ .KQ)은 반대 접미사로 재시도 (manifest 에 실제 사용 티커 기록)
- train.py / evaluate.py 는 --snapshot {tag} 로 이 데이터를 읽어 매번 같은 데이터셋을 만든다
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import pandas as pd

from data_loader import MarketDataLoader
from ml_model import MARKET_TICKER, load_train_universe

SNAPSHOT_ROOT = os.path.join('data', 'snapshots')
MIN_ROWS = 130          # 이보다 짧은 종목은 피처 워밍업도 못 채우므로 제외


def snapshot_dir(tag: str) -> str:
    return os.path.join(SNAPSHOT_ROOT, tag)


def _price_path(tag: str, ticker: str) -> str:
    return os.path.join(snapshot_dir(tag), 'prices', f'{ticker}.parquet')


def load_snapshot(tag: str) -> tuple:
    """→ (frames {ticker: OHLCV DataFrame}, market DataFrame, universe {ticker: {...}}, manifest)"""
    d = snapshot_dir(tag)
    with open(os.path.join(d, 'manifest.json'), encoding='utf-8') as f:
        manifest = json.load(f)
    universe, _ = load_train_universe(os.path.join(d, 'universe.json'))
    frames = {t: pd.read_parquet(_price_path(tag, t)) for t in manifest['ok']}
    market = pd.read_parquet(_price_path(tag, MARKET_TICKER))
    return frames, market, universe, manifest


def _fetch(ticker: str, start: str, end: str):
    """→ (df, 실제 사용 티커, 에러)"""
    candidates = [ticker]
    if ticker.endswith('.KS'):
        candidates.append(ticker[:-3] + '.KQ')
    elif ticker.endswith('.KQ'):
        candidates.append(ticker[:-3] + '.KS')
    err = None
    for cand in candidates:
        for attempt in range(3):
            try:
                df = MarketDataLoader(cand).fetch_range(start, end)
                if len(df):
                    return df, cand, None
                err = 'empty'
                break           # 빈 결과는 재시도해도 같음 → 다음 후보
            except Exception as e:
                err = repr(e)[:200]
                time.sleep(1.0 + attempt)
    return None, None, err


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--universe', default='universe.json')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--start', default='2016-01-01')
    ap.add_argument('--end', default='2026-11-13')
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()

    universe, umeta = load_train_universe(args.universe)
    d = snapshot_dir(args.tag)
    os.makedirs(os.path.join(d, 'prices'), exist_ok=True)
    shutil.copyfile(args.universe, os.path.join(d, 'universe.json'))
    with open(args.universe, 'rb') as f:
        uhash = hashlib.sha256(f.read()).hexdigest()[:16]

    print(f"📦 스냅샷 {args.tag}: {len(universe)}종목 + KOSPI, {args.start} ~ {args.end}")
    mkt, _, merr = _fetch(MARKET_TICKER, args.start, args.end)
    if mkt is None:
        raise SystemExit(f"KOSPI 수집 실패: {merr}")
    mkt.to_parquet(_price_path(args.tag, MARKET_TICKER))

    ok, failed, used = [], {}, {}
    stats = {}

    def job(t):
        return t, *_fetch(t, args.start, args.end)

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for t, df, cand, err in ex.map(job, list(universe)):
            done += 1
            if df is None:
                failed[t] = err
            elif len(df) < MIN_ROWS:
                failed[t] = f'too_short({len(df)})'
            else:
                df.to_parquet(_price_path(args.tag, t))
                ok.append(t)
                stats[t] = {'rows': int(len(df)), 'from': str(df.index[0].date()), 'to': str(df.index[-1].date())}
                if cand != t:
                    used[t] = cand
            if done % 50 == 0:
                print(f"  {done}/{len(universe)} (성공 {len(ok)}, 실패 {len(failed)})")

    manifest = {
        'tag': args.tag, 'start': args.start, 'end': args.end,
        'fetched_at': datetime.now().isoformat(timespec='seconds'),
        'universe_file': args.universe, 'universe_sha256_16': uhash, 'universe_meta': umeta,
        'market': {'ticker': MARKET_TICKER, 'rows': int(len(mkt))},
        'ok': sorted(ok), 'failed': failed, 'ticker_used': used, 'stats': stats,
    }
    with open(os.path.join(d, 'manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
    print(f"✓ 성공 {len(ok)} / 실패 {len(failed)} → {d}")
    if failed:
        print(f"  실패 예: {dict(list(failed.items())[:10])}")
