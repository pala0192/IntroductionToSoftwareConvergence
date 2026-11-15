"""
news_analyzer.py - 한국 주식 뉴스 수집 및 Gemini LLM 감성 분석
"""
# pyrefly: ignore [missing-import]
import feedparser
import html
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from zoneinfo import ZoneInfo

import requests

from google import genai
from google.genai import types


DEFAULT_GEMINI_MODEL = 'gemini-3.5-flash'   # 3.8-flash 는 504/503 지속 (2026-11-14 확인)
CACHE_TTL_OK_SEC = 3600       # 분석 성공 결과: 1시간 재사용 (Gemini 비용/지연 절감)
CACHE_TTL_FAIL_SEC = 300      # 분석 실패 결과: 5분 후 재시도
USER_AGENT = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
              '(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36')


KST = ZoneInfo('Asia/Seoul')


def _clean(text: str) -> str:
    """HTML 엔티티 복원 → 태그 제거 (순서 중요: 반대로 하면 &lt;script&gt; 가 실제 태그로 남음)"""
    return re.sub(r'<[^>]*>', '', html.unescape(text or '')).strip()


def _title_key(title: str) -> str:
    """중복 판정용: Google 제목 끝의 ' - 언론사' 제거, 문자·숫자만 남김"""
    t = re.sub(r'\s+-\s+[^-]{1,30}$', '', title)
    return re.sub(r'[^0-9A-Za-z가-힣]', '', t)[:40]


def _parse_naver_dt(s: str):
    """'202611151456' (KST) → aware datetime"""
    try:
        return datetime.strptime(s, '%Y%m%d%H%M').replace(tzinfo=KST)
    except (ValueError, TypeError):
        return None


def _parse_rss_dt(entry):
    """RSS pubDate (RFC 822) → KST aware datetime"""
    try:
        dt = parsedate_to_datetime(entry.get('published', ''))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(KST)
    except (TypeError, ValueError, IndexError):
        return None


class NewsAnalyzer:
    ARTICLES_TO_ANALYZE = 3       # Gemini로 개별 분석할 기사 수 (병렬 호출)
    MIN_NEWS = 5                  # 이보다 적으면 다음 소스로 보충
    MAX_NEWS = 10                 # 반환 기사 수 상한 (최신순)
    MAX_NEWS_AGE_DAYS = 30        # 이보다 오래된 기사는 현재 판단에서 제외
    GEMINI_TIMEOUT_MS = 30_000   # 호출당 최대 대기 시간 — 초과 시 해당 기사는 실패 처리
    GEMINI_ATTEMPTS = 2           # 503 등 일시 오류 시 재시도 포함 총 시도 횟수

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get('GEMINI_API_KEY', '')
        self.model = os.environ.get('GEMINI_MODEL', DEFAULT_GEMINI_MODEL)

        self.client = None
        if self.api_key:
            try:
                # 타임아웃·재시도 상한이 없으면 Gemini 과부하(503) 시 요청이 수 분간 멈춤
                self.client = genai.Client(
                    api_key=self.api_key,
                    http_options=types.HttpOptions(
                        timeout=self.GEMINI_TIMEOUT_MS,
                        retry_options=types.HttpRetryOptions(attempts=self.GEMINI_ATTEMPTS),
                    ),
                )
                print(f"[NewsAnalyzer] Gemini API 연결 성공 (모델: {self.model})")
            except Exception as e:
                print(f"[NewsAnalyzer] Gemini API 연결 실패: {e}")
        else:
            print("[NewsAnalyzer] ⚠️ GEMINI_API_KEY 없음 → 뉴스 감성 분석 비활성화")

        self._cache = {}          # ticker → (만료 시각, 결과)
        self._cache_lock = threading.Lock()

    def fetch_news(self, ticker: str, company_name: str) -> list:
        """
        키 없는 무료 소스로 한국어 뉴스 수집 → 중복 제거 → 최신순 (오래된 기사 제외)
        1) 네이버 증권 종목뉴스 (종목코드 기준, 회사명 언급 기사만)
        2) Google News RSS  (1이 부족할 때 보충)
        3) Bing News RSS    (그래도 부족할 때 보충)
        """
        # 종목 태그만 걸린 시황 기사 제외: 회사명이 제목·요약에 있는 기사만
        name_key = company_name.replace(' ', '')
        items = [it for it in self._fetch_naver_stock(ticker)
                 if name_key in (it['title'] + it['summary']).replace(' ', '')]
        for source in (self._fetch_google, self._fetch_bing):
            if len(items) >= self.MIN_NEWS:
                break
            items += source(company_name)

        cutoff = datetime.now(KST) - timedelta(days=self.MAX_NEWS_AGE_DAYS)
        seen, out = set(), []
        for it in sorted(items, key=lambda x: x['_dt'], reverse=True):
            key = _title_key(it['title'])
            if key in seen or it['_dt'] < cutoff:
                continue
            seen.add(key)
            it['published'] = it.pop('_dt').strftime('%Y-%m-%d %H:%M')
            out.append(it)
        out = out[:self.MAX_NEWS]
        by_src = {s: sum(1 for x in out if x['source'] == s) for s in {x['source'] for x in out}}
        print(f"[NewsAnalyzer] {ticker}({company_name}) 뉴스 {len(out)}건 수집 {by_src}")
        return out

    @staticmethod
    def _fetch_naver_stock(ticker: str) -> list:
        code = ticker.split('.')[0]
        url = f"https://m.stock.naver.com/api/news/stock/{code}"
        try:
            resp = requests.get(url, params={'pageSize': 20, 'page': 1},
                                headers={'User-Agent': USER_AGENT}, timeout=5)
            resp.raise_for_status()
            out = []
            for group in resp.json() or []:
                for it in group.get('items', [])[:1]:          # 묶음 기사는 대표 1건만
                    title = _clean(it.get('titleFull') or it.get('title', ''))
                    dt = _parse_naver_dt(it.get('datetime', ''))
                    if title and dt:
                        out.append({'title': title, 'summary': _clean(it.get('body', ''))[:300],
                                    'source': 'naver_stock', 'press': it.get('officeName', ''), '_dt': dt})
            return out
        except Exception as e:
            print(f"[NewsAnalyzer] 네이버 증권 뉴스 오류: {e}")
            return []

    @staticmethod
    def _fetch_rss(url: str, params: dict, source: str, limit: int = 10) -> list:
        try:
            resp = requests.get(url, params=params, headers={'User-Agent': USER_AGENT}, timeout=5)
            resp.raise_for_status()
            out = []
            for e in feedparser.parse(resp.content).entries[:limit]:
                title = _clean(e.get('title', ''))
                dt = _parse_rss_dt(e)
                if title and dt:
                    out.append({'title': title, 'summary': _clean(e.get('summary', ''))[:300],
                                'source': source, 'press': '', '_dt': dt})
            return out
        except Exception as e:
            print(f"[NewsAnalyzer] {source} 오류: {e}")
            return []

    def _fetch_google(self, company_name: str) -> list:
        return self._fetch_rss('https://news.google.com/rss/search',
                               {'q': f'{company_name} 주식 when:{self.MAX_NEWS_AGE_DAYS}d',
                                'hl': 'ko', 'gl': 'KR', 'ceid': 'KR:ko'},
                               'google_news')

    def _fetch_bing(self, company_name: str) -> list:
        return self._fetch_rss('https://www.bing.com/news/search',
                               {'q': f'{company_name} 주식', 'format': 'rss', 'setlang': 'ko-KR', 'cc': 'KR'},
                               'bing_news')

    def analyze_sentiment(self, news_items: list, ticker: str, company_name: str) -> dict:
        """Gemini로 뉴스 감성 점수 개별 산출 및 평균 계산"""
        if not self.client or not news_items:
            return {'score': 0.0, 'reason': 'Gemini 미연결 또는 뉴스 없음', 'key_topics': [],
                    'analyzed': False, 'article_reasons': []}

        total_score = 0.0
        analyzed_count = 0
        all_key_topics = []
        article_reasons = []

        # 기사 1건이 판정을 좌우하지 않도록 상위 N건을 병렬로 개별 분석 후 평균
        target_news = news_items[:self.ARTICLES_TO_ANALYZE]
        print(f"[NewsAnalyzer] {ticker} - {len(target_news)}건의 뉴스 개별 분석 시작...")

        with ThreadPoolExecutor(max_workers=len(target_news)) as pool:
            results = list(pool.map(lambda it: self._analyze_article(it, ticker, company_name), target_news))

        for item, result in zip(target_news, results):
            if result is None:
                continue
            score = result['score']
            total_score += score
            analyzed_count += 1
            all_key_topics.extend(result.get('key_topics', []))
            article_reasons.append({
                'title': item['title'],
                'score': round(score, 2),
                'reason': str(result.get('reason', '')).strip(),
            })

        if analyzed_count == 0:
            return {'score': 0.0, 'reason': '모든 개별 기사 분석 실패', 'key_topics': [],
                    'analyzed': False, 'article_reasons': []}

        avg_score = total_score / analyzed_count
        avg_score = max(-1.0, min(1.0, avg_score))

        final_sentiment = "긍정" if avg_score >= 0.2 else "부정" if avg_score <= -0.2 else "중립"
        unique_topics = list(set(all_key_topics))[:5]

        final_result = {
            'score': avg_score,
            'reason': f"{analyzed_count}건의 개별 뉴스 분석 후 평균 반영",
            'key_topics': unique_topics,
            'sentiment': final_sentiment,
            'analyzed': True,
            'article_reasons': article_reasons,
        }

        print(f"[NewsAnalyzer] {ticker} 최종 평균 감성 점수: {avg_score:.2f}")
        return final_result

    def _analyze_article(self, item: dict, ticker: str, company_name: str):
        """기사 1건 Gemini 분석 → {'score', 'reason', 'key_topics'} 또는 실패 시 None"""
        prompt = f"""당신은 주식 투자 전문 애널리스트입니다.
아래는 '{company_name}({ticker})' 종목에 관한 단일 뉴스 기사입니다.
이 뉴스가 단기 주가에 미치는 영향을 분석해주세요.

제목: {item['title']}
내용: {item.get('summary', '')}

반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트 없이 JSON만:
{{
  "score": 0.0,
  "reason": "판단 근거 한 줄 요약",
  "key_topics": ["키워드1", "키워드2"],
  "sentiment": "긍정/부정/중립"
}}

score 기준: -1.0(매우 부정) ~ 0.0(중립) ~ +1.0(매우 긍정)"""

        try:
            response = self.client.models.generate_content(model=self.model, contents=prompt)
            text = response.text.strip()
            if '```json' in text:
                text = text.split('```json')[1].split('```')[0].strip()
            elif '```' in text:
                text = text.split('```')[1].split('```')[0].strip()
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                text = match.group()

            result = json.loads(text)
            score = max(-1.0, min(1.0, float(result.get('score', 0.0))))
            print(f"  -> 개별 분석 완료: score={score:.2f} | {item['title'][:20]}...")
            return {
                'score': score,
                'reason': result.get('reason', ''),
                'key_topics': result.get('key_topics', []),
            }
        except Exception as e:
            print(f"  -> 개별 분석 실패: {str(e)[:120]}")
            return None

    def get_news_score(self, ticker: str, company_name: str) -> dict:
        """전체 뉴스 분석 파이프라인 (종목별 캐시: 성공 1시간 / 실패 5분)"""
        now = time.time()
        with self._cache_lock:
            for k in [k for k, (exp, _) in self._cache.items() if exp <= now]:
                del self._cache[k]
            hit = self._cache.get(ticker)
        if hit:
            print(f"[NewsAnalyzer] 캐시 히트: {ticker}")
            return hit[1]

        news_items = self.fetch_news(ticker, company_name)
        sentiment = self.analyze_sentiment(news_items, ticker, company_name)
        result = {
            'score': sentiment.get('score', 0.0),
            'reason': sentiment.get('reason', ''),
            'key_topics': sentiment.get('key_topics', []),
            'sentiment': sentiment.get('sentiment', '중립'),
            'analyzed': sentiment.get('analyzed', False),
            'article_reasons': sentiment.get('article_reasons', []),
            'news_count': len(news_items),
            'recent_headlines': [item['title'] for item in news_items[:4]]
        }
        ttl = CACHE_TTL_OK_SEC if result['analyzed'] else CACHE_TTL_FAIL_SEC
        with self._cache_lock:
            self._cache[ticker] = (time.time() + ttl, result)
        return result
