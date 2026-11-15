# IntroductionToSoftwareConvergence

Python 기반 퀀트 트레이딩 분석 플랫폼.

- Flask REST API 서버 (`server.py`) + 단일 페이지 UI (`templates/index.html`)
- yfinance 시세 수집(10분 캐시), `ta` 기반 기술 지표 계산
- **기준 확률 HV₀** (`options_analyzer.py`): 최근 2년 변동성으로 목표 도달 확률 (방향 가정 없음)
- **XGBoost 통합 모델** (`train.py`, `ml_model.py`): KOSPI200+KOSDAQ150 학습 유니버스(`universe_train.json`)
  고정 스냅샷(2016-01 ~ 2026-11), 기간(short/long)별. 서빙은 `universe.json` 25종목
  - 라벨: 같은 날 종목들 중 위험 조정 수익률(로그수익률 / σ₆₀√N) 상위 30%
  - 피처 36개: 기술 지표 23개 + 자기 이력 정규화·52주 위치·KOSPI 시장 피처
  - 검증: walk-forward(확장 창, fold마다 purge), 최종 모델은 전체 데이터로 재학습
- **결합** (`probability_engine.py`): walk-forward OOF 예측으로 적합한 보정식
  `logit(실현가능성) = a + b·logit(HV₀) + c·(logit(p_model) − logit(base_rate))`
  - 추천 점수 = 50 + 50 × (같은 변동성 대비 모델 우위) ± 뉴스
- Gemini 뉴스 감성 점수(종목별 캐시). 뉴스는 키 없는 무료 소스에서 수집:
  네이버 증권 종목뉴스(회사명 언급 기사만) → 부족하면 Google News RSS → Bing News RSS, 최근 30일·최신순
- **평가** (`evaluate.py`): XGBoost vs 로지스틱 회귀, 보조지표 성능 / 총 성능, 25종목 및 섹터 대표 5종목

## 실행

```bash
pip install -r requirements.txt                 # 서버 실행용 (Vercel 배포도 이 파일 사용)
pip install -r requirements-train.txt           # 학습·평가·탐색용 (pyarrow, optuna)
python build_universe.py                                          # 학습 유니버스 → universe_train.json
python build_snapshot.py --universe universe_train.json --tag v2-wide   # 시세 스냅샷 → data/snapshots/v2-wide
python train.py --snapshot v2-wide                                # walk-forward 학습 + OOF 보정식 (models/)
python evaluate.py --snapshot v2-wide --tag wf_wide               # 성능 평가 → reports/evaluation_wf_wide.json
python server.py                                                  # http://localhost:5000
```

`.env` 환경변수:

| 변수 | 설명 |
|---|---|
| `GEMINI_API_KEY` | 뉴스 감성 분석 (없으면 뉴스 보정 없이 동작) |
| `GEMINI_MODEL` | 기본 `gemini-3.5-flash` |
| `FLASK_DEBUG` | `1`이면 디버그 모드 (외부 공개 시 사용 금지) |
| `HOST`, `PORT` | 기본 `127.0.0.1:5000` |

## API

| 엔드포인트 | 설명 |
|---|---|
| `GET /api/watchlist` | 분석 가능 종목 (universe.json 내 종목만 허용) |
| `GET /api/periods` | 기간별 거래일 수 / 목표 수익률 |
| `POST /api/analyze` | `{ticker, period_label}` → 실현가능성, 추천 점수·등급·이유, 차트 |
| `GET /api/macd-alert/<ticker>` | MACD 크로스 알림 (UI에서 5분 폴링) |
| `GET /api/model-info` | 모델 메타데이터 |

## 모델 학습

| 파일 | 내용 |
|---|---|
| `universe.json` | 서빙 대상 25종목 (서버) |
| `universe_train.json` | 학습 유니버스 (`build_universe.py` 생성, 종목별 지수 편입 구간 포함) |
| `data/snapshots/{tag}/` | 고정 기간 시세 parquet + `manifest.json` (기간, 성공/실패 종목) + 유니버스 사본 |
| `models/xgb_params.json` | 하이퍼파라미터 (직접 수정, `early_stopping_rounds`는 fold 학습에만 적용) |
| `models/xgb_{short,long}.json` | 서빙 모델 (전체 데이터 재학습, 트리 수 = fold best_iteration 중앙값) |
| `models/metadata_{period}.json` | 피처, 보정식, 스냅샷, fold 요약, 학습 유니버스 정보 |
| `models/cv_report_{period}.json` | 데이터셋 크기와 fold별 train/val/test 행 수·기간·양성비율·best_iteration |
| `models/oof_{period}.parquet` | fold test 예측(OOF) — 보정식 적합·평가용 |

검증: walk-forward(확장 창). fold = `[train_core][purge N][inner_val 252일][purge N][test 252일]`,
첫 train_core 756거래일, 마지막 fold test는 남은 날짜 전부. 단기 5 fold, 장기 4 fold.

학습 유니버스(`build_universe.py`):
- `--mode pit`: pykrx로 반기별 KOSPI200·KOSDAQ150 구성종목을 받아 시점별로 편입된 종목만 사용 (생존 편향 완화).
  `KRX_ID`/`KRX_PW` 환경변수(data.krx.co.kr 계정)가 필요합니다.
- `--mode current`: 네이버 금융의 현재 KOSPI200 + KOSDAQ 시가총액 상위 150 (생존 편향 있음). 기본값 `auto`는 KRX 계정이 없으면 이 모드를 씁니다.

하이퍼파라미터 탐색(`tune.py`, Optuna TPE):
`python tune.py --snapshot v2-wide --period short long --timeout 1800 --apply`
- 마지막 2개 fold는 튜닝에 쓰지 않음(holdout). 목적값 = 앞쪽 fold들의 test AUC 평균
- 탐색 대상에 `early_stopping_rounds`, `es_metric`(early stopping 기준 지표: logloss|auc) 포함, `n_estimators`는 2000 상한
- 학습 행 25% 표본, MedianPruner로 성적 나쁜 trial 중단, 첫 trial = 기존 설정
- 결과 `models/tuning/{period}_best.json`, `--apply` 시 `xgb_params.json` 갱신(최초 1회 `xgb_params.prev.json` 백업)
- 이후 `train.py`·`evaluate.py`를 다시 실행하면 평가에 holdout fold 전용 세트(④⑤)가 추가됨

`train.py` 옵션: `--snapshot TAG`, `--label target|relative` (기본 relative), `--features base|extended` (기본 extended),
`--calibrate-only` (저장된 OOF로 보정식만 재적합). 단계별 실험 모델은 `models_archive/`에 보관
(stage0~3: 기존 70/15/15 holdout, wf_v2wide: 현재 모델).
`analyzer.StrategyAnalyzer.compute_signal_scores` / `ml_model.build_features`는 모델 피처이므로
수정하면 재학습이 필요합니다.
