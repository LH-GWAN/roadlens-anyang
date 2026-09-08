# RoadLens Anyang

안양시 도로 파손 이력과 도로 사진을 바탕으로 점검 우선순위와 순찰 경로를 제안하는 Streamlit 시제품.
2026년 안양시 공공데이터와 AI 활용 대학생 경진대회 출품작.

## 저장소 구조

```
roadlens-anyang/
├── app.py                      # Streamlit 앱 진입점 (6개 화면)
├── requirements.txt
├── .env.example                # 환경변수 양식 (복사해서 .env 로 사용)
├── data/
│   ├── raw/                    # 안양시 공공데이터 CSV 3개 (+ 테스트용 sample_*.csv)
│   └── processed/              # 실행 중 자동 생성되는 SQLite DB
├── models/
│   └── best.onnx               # 학습된 도로 파손 탐지 모델 (YOLOv8n, 입력 640×640)
├── notebooks/
│   └── 안양시 공공데이터.ipynb   # RDD2022 학습용 Colab 노트북
├── outputs/                    # 학습 지표(json/csv), 탐지 예시 이미지
├── src/
│   ├── config.py               # 경로, 가중치, 환경변수 읽기
│   ├── schemas.py              # 표준 열 이름, 열 별칭, 처리상태 정의
│   ├── data_loader.py          # CSV 로딩(BOM/CP949 대응), 기상청 API 호출
│   ├── preprocessing.py        # 좌표 검증, 제외 사유 기록, 중복 제거
│   ├── spatial_analysis.py     # 군집화, 밀도, 어린이보호구역 인접도, 거리행렬
│   ├── image_inference.py      # ONNX/YOLO 추론 (모델 없으면 자동 비활성)
│   ├── privacy.py              # 얼굴 모자이크 등 비식별 처리
│   ├── risk_score.py           # 점검 우선순위 점수 계산, 가중치 재정규화
│   ├── route_optimizer.py      # OR-Tools 기반 순찰 경로 최적화
│   └── repository.py           # SQLite 저장소 (AI 결과 / 담당자 확인 필드 분리)
└── tests/
    ├── test_preprocessing.py
    ├── test_risk_score.py
    ├── test_route_optimizer.py
    └── test_resilience.py      # 모델·API 키가 없어도 앱이 죽지 않는지 검증
```

## 앱 화면

| 화면 | 기능 |
|---|---|
| 1. 서비스 소개 | 서비스 목적, 데이터 기준일, 기능별 활성 상태 |
| 2. 과거 파손 이력 지도 | 유형·날짜·도로명 필터, 마커 클러스터 / 히트맵 / 밀집구간 레이어 |
| 3. 사진 분석 | 도로 사진 업로드 → 비식별 처리 → 파손 탐지 → 점검 건 접수 |
| 4. 점검 우선순위 | 밀집구간별 총점과 항목별 점수, 계산식 표시 |
| 5. 점검목록 및 순찰경로 | 선택한 지점의 방문 순서, 총거리, 지도 경로 |
| 6. 처리결과 기록 | 상태 변경(접수 → 점검예정 → 현장확인 → 보수필요 → 처리완료)과 이력 |

## 설치

Python 3.11 또는 3.12.

```bash
git clone https://github.com/LH-GWAN/roadlens-anyang.git
cd roadlens-anyang
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`models/best.onnx` 가 저장소에 포함되어 있어 별도 모델 다운로드 없이 바로 실행된다.
`.pt` 가중치를 쓰려면 `pip install ultralytics` 를 추가로 설치한다.

## 실행

```bash
streamlit run app.py
```

브라우저에서 `http://localhost:8501` 이 열린다.

### 사용 데이터

저장소에 실제 공공데이터 CSV 가 포함되어 있어 별도 다운로드 없이 바로 동작한다.

| 데이터 | 출처 | 파일 | 건수 |
|---|---|---|---|
| 경기도 안양시_도로위험물 현황 (포트홀·균열) | https://www.data.go.kr/data/15096553/fileData.do | `data/raw/anyang_road_hazard.csv` | 59,891 |
| 경기도 안양시_도로부속물 파손 현황 | https://www.data.go.kr/data/15096549/fileData.do | `data/raw/anyang_road_damage.csv` | 10,562 |
| 전국어린이보호구역표준데이터 | https://www.data.go.kr/data/15012891/standard.do | `data/raw/school_zones.csv` | 전국 14,652 중 안양시 62 사용 |

파손 이력 두 파일은 열 구조가 같아 하나로 합쳐서 쓴다(`source` 열로 출처 구분).
위험물유형은 피로균열·수직균열·수평균열·포트홀·도로수리훼손·쓰레기(도로위험물)와 노면표시훼손·시선유도봉파손(도로부속물)이다.

두 파일 모두 2021년 9~11월 수집분(데이터기준일자 2021-12-17)이며, 2026년 9월 기준 공공데이터포털과 경기데이터드림에 안양시 도로 파손 이력의 더 최신 개방 데이터는 없다.
안양시가 새 파일을 올리면 같은 이름으로 덮어쓰기만 하면 된다.

세 데이터 모두 공공누리 제1유형(출처표시) 이다. 파일을 지우면 `data/raw/sample_*.csv` (검증용 합성 데이터) 로 대체되며 화면에 샘플 데이터 안내가 표시된다.
열 이름에 BOM·공백·표기 차이가 있어도 자동으로 매핑한다.

### 환경변수 (선택)

`.env.example` 을 `.env` 로 복사해서 값을 채운다. 비워 두어도 앱은 동작한다.

| 변수 | 기본값 | 설명 |
|---|---|---|
| `KMA_SERVICE_KEY` | (빈 값) | 기상청 단기예보 인증키. 없으면 강수 정보를 화면에서 직접 입력 |
| `KMA_NX` / `KMA_NY` | 60 / 121 | 기상청 격자 좌표(안양시) |
| `MODEL_PATH` | (빈 값) | 비우면 `models/best.onnx` → `models/best.pt` 순으로 자동 탐색 |
| `DETECTION_CONF_THRESHOLD` | 0.25 | 탐지 신뢰도 임계값 |
| `FACE_BLUR_ENABLED` | true | 얼굴 모자이크 사용 여부 |
| `STORE_ORIGINAL_IMAGE` | false | 원본 사진 저장 여부 (기본은 비식별 사진만 저장) |
| `ROUTING_API_ENABLED` | false | 외부 도로망 API 연동 시에만 true |

### 테스트

```bash
pytest -q
```

## 모델 학습 (Google Colab)

노트북: `notebooks/안양시 공공데이터.ipynb`

1. https://figshare.com/articles/dataset/21431547 에서 RDD2022 ZIP 을 내려받아 본인 Google Drive 에 올린다.
2. Colab 에서 노트북을 열고 런타임을 GPU(T4) 로 설정한다.
3. `2. 실행 설정` 셀에서 `RDD2022_ZIP_PATHS` 를 본인 Drive 경로로 수정한다.
4. `RUN_MODE` 를 고르고 셀을 위에서 아래로 순서대로 실행한다.

| 모드 | 클래스별 이미지 상한 | epochs | 모델 | T4 기준 소요 |
|---|---|---|---|---|
| `smoke_test` | 40 | 3 | yolov8n | 수 분 |
| `prototype_train` | 1,500 | 30 | yolov8n | 30분~1시간 |
| `accuracy_train` | 제한 없음 | 60 | yolov8n | 약 3시간 |
| `full_train` | 제한 없음 | 150 | yolov8s | 여러 세션 |

5. 마지막 셀에서 내려받은 `best.onnx` 를 `models/best.onnx` 에 덮어쓴다.

학습 도중 Colab 사용 한도에 걸리면 11-C 셀로 체크포인트 번들을 내보내고, 다른 계정에서 11-D 셀로 이어서 학습할 수 있다.

현재 포함된 `best.onnx` 는 `prototype_train` 모드로 학습한 결과이며, 지표는 `outputs/rdd2022_prototype_train_metrics.json` 에 있다.

## 데이터 정제 규칙

`src/preprocessing.py` 에서 CSV 를 읽은 뒤 아래 순서로 정제한다. 제외된 행은 사유별 건수로 집계된다.

1. 위도·경도를 숫자로 변환한다. 비어 있거나 숫자가 아니면 제외한다.
2. 안양시 범위(위도 37.34~37.46, 경도 126.87~127.00)를 벗어난 좌표는 제외한다.
3. 수집날짜·데이터기준일자를 날짜형으로 변환하고, 위험물유형·도로명주소의 앞뒤 공백을 제거한다.
4. 같은 출처 데이터셋 안에서 분류번호가 같은 행은 중복으로 보고 첫 행만 남긴다(분류번호가 없으면 좌표·유형·수집날짜 기준).
5. 어린이보호구역은 전국 데이터에서 주소에 "안양"이 포함된 행만 남긴 뒤 같은 좌표 범위 검사를 적용한다.

현재 포함된 안양시 파손 CSV 두 개는 70,453건 전부 통과한다(제외 0건, 중복 0건).

## 점검 우선순위 계산식

`src/risk_score.py`. 항목별 점수를 0~100 으로 정규화한 뒤 가중합으로 총점(0~100)을 구한다.

| 항목 | 가중치 | 점수 산출 방법 |
|---|---|---|
| 과거 파손 이력 밀도 | 0.30 | 반경 R m 안의 과거 이력 건수 ÷ 기준 건수 × 100 (기준 건수 이상이면 100) |
| 사진상 파손 정도 | 0.25 | 유형 가중치×신뢰도 최댓값 60점 + 탐지 개수(최대 5개) 20점 + 파손 면적 비율(최대 30%) 20점 |
| 어린이보호구역 인접도 | 0.20 | (1 − 최근접 보호구역 거리 ÷ 기준 거리 500 m) × 100, 기준 거리 이상이면 0 |
| 도로구간 중요도 | 0.15 | 같은 노드링크(또는 도로명)에 누적된 이력 건수 ÷ 최대 건수 × 100 |
| 강수 영향 | 0.10 | 강수량 ÷ 30 mm × 100 (30 mm 이상이면 100) |

사진 유형 가중치는 포트홀(D40) 1.0, 거북등 균열(D20) 0.8, 종·횡방향 균열(D00·D10) 0.6 이다.

```
총점 = Σ (항목 점수 × 적용 가중치)
적용 가중치 = 기본 가중치 ÷ (사용 가능한 항목의 기본 가중치 합)
```

사진을 분석하지 않았거나 강수 정보가 없는 등 데이터가 없는 항목은 계산에서 빼고, 남은 항목의 가중치 합이 1 이 되도록 재정규화한다.
예를 들어 사진 점수가 없으면 나머지 네 항목의 가중치는 0.30/0.75, 0.20/0.75, 0.15/0.75, 0.10/0.75 로 조정된다.
사용 가능한 항목이 하나도 없으면 총점 0 으로 표시하며, 이는 "위험도가 낮다"가 아니라 "판단 근거가 없다"는 뜻이다.

도로구간 중요도는 도로등급·교통량 데이터가 없어 해당 구간에 누적된 과거 이력 건수를 대리지표로 쓴다.
순찰 경로는 외부 도로망 API 를 연동하지 않은 상태에서는 직선(Haversine) 거리 기반 근사 경로다.
