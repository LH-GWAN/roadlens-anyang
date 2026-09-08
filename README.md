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
│   ├── raw/                    # 공공데이터 CSV 배치 위치 (+ 테스트용 sample_*.csv)
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

### 실제 공공데이터 사용

기본 상태에서는 `data/raw/sample_*.csv` (검증용 합성 데이터) 로 동작하며 화면 상단에 빨간 경고가 뜬다.
실제 데이터를 쓰려면 공공데이터포털에서 아래 두 파일을 내려받아 정해진 이름으로 저장한다. 코드 수정은 필요 없다.

| 데이터 | 출처 | 저장 위치 |
|---|---|---|
| 경기도 안양시_도로부속물 파손 현황 | https://www.data.go.kr/data/15096549/fileData.do | `data/raw/anyang_road_damage.csv` |
| 전국어린이보호구역표준데이터 | https://www.data.go.kr/data/15012891/standard.do | `data/raw/school_zones.csv` |

파일을 넣고 앱을 다시 실행하면 초록색 "실제 내려받은 공공데이터 파일" 표시로 바뀐다.
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
