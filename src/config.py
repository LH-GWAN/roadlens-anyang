"""환경설정. 비밀정보는 코드에 두지 않고 .env 또는 환경변수에서만 읽는다."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # python-dotenv 는 선택 의존성처럼 동작하게 둔다.
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv(*_args, **_kwargs):  # type: ignore[misc]
        return False


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
ASSETS_DIR = PROJECT_ROOT / "assets"

# 사용자가 직접 내려받아 저장하는 원본 CSV 경로
DAMAGE_CSV_PATH = DATA_RAW_DIR / "anyang_road_damage.csv"
SCHOOL_ZONE_CSV_PATH = DATA_RAW_DIR / "school_zones.csv"

# 테스트/데모 전용 샘플 (실제 안양시 데이터가 아님을 화면에서 반드시 표기)
SAMPLE_DAMAGE_CSV_PATH = DATA_RAW_DIR / "sample_anyang_road_damage.csv"
SAMPLE_SCHOOL_ZONE_CSV_PATH = DATA_RAW_DIR / "sample_school_zones.csv"

DB_PATH = DATA_PROCESSED_DIR / "roadlens.sqlite3"
UPLOAD_IMAGE_DIR = DATA_PROCESSED_DIR / "images"

# 안양시 대략 행정구역 경계 (좌표 유효성 1차 검증용, 여유를 둔 bounding box)
ANYANG_BBOX = {
    "lat_min": 37.34,
    "lat_max": 37.46,
    "lon_min": 126.87,
    "lon_max": 127.00,
}
ANYANG_CENTER = (37.3943, 126.9568)  # 안양시청 부근, 지도 초기 중심

# 점검 우선순위 가중치 (프롬프트 정의값)
RISK_WEIGHTS: dict[str, float] = {
    "history_density": 0.30,   # 과거 파손 이력 밀도
    "photo_severity": 0.25,    # 사진상 파손 정도
    "school_zone": 0.20,       # 어린이보호구역 인접도
    "road_importance": 0.15,   # 도로구간 중요도
    "rainfall": 0.10,          # 강수 영향
}

RISK_COMPONENT_LABELS: dict[str, str] = {
    "history_density": "과거 파손 이력 밀도",
    "photo_severity": "사진상 파손 정도",
    "school_zone": "어린이보호구역 인접도",
    "road_importance": "도로구간 중요도",
    "rainfall": "강수 영향",
}

# 어린이보호구역 인접도 점수: 이 거리(m) 이상이면 0점
SCHOOL_ZONE_MAX_DISTANCE_M = 500.0

# RDD2022 기준 탐지 클래스 (모델이 실제로 학습된 클래스와 일치해야 함)
RDD2022_CLASSES = [
    "D00",  # 종방향 균열
    "D10",  # 횡방향 균열
    "D20",  # 거북등 균열
    "D40",  # 포트홀
]
RDD2022_CLASS_LABELS_KO = {
    "D00": "종방향 균열",
    "D10": "횡방향 균열",
    "D20": "거북등 균열",
    "D40": "포트홀",
}

# 개인정보 보유기간 안내 (화면 표시용)
PHOTO_RETENTION_NOTICE = (
    "업로드된 사진은 비식별 처리 후 시제품 시연 목적에 한해 보관하며, "
    "시연 종료 또는 대회 심사 종료 후 즉시 삭제합니다. "
    "위치정보는 점검 이력 관리 목적으로만 사용하고 개인 식별 정보와 결합하지 않습니다."
)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass
class AppConfig:
    """실행 시점 설정 스냅샷."""

    kma_service_key: str | None = None
    kma_nx: int = 60          # 기상청 격자 X (안양시 기본값)
    kma_ny: int = 121         # 기상청 격자 Y (안양시 기본값)
    model_path: Path | None = None
    model_kind: str | None = None          # "onnx" | "pt" | None
    face_blur_enabled: bool = True
    store_original_image: bool = False     # 기본값: 비식별 사진만 저장
    detection_conf_threshold: float = 0.25
    routing_api_enabled: bool = False      # 외부 도로망 API 사용 가능 여부
    notes: list[str] = field(default_factory=list)

    @property
    def kma_enabled(self) -> bool:
        return bool(self.kma_service_key)

    @property
    def model_available(self) -> bool:
        return self.model_path is not None and self.model_path.exists()


def find_model_file() -> tuple[Path | None, str | None]:
    """models/ 에서 사용할 모델 파일을 찾는다. 없으면 (None, None).

    ONNX 를 우선한다(추가 의존성이 적음).
    """
    env_path = os.getenv("MODEL_PATH", "").strip()
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend([MODELS_DIR / "best.onnx", MODELS_DIR / "best.pt"])

    for path in candidates:
        if path.exists() and path.is_file():
            suffix = path.suffix.lower()
            if suffix == ".onnx":
                return path, "onnx"
            if suffix == ".pt":
                return path, "pt"
    return None, None


def ensure_directories() -> None:
    for directory in (
        DATA_RAW_DIR,
        DATA_PROCESSED_DIR,
        MODELS_DIR,
        OUTPUTS_DIR,
        ASSETS_DIR,
        UPLOAD_IMAGE_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def load_config(dotenv_path: Path | None = None) -> AppConfig:
    """.env 와 환경변수를 읽어 설정을 만든다. 값이 없으면 안전한 기본값을 쓴다."""
    load_dotenv(dotenv_path or (PROJECT_ROOT / ".env"), override=False)

    key = (os.getenv("KMA_SERVICE_KEY") or "").strip() or None
    model_path, model_kind = find_model_file()

    cfg = AppConfig(
        kma_service_key=key,
        kma_nx=int(_env_float("KMA_NX", 60)),
        kma_ny=int(_env_float("KMA_NY", 121)),
        model_path=model_path,
        model_kind=model_kind,
        face_blur_enabled=_env_bool("FACE_BLUR_ENABLED", True),
        store_original_image=_env_bool("STORE_ORIGINAL_IMAGE", False),
        detection_conf_threshold=_env_float("DETECTION_CONF_THRESHOLD", 0.25),
        routing_api_enabled=_env_bool("ROUTING_API_ENABLED", False),
    )

    if not cfg.kma_enabled:
        cfg.notes.append(
            "KMA_SERVICE_KEY 가 설정되지 않아 기상청 단기예보 조회를 사용할 수 없습니다. "
            "강수량은 직접 입력해야 합니다."
        )
    if not cfg.model_available:
        cfg.notes.append(
            "탐지 모델 파일이 없어 사진 분석 기능을 사용할 수 없습니다. "
            f"models/best.onnx 또는 models/best.pt 를 배치하세요. (기준 경로: {MODELS_DIR})"
        )
    return cfg
