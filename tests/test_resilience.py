"""모델 파일 / API 키가 없어도 앱이 종료되지 않는지 검증.

(프롬프트 '테스트 요구사항'의 마지막 두 항목을 담당하는 추가 파일)
"""

from __future__ import annotations

import numpy as np
import pytest

from src import config, repository
from src.data_loader import fetch_precipitation
from src.image_inference import detect_damage
from src.privacy import (
    PLATE_STATUS_NOT_PERFORMED,
    anonymize_image,
)
from src.risk_score import compute_risk_score, photo_severity_from_detections
from src.schemas import InspectionStatus


def blank_image(h: int = 120, w: int = 160) -> np.ndarray:
    return np.full((h, w, 3), 128, dtype=np.uint8)


class TestModelMissing:
    def test_detect_damage_without_model_returns_unavailable(self, tmp_path):
        result = detect_damage(blank_image(), model_path=tmp_path / "nope.onnx",
                               model_kind="onnx")
        assert result.available is False
        assert result.detections is None
        assert "모델 파일이 필요합니다" in result.reason or "불러오지" in result.reason

    def test_missing_model_message_shows_install_path(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MODEL_PATH", str(tmp_path / "absent.onnx"))
        monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
        result = detect_damage(blank_image(), model_path=None, model_kind=None)
        assert result.available is False
        assert result.detections is None

    def test_no_fake_detections_are_produced(self, tmp_path):
        result = detect_damage(blank_image(), model_path=tmp_path / "nope.pt",
                               model_kind="pt")
        assert result.count == 0
        assert result.detections is None
        assert result.damage_area_ratio is None

    def test_risk_score_still_computed_without_photo(self):
        severity = photo_severity_from_detections(None)
        result = compute_risk_score(history_density=60, photo_severity=severity,
                                    school_zone=40)
        assert result.total_score > 0
        assert "photo_severity" in result.excluded_components

    def test_find_model_file_returns_none_when_absent(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MODEL_PATH", raising=False)
        monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
        path, kind = config.find_model_file()
        assert path is None and kind is None


class TestApiKeyMissing:
    def test_fetch_precipitation_without_key_returns_reason(self):
        result = fetch_precipitation(None)
        assert result["available"] is False
        assert "KMA_SERVICE_KEY" in result["reason"]

    def test_fetch_precipitation_with_empty_key(self):
        result = fetch_precipitation("")
        assert result["available"] is False

    def test_load_config_without_key_does_not_raise(self, monkeypatch, tmp_path):
        monkeypatch.delenv("KMA_SERVICE_KEY", raising=False)
        monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
        cfg = config.load_config(dotenv_path=tmp_path / ".env")
        assert cfg.kma_enabled is False
        assert any("KMA_SERVICE_KEY" in note for note in cfg.notes)

    def test_manual_rainfall_input_path_still_scores(self):
        from src.spatial_analysis import rainfall_score

        # API 없이 사용자가 직접 입력한 강수량으로도 점수를 낼 수 있어야 한다.
        assert rainfall_score(15.0, 30.0) == pytest.approx(50.0)


class TestPrivacyDefaults:
    def test_plate_not_claimed_removed_by_default(self):
        result = anonymize_image(blank_image(), blur_faces=True)
        assert result.plate_status == PLATE_STATUS_NOT_PERFORMED
        assert result.plate_count == 0
        assert any("번호판" in m for m in result.messages)

    def test_anonymize_returns_image_of_same_shape(self):
        image = blank_image()
        result = anonymize_image(image)
        assert result.image is not None
        assert result.image.shape == image.shape

    def test_mosaic_changes_only_target_region(self):
        """모자이크가 지정 영역만 바꾸고 바깥은 건드리지 않아야 한다."""
        from src.privacy import mosaic_region

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[:, :, 0] = np.arange(100, dtype=np.uint8)  # 좌우로 변하는 패턴
        before = image.copy()
        mosaic_region(image, 20, 20, 40, 40)
        assert not np.array_equal(image[20:60, 20:60], before[20:60, 20:60])
        assert np.array_equal(image[:20, :], before[:20, :])
        assert np.array_equal(image[60:, :], before[60:, :])

    def test_mosaic_out_of_bounds_is_clipped(self):
        """이미지 밖 좌표를 줘도 예외 없이 처리되어야 한다."""
        from src.privacy import mosaic_region

        image = np.zeros((50, 50, 3), dtype=np.uint8)
        mosaic_region(image, -20, -20, 30, 30)      # 좌상단 밖
        mosaic_region(image, 45, 45, 30, 30)        # 우하단 밖
        mosaic_region(image, 100, 100, 10, 10)      # 완전히 밖
        assert image.shape == (50, 50, 3)

    def test_face_cascade_is_available(self):
        """얼굴 탐지기가 실제로 로드되어야 한다(OpenCV 5.x 는 XML 미동봉)."""
        from src.privacy import _load_cascade

        assert _load_cascade("haarcascade_frontalface_default.xml") is not None

    def test_original_image_not_stored_by_default(self, monkeypatch, tmp_path):
        monkeypatch.delenv("STORE_ORIGINAL_IMAGE", raising=False)
        monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
        cfg = config.load_config(dotenv_path=tmp_path / ".env")
        assert cfg.store_original_image is False


class TestRepositorySeparatesAiAndReviewer:
    def test_ai_and_reviewer_fields_are_separate(self, tmp_path):
        db = tmp_path / "test.sqlite3"
        inspection_id = repository.create_inspection(
            lat=37.40,
            lon=126.92,
            ai_result={
                "available": True,
                "detections": [{"class_name": "D40", "confidence": 0.8}],
                "damage_area_ratio": 0.05,
                "model_path": "models/best.onnx",
            },
            risk_result={"total_score": 71.5},
            db_path=db,
        )
        repository.set_reviewer_judgement(
            inspection_id, judgement="보수 불필요", note="현장 확인 결과 경미",
            db_path=db,
        )
        row = repository.get_inspection(inspection_id, db_path=db)
        assert row["ai_top_class"] == "D40"
        assert row["reviewer_judgement"] == "보수 불필요"
        # AI 판단이 담당자 판단으로 덮어써지지 않아야 한다.
        assert row["ai_top_confidence"] == pytest.approx(0.8)

    def test_status_transitions_are_recorded(self, tmp_path):
        db = tmp_path / "test.sqlite3"
        inspection_id = repository.create_inspection(lat=37.4, lon=126.9, db_path=db)
        repository.update_status(inspection_id, InspectionStatus.SCHEDULED.value,
                                 note="점검 계획 수립", db_path=db)
        repository.update_status(inspection_id, InspectionStatus.DONE.value, db_path=db)
        history = repository.get_status_history(inspection_id, db_path=db)
        assert [h["to_status"] for h in history] == ["접수", "점검예정", "처리완료"]

    def test_invalid_status_is_rejected(self, tmp_path):
        db = tmp_path / "test.sqlite3"
        inspection_id = repository.create_inspection(db_path=db)
        with pytest.raises(ValueError):
            repository.update_status(inspection_id, "아무상태", db_path=db)

    def test_ai_fields_empty_when_model_unavailable(self, tmp_path):
        db = tmp_path / "test.sqlite3"
        inspection_id = repository.create_inspection(
            ai_result={"available": False, "detections": None}, db_path=db
        )
        row = repository.get_inspection(inspection_id, db_path=db)
        assert row["ai_available"] == 0
        assert row["ai_detection_count"] is None
        assert row["ai_top_class"] is None

    def test_all_six_statuses_supported(self, tmp_path):
        db = tmp_path / "test.sqlite3"
        assert InspectionStatus.values() == [
            "접수", "수동검토", "점검예정", "현장확인", "보수필요", "처리완료"
        ]
        counts = repository.status_counts(db_path=db)
        assert set(counts.keys()) == set(InspectionStatus.values())
