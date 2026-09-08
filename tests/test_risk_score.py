"""점검 우선순위 점수 범위 / 재정규화 테스트."""

from __future__ import annotations

import itertools

import pytest

from src.config import RISK_WEIGHTS
from src.risk_score import compute_risk_score, photo_severity_from_detections
from src.spatial_analysis import (
    history_density_score,
    rainfall_score,
    school_zone_score,
)


class TestScoreRange:
    def test_all_max_gives_100(self):
        result = compute_risk_score(100, 100, 100, 100, 100)
        assert result.total_score == 100.0

    def test_all_zero_gives_0(self):
        result = compute_risk_score(0, 0, 0, 0, 0)
        assert result.total_score == 0.0

    def test_out_of_range_inputs_are_clamped(self):
        result = compute_risk_score(1000, -50, 200, 100, 100)
        assert 0.0 <= result.total_score <= 100.0
        assert result.components["photo_severity"]["score"] == 0.0
        assert result.components["history_density"]["score"] == 100.0

    @pytest.mark.parametrize("values", [
        (0, 0, 0, 0, 0),
        (100, 0, 50, 25, 75),
        (33.3, 66.6, 99.9, 0.1, 50),
        (None, 80, None, 40, None),
        (None, None, None, None, None),
    ])
    def test_score_always_within_0_100(self, values):
        result = compute_risk_score(*values)
        assert 0.0 <= result.total_score <= 100.0

    def test_every_missing_combination_stays_in_range(self):
        keys = list(RISK_WEIGHTS.keys())
        for r in range(len(keys) + 1):
            for missing in itertools.combinations(keys, r):
                kwargs = {k: (None if k in missing else 100.0) for k in keys}
                result = compute_risk_score(**kwargs)
                assert 0.0 <= result.total_score <= 100.0


class TestRenormalization:
    def test_missing_component_is_excluded_not_zeroed(self):
        result = compute_risk_score(
            history_density=100, photo_severity=None, school_zone=100,
            road_importance=100, rainfall=100,
        )
        # 사진 점수만 없음 -> 나머지 항목이 100점이므로 총점도 100점이어야 한다.
        assert result.total_score == 100.0
        assert "photo_severity" in result.excluded_components
        assert result.components["photo_severity"]["available"] is False

    def test_effective_weights_sum_to_one(self):
        result = compute_risk_score(history_density=50, school_zone=50)
        total_weight = sum(
            c["effective_weight"] for c in result.components.values() if c["available"]
        )
        assert total_weight == pytest.approx(1.0, abs=1e-3)

    def test_single_available_component_uses_its_own_value(self):
        result = compute_risk_score(history_density=42.0)
        assert result.total_score == pytest.approx(42.0, abs=0.01)
        assert result.used_components == ["history_density"]
        assert len(result.excluded_components) == 4

    def test_no_data_reports_zero_with_explanation(self):
        result = compute_risk_score()
        assert result.total_score == 0.0
        assert len(result.excluded_components) == 5
        assert any("근거가 없다" in note for note in result.notes)

    def test_formula_lists_only_used_components(self):
        result = compute_risk_score(history_density=50, rainfall=10)
        assert "과거 파손 이력 밀도" in result.formula
        assert "사진상 파손 정도" not in result.formula

    def test_weights_match_specification(self):
        assert RISK_WEIGHTS["history_density"] == 0.30
        assert RISK_WEIGHTS["photo_severity"] == 0.25
        assert RISK_WEIGHTS["school_zone"] == 0.20
        assert RISK_WEIGHTS["road_importance"] == 0.15
        assert RISK_WEIGHTS["rainfall"] == 0.10
        assert sum(RISK_WEIGHTS.values()) == pytest.approx(1.0)

    def test_full_weights_are_applied_when_all_present(self):
        result = compute_risk_score(100, 0, 0, 0, 0)
        assert result.total_score == pytest.approx(30.0, abs=0.01)


class TestComponentScores:
    def test_history_density_score_is_capped(self):
        assert history_density_score(1000, reference_count=20) == 100.0
        assert history_density_score(0) == 0.0
        assert 0 <= history_density_score(10, 20) <= 100

    def test_school_zone_score_none_when_no_data(self):
        assert school_zone_score(None) is None

    def test_school_zone_score_decreases_with_distance(self):
        near = school_zone_score(50.0, 500.0)
        far = school_zone_score(400.0, 500.0)
        assert near > far
        assert school_zone_score(600.0, 500.0) == 0.0

    def test_rainfall_score_none_when_no_data(self):
        assert rainfall_score(None) is None
        assert rainfall_score(0.0) == 0.0
        assert rainfall_score(100.0, 30.0) == 100.0


class TestPhotoSeverity:
    def test_none_when_not_analyzed(self):
        # 모델이 없어 분석하지 않은 경우 -> 항목 제외(None)
        assert photo_severity_from_detections(None) is None

    def test_zero_when_analyzed_but_nothing_found(self):
        assert photo_severity_from_detections([]) == 0.0

    def test_pothole_scores_higher_than_crack(self):
        pothole = photo_severity_from_detections(
            [{"class_name": "D40", "confidence": 0.9}], damage_area_ratio=0.1
        )
        crack = photo_severity_from_detections(
            [{"class_name": "D00", "confidence": 0.9}], damage_area_ratio=0.1
        )
        assert pothole > crack

    def test_severity_within_range(self):
        score = photo_severity_from_detections(
            [{"class_name": "D40", "confidence": 1.0}] * 20, damage_area_ratio=1.0
        )
        assert 0.0 <= score <= 100.0
