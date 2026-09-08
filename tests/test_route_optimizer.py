"""순찰 경로 최적화 테스트 (0개/1개 선택 포함)."""

from __future__ import annotations

import numpy as np
import pytest

from src.route_optimizer import (
    DISTANCE_MODE_HAVERSINE,
    DISTANCE_MODE_ROAD_API,
    optimize_route,
    format_distance,
)
from src.spatial_analysis import haversine_m, haversine_matrix_m

ANYANG_POINTS = [
    (37.3943, 126.9568),
    (37.4009, 126.9220),
    (37.3810, 126.9330),
    (37.4100, 126.9500),
    (37.3900, 126.9700),
]


class TestEdgeCases:
    def test_zero_points_returns_empty_route(self):
        result = optimize_route([])
        assert result.order == []
        assert result.total_distance_m == 0.0
        assert any("없습니다" in m for m in result.messages)

    def test_one_point_returns_single_stop(self):
        result = optimize_route([ANYANG_POINTS[0]])
        assert result.order == [0]
        assert result.total_distance_m == 0.0
        assert result.leg_distances_m == []

    def test_two_points_distance_matches_haversine(self):
        result = optimize_route(ANYANG_POINTS[:2])
        expected = haversine_m(*ANYANG_POINTS[0], *ANYANG_POINTS[1])
        assert result.total_distance_m == pytest.approx(expected, rel=1e-6)
        assert sorted(result.order) == [0, 1]


class TestOptimization:
    def test_all_points_visited_exactly_once(self):
        result = optimize_route(ANYANG_POINTS, time_limit_s=1)
        assert sorted(result.order) == list(range(len(ANYANG_POINTS)))

    def test_round_trip_returns_to_start(self):
        result = optimize_route(ANYANG_POINTS, return_to_start=True, time_limit_s=1)
        assert len(result.leg_distances_m) == len(ANYANG_POINTS)
        assert result.total_distance_m > 0

    def test_open_route_has_n_minus_1_legs(self):
        result = optimize_route(ANYANG_POINTS, return_to_start=False, time_limit_s=1)
        assert len(result.leg_distances_m) == len(ANYANG_POINTS) - 1

    def test_optimized_route_not_worse_than_input_order(self):
        matrix = haversine_matrix_m(ANYANG_POINTS)
        naive = sum(
            matrix[i][i + 1] for i in range(len(ANYANG_POINTS) - 1)
        )
        result = optimize_route(ANYANG_POINTS, time_limit_s=2)
        assert result.total_distance_m <= naive + 1e-6

    def test_custom_distance_matrix_is_used(self):
        matrix = haversine_matrix_m(ANYANG_POINTS[:3]) * 2
        result = optimize_route(
            ANYANG_POINTS[:3], distance_matrix_m=matrix, time_limit_s=1
        )
        plain = optimize_route(ANYANG_POINTS[:3], time_limit_s=1)
        assert result.total_distance_m == pytest.approx(
            plain.total_distance_m * 2, rel=1e-6
        )

    def test_mismatched_matrix_raises(self):
        with pytest.raises(ValueError):
            optimize_route(ANYANG_POINTS, distance_matrix_m=np.zeros((2, 2)))


class TestApproximationNotice:
    def test_haversine_mode_is_flagged_approximate(self):
        result = optimize_route(ANYANG_POINTS[:3], time_limit_s=1)
        assert result.distance_mode == DISTANCE_MODE_HAVERSINE
        assert result.is_approximate is True
        assert any("근사 경로" in m for m in result.messages)

    def test_road_api_mode_is_not_flagged_approximate(self):
        result = optimize_route(
            ANYANG_POINTS[:3],
            distance_matrix_m=haversine_matrix_m(ANYANG_POINTS[:3]),
            distance_mode=DISTANCE_MODE_ROAD_API,
            time_limit_s=1,
        )
        assert result.is_approximate is False
        assert not any("근사 경로" in m for m in result.messages)


class TestDistanceHelpers:
    def test_matrix_shape_for_small_inputs(self):
        assert haversine_matrix_m([]).shape == (0, 0)
        assert haversine_matrix_m([ANYANG_POINTS[0]]).shape == (1, 1)

    def test_matrix_is_symmetric_with_zero_diagonal(self):
        matrix = haversine_matrix_m(ANYANG_POINTS)
        assert np.allclose(matrix, matrix.T)
        assert np.allclose(np.diag(matrix), 0.0)

    def test_haversine_known_distance(self):
        # 위도 1도 ≈ 111km
        distance = haversine_m(37.0, 127.0, 38.0, 127.0)
        assert 110_000 < distance < 112_000

    def test_format_distance(self):
        assert format_distance(500) == "500 m"
        assert "km" in format_distance(2500)
