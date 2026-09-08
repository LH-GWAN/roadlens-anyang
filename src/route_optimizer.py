"""순찰 경로 최적화.

도로망 데이터가 없으므로 기본은 Haversine 직선거리 기반 근사 경로이다.
외부 도로망 API 를 쓸 수 있을 때만 실제 도로거리 계산을 활성화한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .spatial_analysis import haversine_matrix_m

DISTANCE_MODE_HAVERSINE = "haversine_approx"
DISTANCE_MODE_ROAD_API = "road_network_api"

HAVERSINE_NOTICE = (
    "도로망 데이터가 없어 직선(Haversine) 거리 기반의 근사 경로입니다. "
    "실제 도로 주행거리와 다를 수 있습니다."
)


@dataclass
class RouteResult:
    order: list[int] = field(default_factory=list)          # 입력 인덱스 방문 순서
    total_distance_m: float = 0.0
    leg_distances_m: list[float] = field(default_factory=list)
    distance_mode: str = DISTANCE_MODE_HAVERSINE
    is_approximate: bool = True
    solver: str = ""
    messages: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "order": self.order,
            "total_distance_m": self.total_distance_m,
            "leg_distances_m": self.leg_distances_m,
            "distance_mode": self.distance_mode,
            "is_approximate": self.is_approximate,
            "solver": self.solver,
            "messages": self.messages,
        }


def optimize_route(
    points: list[tuple[float, float]],
    distance_matrix_m: np.ndarray | None = None,
    return_to_start: bool = False,
    time_limit_s: int = 5,
    distance_mode: str = DISTANCE_MODE_HAVERSINE,
) -> RouteResult:
    """방문 순서를 계산한다.

    - 점이 0개면 빈 경로, 1개면 그 지점 하나만 담은 경로를 오류 없이 반환한다.
    - OR-Tools 를 쓸 수 없으면 최근접 이웃 + 2-opt 로 대체한다.
    """
    n = len(points)
    result = RouteResult(distance_mode=distance_mode)
    result.is_approximate = distance_mode != DISTANCE_MODE_ROAD_API
    if result.is_approximate:
        result.messages.append(HAVERSINE_NOTICE)

    if n == 0:
        result.solver = "none"
        result.messages.append("선택된 점검 대상이 없습니다.")
        return result
    if n == 1:
        result.order = [0]
        result.solver = "none"
        result.messages.append("점검 대상이 1개여서 경로 계산이 필요하지 않습니다.")
        return result

    matrix = (
        np.asarray(distance_matrix_m, dtype=float)
        if distance_matrix_m is not None
        else haversine_matrix_m(points)
    )
    if matrix.shape != (n, n):
        raise ValueError(f"거리행렬 크기가 맞지 않습니다: {matrix.shape} != ({n}, {n})")

    if n == 2:
        result.order = [0, 1]
        result.solver = "trivial"
    else:
        order, solver = _solve_tsp(matrix, return_to_start, time_limit_s)
        result.order = order
        result.solver = solver

    legs = []
    seq = result.order + ([result.order[0]] if return_to_start and n > 1 else [])
    for a, b in zip(seq[:-1], seq[1:]):
        legs.append(float(matrix[a][b]))
    result.leg_distances_m = legs
    result.total_distance_m = float(sum(legs))
    return result


def _solve_tsp(
    matrix: np.ndarray, return_to_start: bool, time_limit_s: int
) -> tuple[list[int], str]:
    """OR-Tools 우선, 실패 시 휴리스틱."""
    try:
        return _solve_with_ortools(matrix, return_to_start, time_limit_s), "OR-Tools"
    except Exception:
        return _solve_nearest_neighbor_2opt(matrix, return_to_start), "nearest-neighbor+2opt"


def _solve_with_ortools(
    matrix: np.ndarray, return_to_start: bool, time_limit_s: int
) -> list[int]:
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    n = matrix.shape[0]
    scaled = np.rint(matrix).astype(int)

    if return_to_start:
        manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    else:
        # 종료 지점을 자유롭게 두기 위해 가상 종점(모든 거리 0)을 추가한다.
        n_ext = n + 1
        extended = np.zeros((n_ext, n_ext), dtype=int)
        extended[:n, :n] = scaled
        scaled = extended
        manager = pywrapcp.RoutingIndexManager(n_ext, 1, [0], [n])

    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index: int, to_index: int) -> int:
        return int(scaled[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)])

    transit_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_index)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    params.time_limit.FromSeconds(max(1, int(time_limit_s)))

    solution = routing.SolveWithParameters(params)
    if solution is None:
        raise RuntimeError("OR-Tools 가 해를 찾지 못했습니다.")

    order: list[int] = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        node = manager.IndexToNode(index)
        if node < n:
            order.append(node)
        index = solution.Value(routing.NextVar(index))
    last = manager.IndexToNode(index)
    if last < n and last not in order:
        order.append(last)
    return order


def _solve_nearest_neighbor_2opt(
    matrix: np.ndarray, return_to_start: bool
) -> list[int]:
    n = matrix.shape[0]
    unvisited = set(range(1, n))
    order = [0]
    while unvisited:
        current = order[-1]
        nxt = min(unvisited, key=lambda j: matrix[current][j])
        order.append(nxt)
        unvisited.remove(nxt)

    def path_length(seq: list[int]) -> float:
        full = seq + ([seq[0]] if return_to_start else [])
        return float(sum(matrix[a][b] for a, b in zip(full[:-1], full[1:])))

    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for k in range(i + 1, n):
                candidate = order[:i] + order[i : k + 1][::-1] + order[k + 1 :]
                if path_length(candidate) + 1e-9 < path_length(order):
                    order = candidate
                    improved = True
    return order


def format_distance(meters: float) -> str:
    if meters < 1000:
        return f"{meters:,.0f} m"
    return f"{meters / 1000:,.2f} km"
