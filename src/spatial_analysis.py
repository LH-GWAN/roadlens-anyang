"""공간 분석: 밀집구간 군집화, 이력 밀도, 어린이보호구역 인접도, 도로구간 중요도."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from . import config
from .schemas import COL_LAT, COL_LON, COL_NODE_LINK, COL_ROAD_NAME

EARTH_RADIUS_M = 6_371_000.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """두 좌표 사이의 대권거리(m)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


def haversine_matrix_m(points: list[tuple[float, float]]) -> np.ndarray:
    """(lat, lon) 목록에 대한 거리행렬(m). 점이 0~1개여도 올바른 형태를 반환."""
    n = len(points)
    matrix = np.zeros((n, n), dtype=float)
    if n < 2:
        return matrix
    coords = np.radians(np.asarray(points, dtype=float))
    lat = coords[:, 0][:, None]
    lon = coords[:, 1][:, None]
    dlat = lat - lat.T
    dlon = lon - lon.T
    a = np.sin(dlat / 2) ** 2 + np.cos(lat) * np.cos(lat.T) * np.sin(dlon / 2) ** 2
    matrix = 2 * EARTH_RADIUS_M * np.arcsin(np.clip(np.sqrt(a), 0, 1))
    np.fill_diagonal(matrix, 0.0)
    return matrix


# ---------------------------------------------------------------------------
# 밀집구간 군집화
# ---------------------------------------------------------------------------
def cluster_damage_points(
    frame: pd.DataFrame,
    eps_m: float = 150.0,
    min_samples: int = 5,
    algorithm: str = "auto",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """파손 지점을 군집화해 밀집구간을 찾는다.

    algorithm: "auto" | "hdbscan" | "dbscan"
    - "auto" 는 scikit-learn 1.3+ 의 HDBSCAN 을 우선 사용하고, 없으면 DBSCAN 으로 내려간다.
    - 반환 DataFrame 에 'cluster' 열(-1 은 잡음)을 추가한다.
    """
    info: dict[str, Any] = {"algorithm": None, "n_clusters": 0, "n_noise": 0}
    if frame.empty:
        result = frame.copy()
        result["cluster"] = pd.Series(dtype="int64")
        info["reason"] = "데이터가 없습니다."
        return result, info

    coords = frame[[COL_LAT, COL_LON]].astype(float).to_numpy()
    if len(coords) < max(2, min_samples):
        result = frame.copy()
        result["cluster"] = -1
        info["reason"] = (
            f"군집화에 필요한 최소 건수({max(2, min_samples)}건) 미만이라 밀집구간을 계산하지 않았습니다."
        )
        return result, info

    radians = np.radians(coords)
    labels = None
    used = None

    if algorithm in {"auto", "hdbscan"}:
        try:
            import inspect

            from sklearn.cluster import HDBSCAN  # scikit-learn >= 1.3

            kwargs: dict[str, Any] = {
                "min_cluster_size": max(2, int(min_samples)),
                "metric": "haversine",
                "cluster_selection_epsilon": eps_m / EARTH_RADIUS_M,
            }
            # sklearn 1.9+ 의 copy 기본값 변경 경고를 피한다(입력은 이미 임시 배열).
            if "copy" in inspect.signature(HDBSCAN).parameters:
                kwargs["copy"] = True
            model = HDBSCAN(**kwargs)
            labels = model.fit_predict(radians)
            used = "HDBSCAN (scikit-learn)"
        except Exception as exc:  # 미지원 버전 등
            info["hdbscan_error"] = str(exc)
            labels = None

    if labels is None:
        from sklearn.cluster import DBSCAN

        model = DBSCAN(
            eps=eps_m / EARTH_RADIUS_M,
            min_samples=max(2, int(min_samples)),
            metric="haversine",
            algorithm="ball_tree",
        )
        labels = model.fit_predict(radians)
        used = "DBSCAN"

    result = frame.copy()
    result["cluster"] = labels.astype(int)
    info["algorithm"] = used
    info["n_clusters"] = int(len(set(int(x) for x in labels if x != -1)))
    info["n_noise"] = int(sum(1 for x in labels if x == -1))
    info["eps_m"] = eps_m
    info["min_samples"] = min_samples
    return result, info


def cluster_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """군집별 중심좌표/건수/대표 도로명 요약."""
    if frame.empty or "cluster" not in frame.columns:
        return pd.DataFrame(
            columns=["cluster", "count", "lat", "lon", "road_address", "hazard_types"]
        )
    valid = frame.loc[frame["cluster"] >= 0]
    if valid.empty:
        return pd.DataFrame(
            columns=["cluster", "count", "lat", "lon", "road_address", "hazard_types"]
        )

    rows = []
    for cluster_id, group in valid.groupby("cluster"):
        road = ""
        if COL_ROAD_NAME in group.columns:
            names = group[COL_ROAD_NAME].dropna().astype(str)
            road = names.mode().iloc[0] if not names.empty else ""
        hazard_types = ""
        if "hazard_type" in group.columns:
            uniq = group["hazard_type"].dropna().astype(str).unique().tolist()
            hazard_types = ", ".join(uniq[:5])
        rows.append(
            {
                "cluster": int(cluster_id),
                "count": int(len(group)),
                "lat": float(group[COL_LAT].mean()),
                "lon": float(group[COL_LON].mean()),
                "road_address": road,
                "hazard_types": hazard_types,
            }
        )
    summary = pd.DataFrame(rows).sort_values("count", ascending=False)
    return summary.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 과거 파손 이력 밀도
# ---------------------------------------------------------------------------
def distances_from_point_m(
    frame: pd.DataFrame, lat: float, lon: float,
    lat_col: str = COL_LAT, lon_col: str = COL_LON,
) -> np.ndarray:
    """한 지점에서 DataFrame 의 모든 좌표까지의 거리(m) 벡터.

    N×N 거리행렬을 만들지 않으므로 수만 건에서도 메모리를 크게 쓰지 않는다.
    """
    if frame.empty:
        return np.empty(0, dtype=float)
    coords = np.radians(frame[[lat_col, lon_col]].astype(float).to_numpy())
    lat_r, lon_r = math.radians(lat), math.radians(lon)
    dlat = coords[:, 0] - lat_r
    dlon = coords[:, 1] - lon_r
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat_r) * np.cos(coords[:, 0]) * np.sin(dlon / 2) ** 2
    )
    return 2 * EARTH_RADIUS_M * np.arcsin(np.clip(np.sqrt(a), 0, 1))


def history_density_count(
    frame: pd.DataFrame, lat: float, lon: float, radius_m: float = 300.0
) -> int:
    """지정 좌표 반경 내 과거 파손 이력 건수."""
    if frame.empty:
        return 0
    return int((distances_from_point_m(frame, lat, lon) <= radius_m).sum())


def history_density_score(
    count: int, reference_count: int = 20
) -> float:
    """반경 내 건수를 0~100 점수로 변환.

    reference_count 건 이상이면 100점. 상한을 넘어도 100 을 유지한다.
    (reference_count 는 데이터 분포를 보고 화면에서 조정할 수 있다.)
    """
    if reference_count <= 0:
        return 0.0
    return float(min(100.0, max(0.0, count / reference_count * 100.0)))


def nearest_school_zone_distance_m(
    school_zones: pd.DataFrame, lat: float, lon: float
) -> float | None:
    """가장 가까운 어린이보호구역까지의 거리(m). 데이터가 없으면 None."""
    if school_zones is None or school_zones.empty:
        return None
    if "lat" not in school_zones.columns or "lon" not in school_zones.columns:
        return None
    coords = school_zones[["lat", "lon"]].astype(float).dropna()
    if coords.empty:
        return None
    dist = distances_from_point_m(coords, lat, lon, lat_col="lat", lon_col="lon")
    return float(dist.min())


def school_zone_score(
    distance_m: float | None, max_distance_m: float | None = None
) -> float | None:
    """어린이보호구역 인접도 점수(0~100). 거리 정보가 없으면 None."""
    if distance_m is None:
        return None
    limit = max_distance_m or config.SCHOOL_ZONE_MAX_DISTANCE_M
    if limit <= 0:
        return 0.0
    if distance_m >= limit:
        return 0.0
    return float(max(0.0, min(100.0, (1.0 - distance_m / limit) * 100.0)))


# ---------------------------------------------------------------------------
# 도로구간 중요도
# ---------------------------------------------------------------------------
def road_importance_table(frame: pd.DataFrame) -> pd.DataFrame:
    """노드링크(없으면 도로명주소)별 파손 이력 건수를 집계한다.

    도로망 등급/교통량 데이터가 없으므로, 이 시제품에서는 '해당 도로구간에
    누적된 과거 파손 이력 건수'를 도로구간 중요도의 대리지표로 사용한다.
    (README 의 '데이터 및 모델 한계' 참조)
    """
    key = None
    if COL_NODE_LINK in frame.columns and frame[COL_NODE_LINK].notna().any():
        key = COL_NODE_LINK
    elif COL_ROAD_NAME in frame.columns and frame[COL_ROAD_NAME].notna().any():
        key = COL_ROAD_NAME
    if key is None or frame.empty:
        return pd.DataFrame(columns=["key", "count"])

    counts = (
        frame[key].dropna().astype(str).str.strip().replace("", np.nan).dropna()
        .value_counts().rename_axis("key").reset_index(name="count")
    )
    return counts


def road_importance_score(
    table: pd.DataFrame, key_value: str | None
) -> float | None:
    """도로구간 중요도 점수(0~100). 집계표나 키가 없으면 None(=데이터 없음)."""
    if table is None or table.empty or not key_value:
        return None
    key_value = str(key_value).strip()
    row = table.loc[table["key"] == key_value]
    if row.empty:
        return 0.0
    max_count = float(table["count"].max())
    if max_count <= 0:
        return 0.0
    return float(min(100.0, float(row["count"].iloc[0]) / max_count * 100.0))


def rainfall_score(rain_mm: float | None, saturation_mm: float = 30.0) -> float | None:
    """강수 영향 점수(0~100). 강수 정보가 없으면 None.

    saturation_mm 이상이면 100점으로 본다.
    """
    if rain_mm is None:
        return None
    if saturation_mm <= 0:
        return 0.0
    return float(min(100.0, max(0.0, rain_mm / saturation_mm * 100.0)))
