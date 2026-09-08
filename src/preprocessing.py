"""좌표 검증, 중복 제거, 날짜 파싱 등 전처리. 제외 사유를 모두 기록한다."""

from __future__ import annotations

import pandas as pd

from . import config
from .schemas import (
    COL_BASE_DATE,
    COL_CATEGORY_NO,
    COL_COLLECTED_AT,
    COL_HAZARD_TYPE,
    COL_LAT,
    COL_LON,
    COL_ROAD_NAME,
    CleaningReport,
    EXCL_LAT_MISSING,
    EXCL_LAT_NOT_NUMERIC,
    EXCL_LON_MISSING,
    EXCL_LON_NOT_NUMERIC,
    EXCL_OUT_OF_BBOX,
)

# 제외 사유를 담는 열 (원본 유지 버전에서 사용)
COL_EXCLUDE_REASON = "exclude_reason"


def _to_numeric_with_reason(
    series: pd.Series, missing_reason: str, not_numeric_reason: str
) -> tuple[pd.Series, pd.Series]:
    """숫자 변환 결과와 행별 제외 사유(없으면 빈 문자열)를 반환."""
    raw = series.astype("string").str.strip()
    # 천단위 구분자/따옴표 제거
    cleaned = raw.str.replace(",", "", regex=False).str.replace('"', "", regex=False)
    numeric = pd.to_numeric(cleaned, errors="coerce")

    is_missing = cleaned.isna() | (cleaned == "")
    reason = pd.Series("", index=series.index, dtype="object")
    reason[is_missing] = missing_reason
    reason[(~is_missing) & numeric.isna()] = not_numeric_reason
    return numeric, reason


def clean_damage_frame(
    frame: pd.DataFrame,
    bbox: dict[str, float] | None = None,
    dedup_subset: list[str] | None = None,
) -> tuple[pd.DataFrame, CleaningReport]:
    """파손 이력 데이터를 정제한다.

    - 위/경도를 숫자로 변환하고, 변환 실패 및 안양시 범위 이탈 행을 제외 사유와 함께 기록
    - 수집날짜/데이터기준일자를 datetime 으로 파싱
    - 중복 제거 전후 건수를 보고

    반환: (정제된 DataFrame, CleaningReport)
    """
    bbox = bbox or config.ANYANG_BBOX
    report = CleaningReport(rows_in=int(len(frame)))

    if frame.empty:
        report.rows_before_dedup = 0
        report.rows_after_dedup = 0
        report.messages.append("입력 데이터가 비어 있습니다.")
        return frame.copy(), report

    work = frame.copy()

    for col in (COL_LAT, COL_LON):
        if col not in work.columns:
            work[col] = pd.NA

    lat, lat_reason = _to_numeric_with_reason(
        work[COL_LAT], EXCL_LAT_MISSING, EXCL_LAT_NOT_NUMERIC
    )
    lon, lon_reason = _to_numeric_with_reason(
        work[COL_LON], EXCL_LON_MISSING, EXCL_LON_NOT_NUMERIC
    )
    work[COL_LAT] = lat
    work[COL_LON] = lon

    reason = lat_reason.where(lat_reason != "", lon_reason)

    in_bbox = (
        lat.between(bbox["lat_min"], bbox["lat_max"])
        & lon.between(bbox["lon_min"], bbox["lon_max"])
    )
    coords_ok = lat.notna() & lon.notna()
    out_of_bbox = coords_ok & ~in_bbox
    reason[out_of_bbox] = EXCL_OUT_OF_BBOX

    work[COL_EXCLUDE_REASON] = reason
    excluded = work.loc[reason != ""]
    report.excluded_counts = (
        excluded[COL_EXCLUDE_REASON].value_counts().to_dict() if len(excluded) else {}
    )
    report.excluded_counts = {str(k): int(v) for k, v in report.excluded_counts.items()}

    kept = work.loc[reason == ""].drop(columns=[COL_EXCLUDE_REASON]).copy()

    # 날짜 파싱
    for col in (COL_COLLECTED_AT, COL_BASE_DATE):
        if col in kept.columns:
            kept[col] = parse_dates(kept[col])

    # 문자열 항목 정리
    for col in (COL_HAZARD_TYPE, COL_ROAD_NAME, COL_CATEGORY_NO):
        if col in kept.columns:
            kept[col] = kept[col].astype("string").str.strip()

    # 중복 제거
    report.rows_before_dedup = int(len(kept))
    subset = dedup_subset or _default_dedup_subset(kept)
    if subset:
        kept = kept.drop_duplicates(subset=subset, keep="first")
    else:
        kept = kept.drop_duplicates()
    report.rows_after_dedup = int(len(kept))
    report.duplicates_removed = report.rows_before_dedup - report.rows_after_dedup
    report.messages.append(
        f"중복 제거 기준 열: {', '.join(subset) if subset else '전체 열'}"
    )
    report.messages.append(
        f"중복 제거 전 {report.rows_before_dedup:,}건 -> 후 {report.rows_after_dedup:,}건 "
        f"(제거 {report.duplicates_removed:,}건)"
    )

    kept = kept.reset_index(drop=True)
    report.rows_out = int(len(kept))
    return kept, report


def _default_dedup_subset(frame: pd.DataFrame) -> list[str]:
    """분류번호가 있으면 그것을 기준으로, 없으면 좌표+유형+수집날짜를 기준으로."""
    if COL_CATEGORY_NO in frame.columns and frame[COL_CATEGORY_NO].notna().any():
        return [COL_CATEGORY_NO]
    candidates = [COL_LAT, COL_LON, COL_HAZARD_TYPE, COL_COLLECTED_AT]
    return [c for c in candidates if c in frame.columns]


def parse_dates(series: pd.Series) -> pd.Series:
    """여러 날짜 표기(YYYY-MM-DD, YYYYMMDD, YYYY.MM.DD 등)를 datetime 으로 변환."""
    text = series.astype("string").str.strip()
    text = text.str.replace(r"[./]", "-", regex=True)
    parsed = pd.to_datetime(text, errors="coerce", format="mixed")
    # YYYYMMDD 형식 보정
    remaining = parsed.isna() & text.notna()
    if remaining.any():
        fallback = pd.to_datetime(
            text[remaining], errors="coerce", format="%Y%m%d"
        )
        parsed = parsed.copy()
        parsed[remaining] = fallback
    return parsed


def clean_school_zone_frame(
    frame: pd.DataFrame, bbox: dict[str, float] | None = None
) -> tuple[pd.DataFrame, CleaningReport]:
    """어린이보호구역 좌표를 숫자로 변환하고 범위를 벗어난 행을 제외한다."""
    bbox = bbox or config.ANYANG_BBOX
    report = CleaningReport(rows_in=int(len(frame)))
    if frame.empty:
        return frame.copy(), report

    work = frame.copy()
    for col in ("lat", "lon"):
        if col not in work.columns:
            work[col] = pd.NA

    lat, lat_reason = _to_numeric_with_reason(
        work["lat"], EXCL_LAT_MISSING, EXCL_LAT_NOT_NUMERIC
    )
    lon, lon_reason = _to_numeric_with_reason(
        work["lon"], EXCL_LON_MISSING, EXCL_LON_NOT_NUMERIC
    )
    work["lat"] = lat
    work["lon"] = lon

    reason = lat_reason.where(lat_reason != "", lon_reason)
    coords_ok = lat.notna() & lon.notna()
    in_bbox = (
        lat.between(bbox["lat_min"], bbox["lat_max"])
        & lon.between(bbox["lon_min"], bbox["lon_max"])
    )
    reason[coords_ok & ~in_bbox] = EXCL_OUT_OF_BBOX

    counts = reason[reason != ""].value_counts().to_dict()
    report.excluded_counts = {str(k): int(v) for k, v in counts.items()}

    kept = work.loc[reason == ""].copy()
    report.rows_before_dedup = int(len(kept))
    kept = kept.drop_duplicates(subset=["lat", "lon"], keep="first").reset_index(drop=True)
    report.rows_after_dedup = int(len(kept))
    report.duplicates_removed = report.rows_before_dedup - report.rows_after_dedup
    report.rows_out = int(len(kept))
    return kept, report


def summarize_base_date(frame: pd.DataFrame) -> str | None:
    """데이터기준일자 대표값(최댓값)을 문자열로 반환. 없으면 None."""
    if COL_BASE_DATE not in frame.columns:
        return None
    series = frame[COL_BASE_DATE].dropna()
    if series.empty:
        return None
    try:
        return pd.to_datetime(series).max().strftime("%Y-%m-%d")
    except Exception:
        return str(series.iloc[-1])
