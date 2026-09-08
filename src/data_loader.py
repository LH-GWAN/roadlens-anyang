"""CSV / 외부 API 로딩. 실패해도 예외를 밖으로 던지지 않고 사유를 보고한다."""

from __future__ import annotations

import io
import re
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from . import config
from .schemas import (
    DAMAGE_COLUMN_ALIASES,
    DAMAGE_STANDARD_COLUMNS,
    SCHOOL_ZONE_COLUMN_ALIASES,
    SCHOOL_ZONE_STANDARD_COLUMNS,
    SZ_ADDRESS,
    SZ_SIDO,
    SZ_SIGUNGU,
    LoadReport,
)

# 공공데이터포털 CSV 에서 자주 쓰이는 인코딩 순서
CANDIDATE_ENCODINGS = ("utf-8-sig", "cp949", "euc-kr", "utf-8")

_NON_WORD_RE = re.compile(r"[\s 　_\-()\[\]{}.,/\\'\"`~!@#$%^&*+=|:;?<>]")


def normalize_column_name(name: Any) -> str:
    """열 이름을 비교 가능한 형태로 정규화한다.

    - BOM(U+FEFF) 및 제로폭 문자 제거
    - 유니코드 NFKC 정규화 (전각/반각 통일)
    - 공백/괄호/기호 제거, 소문자화

    예) '\\ufeff 위 도 (WGS84)' -> 'wgs84' 가 되면 곤란하므로 괄호 안 내용은
        제거하지 않고 기호만 제거한다: '위도wgs84'
    """
    text = "" if name is None else str(name)
    text = text.replace("﻿", "").replace("​", "")
    text = unicodedata.normalize("NFKC", text)
    text = _NON_WORD_RE.sub("", text)
    return text.strip().lower()


def _match_alias(normalized: str, aliases: dict[str, str]) -> str | None:
    """정확히 일치 -> 접두 일치(단위/좌표계 표기 꼬리표 허용) 순으로 찾는다."""
    if normalized in aliases:
        return aliases[normalized]
    # '위도wgs84', '경도epsg4326' 처럼 뒤에 표기가 붙는 경우를 허용한다.
    best: tuple[int, str] | None = None
    for alias, standard in aliases.items():
        if normalized.startswith(alias) and len(alias) >= 2:
            if best is None or len(alias) > best[0]:
                best = (len(alias), standard)
    return best[1] if best else None


def build_column_mapping(
    columns: list[Any], aliases: dict[str, str]
) -> dict[str, str]:
    """원본 열 이름 -> 표준 열 이름 매핑. 중복 매핑은 먼저 나온 열을 우선한다."""
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for col in columns:
        standard = _match_alias(normalize_column_name(col), aliases)
        if standard and standard not in used:
            mapping[str(col)] = standard
            used.add(standard)
    return mapping


def read_csv_tolerant(path: Path) -> tuple[pd.DataFrame, str]:
    """인코딩을 순차 시도하며 CSV 를 읽는다. 성공한 인코딩 이름을 함께 반환."""
    raw = path.read_bytes()
    last_error: Exception | None = None
    for encoding in CANDIDATE_ENCODINGS:
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        try:
            frame = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=True)
        except Exception as exc:  # pandas 파싱 오류
            last_error = exc
            continue
        return frame, encoding
    raise ValueError(f"CSV 를 읽지 못했습니다: {path} ({last_error})")


def _apply_mapping(
    frame: pd.DataFrame,
    aliases: dict[str, str],
    standard_columns: list[str],
    report: LoadReport,
) -> pd.DataFrame:
    report.raw_columns = [str(c) for c in frame.columns]
    mapping = build_column_mapping(list(frame.columns), aliases)
    report.mapped_columns = mapping

    renamed = frame.rename(columns=mapping)
    # 이름이 겹치는 열이 생기면 먼저 나온 열만 남긴다(뒤 열은 버림).
    if renamed.columns.duplicated().any():
        renamed = renamed.loc[:, ~renamed.columns.duplicated()]
        report.messages.append("이름이 중복된 열이 있어 첫 번째 열만 사용했습니다.")

    missing = [c for c in standard_columns if c not in renamed.columns]
    report.missing_standard_columns = missing
    for col in missing:
        renamed[col] = pd.NA
    if missing:
        report.messages.append(
            "원본에서 찾지 못한 표준 항목(빈 값으로 채움): " + ", ".join(missing)
        )
    report.row_count = int(len(renamed))
    return renamed


def load_damage_csv(path: Path | None = None) -> tuple[pd.DataFrame, LoadReport]:
    """안양시 도로부속물 파손 현황 CSV 로딩.

    실제 데이터 파일이 없으면 샘플(테스트 전용)을 쓰고 is_sample=True 로 표시한다.
    """
    target = Path(path) if path else config.DAMAGE_CSV_PATH
    is_sample = False
    if not target.exists():
        if config.SAMPLE_DAMAGE_CSV_PATH.exists():
            target = config.SAMPLE_DAMAGE_CSV_PATH
            is_sample = True
        else:
            report = LoadReport(source_path=str(target))
            report.messages.append(
                f"파일이 없습니다: {target}. "
                "공공데이터포털에서 내려받아 data/raw/anyang_road_damage.csv 로 저장하세요."
            )
            empty = pd.DataFrame(columns=DAMAGE_STANDARD_COLUMNS)
            return empty, report

    report = LoadReport(source_path=str(target), is_sample=is_sample)
    if is_sample:
        report.messages.append(
            "실제 안양시 데이터가 없어 테스트 전용 샘플을 표시하고 있습니다. "
            "이 값은 실제 파손 이력이 아닙니다."
        )
    frame, encoding = read_csv_tolerant(target)
    report.encoding_used = encoding
    mapped = _apply_mapping(frame, DAMAGE_COLUMN_ALIASES, DAMAGE_STANDARD_COLUMNS, report)
    return mapped, report


def load_school_zone_csv(
    path: Path | None = None, sigungu_keyword: str = "안양"
) -> tuple[pd.DataFrame, LoadReport]:
    """전국어린이보호구역 표준데이터 CSV 로딩 후 안양시 자료만 필터링."""
    target = Path(path) if path else config.SCHOOL_ZONE_CSV_PATH
    is_sample = False
    if not target.exists():
        if config.SAMPLE_SCHOOL_ZONE_CSV_PATH.exists():
            target = config.SAMPLE_SCHOOL_ZONE_CSV_PATH
            is_sample = True
        else:
            report = LoadReport(source_path=str(target))
            report.messages.append(
                f"파일이 없습니다: {target}. "
                "공공데이터포털에서 내려받아 data/raw/school_zones.csv 로 저장하세요."
            )
            return pd.DataFrame(columns=SCHOOL_ZONE_STANDARD_COLUMNS), report

    report = LoadReport(source_path=str(target), is_sample=is_sample)
    if is_sample:
        report.messages.append(
            "실제 어린이보호구역 데이터가 없어 테스트 전용 샘플을 사용하고 있습니다."
        )
    frame, encoding = read_csv_tolerant(target)
    report.encoding_used = encoding
    mapped = _apply_mapping(
        frame, SCHOOL_ZONE_COLUMN_ALIASES, SCHOOL_ZONE_STANDARD_COLUMNS, report
    )

    before = len(mapped)
    filtered = filter_school_zones_by_region(mapped, sigungu_keyword)
    report.messages.append(
        f"'{sigungu_keyword}' 필터 적용: {before:,}건 -> {len(filtered):,}건"
    )
    report.row_count = int(len(filtered))
    return filtered, report


def filter_school_zones_by_region(
    frame: pd.DataFrame, keyword: str = "안양"
) -> pd.DataFrame:
    """시군구명 / 주소 문자열에 키워드가 포함된 행만 남긴다."""
    if frame.empty:
        return frame
    mask = pd.Series(False, index=frame.index)
    for col in (SZ_SIGUNGU, SZ_ADDRESS, SZ_SIDO):
        if col in frame.columns:
            mask |= frame[col].astype(str).str.contains(keyword, na=False)
    if not mask.any():
        # 키워드가 전혀 없으면 전국 데이터가 아닐 수 있으므로 원본을 유지한다.
        return frame
    return frame.loc[mask].copy()


# ---------------------------------------------------------------------------
# 기상청 단기예보 (KMA_SERVICE_KEY 가 있을 때만 호출)
# ---------------------------------------------------------------------------
KMA_VILAGE_FCST_URL = (
    "https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
)


def _latest_base_datetime(now: datetime | None = None) -> tuple[str, str]:
    """단기예보 발표시각(02,05,08,11,14,17,20,23시) 중 직전 시각을 고른다."""
    now = now or datetime.now()
    slots = [2, 5, 8, 11, 14, 17, 20, 23]
    candidate = now - timedelta(minutes=45)  # 발표 후 제공 지연 고려
    for hour in reversed(slots):
        if candidate.hour >= hour:
            return candidate.strftime("%Y%m%d"), f"{hour:02d}00"
    prev = candidate - timedelta(days=1)
    return prev.strftime("%Y%m%d"), "2300"


def fetch_precipitation(
    service_key: str | None,
    nx: int = 60,
    ny: int = 121,
    timeout: float = 8.0,
) -> dict[str, Any]:
    """기상청 단기예보에서 강수량(RN1/PCP)과 강수확률(POP)을 조회한다.

    반환값 예:
      {"available": False, "reason": "...", "source": "unavailable"}
      {"available": True, "rain_mm": 3.5, "pop": 60, "base": "...", "source": "kma_api"}

    키가 없거나 호출이 실패하면 available=False 와 사유를 반환한다(예외 없음).
    """
    if not service_key:
        return {
            "available": False,
            "source": "unavailable",
            "reason": "KMA_SERVICE_KEY 가 설정되지 않았습니다.",
        }

    try:
        import requests
    except ImportError:  # pragma: no cover
        return {
            "available": False,
            "source": "unavailable",
            "reason": "requests 패키지가 설치되지 않았습니다.",
        }

    base_date, base_time = _latest_base_datetime()
    params = {
        "serviceKey": service_key,
        "pageNo": 1,
        "numOfRows": 300,
        "dataType": "JSON",
        "base_date": base_date,
        "base_time": base_time,
        "nx": nx,
        "ny": ny,
    }
    try:
        response = requests.get(KMA_VILAGE_FCST_URL, params=params, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        return {
            "available": False,
            "source": "unavailable",
            "reason": f"기상청 API 호출 실패: {exc}",
        }

    try:
        items = payload["response"]["body"]["items"]["item"]
    except (KeyError, TypeError):
        header = (payload or {}).get("response", {}).get("header", {})
        return {
            "available": False,
            "source": "unavailable",
            "reason": (
                "기상청 응답 형식이 예상과 다릅니다: "
                f"{header.get('resultCode', '?')} {header.get('resultMsg', '')}"
            ),
        }

    return parse_kma_items(items, base_date, base_time)


def parse_kma_items(
    items: list[dict[str, Any]], base_date: str, base_time: str
) -> dict[str, Any]:
    """단기예보 item 목록에서 가장 이른 예보시각의 PCP/POP 를 추출한다."""
    if not items:
        return {
            "available": False,
            "source": "unavailable",
            "reason": "기상청 응답에 예보 항목이 없습니다.",
        }

    rows = sorted(items, key=lambda x: (str(x.get("fcstDate")), str(x.get("fcstTime"))))
    first_slot = (rows[0].get("fcstDate"), rows[0].get("fcstTime"))
    rain_mm: float | None = None
    pop: float | None = None
    for row in rows:
        if (row.get("fcstDate"), row.get("fcstTime")) != first_slot:
            continue
        category = str(row.get("category", ""))
        value = str(row.get("fcstValue", "")).strip()
        if category == "PCP":
            rain_mm = parse_pcp_value(value)
        elif category == "POP":
            try:
                pop = float(value)
            except ValueError:
                pop = None

    if rain_mm is None and pop is None:
        return {
            "available": False,
            "source": "unavailable",
            "reason": "강수 관련 항목(PCP/POP)을 찾지 못했습니다.",
        }

    return {
        "available": True,
        "source": "kma_api",
        "rain_mm": rain_mm if rain_mm is not None else 0.0,
        "pop": pop,
        "base": f"{base_date} {base_time}",
        "fcst": f"{first_slot[0]} {first_slot[1]}",
    }


def parse_pcp_value(value: str) -> float:
    """'강수없음', '1mm 미만', '30.0mm', '50.0mm 이상' 형식을 mm 숫자로 변환."""
    text = (value or "").strip()
    if text in {"", "강수없음", "-", "null", "None"}:
        return 0.0
    if "미만" in text:
        return 0.5
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if not match:
        return 0.0
    return float(match.group(1))
