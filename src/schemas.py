"""데이터 표준 스키마, 열 이름 별칭, 상태 정의.

공공데이터포털 CSV는 배포 시점에 따라 열 이름 표기(공백/괄호/영문 병기)가
달라질 수 있으므로, 원본 열 이름을 내부 표준 이름으로 매핑해 사용한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# 내부 표준 열 이름 (도로부속물 파손 현황)
# ---------------------------------------------------------------------------
COL_CATEGORY_NO = "category_no"      # 분류번호
COL_HAZARD_TYPE = "hazard_type"      # 위험물유형
COL_COLLECTED_AT = "collected_at"    # 수집날짜
COL_LAT = "lat"                      # 위도
COL_LON = "lon"                      # 경도
COL_NODE_LINK = "node_link"          # 노드링크
COL_START_NODE = "start_node"        # 시작노드
COL_END_NODE = "end_node"            # 끝노드
COL_ROAD_NAME = "road_address"       # 도로명주소
COL_BASE_DATE = "base_date"          # 데이터기준일자
COL_SOURCE = "source"                # 출처 데이터셋 이름 (여러 CSV 를 합칠 때)

DAMAGE_STANDARD_COLUMNS = [
    COL_CATEGORY_NO,
    COL_HAZARD_TYPE,
    COL_COLLECTED_AT,
    COL_LAT,
    COL_LON,
    COL_NODE_LINK,
    COL_START_NODE,
    COL_END_NODE,
    COL_ROAD_NAME,
    COL_BASE_DATE,
]

# 원본 열 이름(정규화 후) -> 내부 표준 이름
# 정규화 규칙은 normalize_column_name() 참고: BOM/공백/특수문자 제거 + 소문자화
DAMAGE_COLUMN_ALIASES: dict[str, str] = {
    # 분류번호
    "분류번호": COL_CATEGORY_NO,
    "분류no": COL_CATEGORY_NO,
    "관리번호": COL_CATEGORY_NO,
    "일련번호": COL_CATEGORY_NO,
    "id": COL_CATEGORY_NO,
    # 위험물유형
    "위험물유형": COL_HAZARD_TYPE,
    "위험물종류": COL_HAZARD_TYPE,
    "파손유형": COL_HAZARD_TYPE,
    "유형": COL_HAZARD_TYPE,
    "위험물유형명": COL_HAZARD_TYPE,
    "hazardtype": COL_HAZARD_TYPE,
    # 수집날짜
    "수집날짜": COL_COLLECTED_AT,
    "수집일자": COL_COLLECTED_AT,
    "수집일": COL_COLLECTED_AT,
    "발생일자": COL_COLLECTED_AT,
    "collecteddate": COL_COLLECTED_AT,
    # 위도 / 경도
    "위도": COL_LAT,
    "위도y": COL_LAT,
    "lat": COL_LAT,
    "latitude": COL_LAT,
    "y좌표": COL_LAT,
    "경도": COL_LON,
    "경도x": COL_LON,
    "lon": COL_LON,
    "lng": COL_LON,
    "longitude": COL_LON,
    "x좌표": COL_LON,
    # 노드링크 계열
    "노드링크": COL_NODE_LINK,
    "노드링크id": COL_NODE_LINK,
    "링크id": COL_NODE_LINK,
    "nodelink": COL_NODE_LINK,
    "시작노드": COL_START_NODE,
    "시작노드id": COL_START_NODE,
    "startnode": COL_START_NODE,
    "끝노드": COL_END_NODE,
    "종료노드": COL_END_NODE,
    "끝노드id": COL_END_NODE,
    "endnode": COL_END_NODE,
    # 도로명주소
    "도로명주소": COL_ROAD_NAME,
    "도로명": COL_ROAD_NAME,
    "소재지도로명주소": COL_ROAD_NAME,
    "주소": COL_ROAD_NAME,
    "roadaddress": COL_ROAD_NAME,
    # 데이터기준일자
    "데이터기준일자": COL_BASE_DATE,
    "데이터기준일": COL_BASE_DATE,
    "기준일자": COL_BASE_DATE,
    "basedate": COL_BASE_DATE,
}


# ---------------------------------------------------------------------------
# 내부 표준 열 이름 (전국어린이보호구역 표준데이터)
# ---------------------------------------------------------------------------
SZ_NAME = "zone_name"            # 대상시설명
SZ_LAT = "lat"
SZ_LON = "lon"
SZ_ADDRESS = "address"           # 소재지도로명주소 / 지번주소
SZ_SIDO = "sido"                 # 시도명
SZ_SIGUNGU = "sigungu"           # 시군구명
SZ_FACILITY_KIND = "facility_kind"  # 시설종류

SCHOOL_ZONE_STANDARD_COLUMNS = [
    SZ_NAME,
    SZ_LAT,
    SZ_LON,
    SZ_ADDRESS,
    SZ_SIDO,
    SZ_SIGUNGU,
    SZ_FACILITY_KIND,
]

SCHOOL_ZONE_COLUMN_ALIASES: dict[str, str] = {
    "대상시설명": SZ_NAME,
    "시설명": SZ_NAME,
    "학교명": SZ_NAME,
    "명칭": SZ_NAME,
    "위도": SZ_LAT,
    "lat": SZ_LAT,
    "latitude": SZ_LAT,
    "경도": SZ_LON,
    "lon": SZ_LON,
    "lng": SZ_LON,
    "longitude": SZ_LON,
    "소재지도로명주소": SZ_ADDRESS,
    "소재지지번주소": SZ_ADDRESS,
    "도로명주소": SZ_ADDRESS,
    "지번주소": SZ_ADDRESS,
    "주소": SZ_ADDRESS,
    "시도명": SZ_SIDO,
    "시도": SZ_SIDO,
    "시군구명": SZ_SIGUNGU,
    "시군구": SZ_SIGUNGU,
    "대상시설종류": SZ_FACILITY_KIND,
    "시설종류": SZ_FACILITY_KIND,
}


# ---------------------------------------------------------------------------
# 처리상태
# ---------------------------------------------------------------------------
class InspectionStatus(str, Enum):
    """점검 처리상태. 화면/DB 공통으로 이 값만 사용한다."""

    RECEIVED = "접수"
    MANUAL_REVIEW = "수동검토"
    SCHEDULED = "점검예정"
    FIELD_CHECKED = "현장확인"
    REPAIR_NEEDED = "보수필요"
    DONE = "처리완료"

    @classmethod
    def values(cls) -> list[str]:
        return [s.value for s in cls]


# ---------------------------------------------------------------------------
# 결과 자료구조
# ---------------------------------------------------------------------------
@dataclass
class LoadReport:
    """CSV 로딩 결과 요약. 화면에 그대로 표시할 수 있도록 문자열 위주로 구성."""

    source_path: str
    is_sample: bool = False
    encoding_used: str | None = None
    raw_columns: list[str] = field(default_factory=list)
    mapped_columns: dict[str, str] = field(default_factory=dict)
    missing_standard_columns: list[str] = field(default_factory=list)
    row_count: int = 0
    messages: list[str] = field(default_factory=list)


@dataclass
class CleaningReport:
    """전처리 결과 요약(제외 사유/중복 제거 건수 포함)."""

    rows_in: int = 0
    rows_out: int = 0
    excluded_counts: dict[str, int] = field(default_factory=dict)
    rows_before_dedup: int = 0
    rows_after_dedup: int = 0
    duplicates_removed: int = 0
    messages: list[str] = field(default_factory=list)

    @property
    def excluded_total(self) -> int:
        return int(sum(self.excluded_counts.values()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "excluded_total": self.excluded_total,
            "excluded_counts": dict(self.excluded_counts),
            "rows_before_dedup": self.rows_before_dedup,
            "rows_after_dedup": self.rows_after_dedup,
            "duplicates_removed": self.duplicates_removed,
            "messages": list(self.messages),
        }


# 제외 사유 코드 (전처리에서 사용)
EXCL_LAT_NOT_NUMERIC = "위도 값이 숫자가 아님"
EXCL_LON_NOT_NUMERIC = "경도 값이 숫자가 아님"
EXCL_LAT_MISSING = "위도 값 없음"
EXCL_LON_MISSING = "경도 값 없음"
EXCL_OUT_OF_BBOX = "안양시 좌표 범위를 벗어남"
