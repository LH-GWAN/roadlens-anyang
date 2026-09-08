"""CSV 열 이름 매핑 / 좌표 검증 / 중복 제거 테스트."""

from __future__ import annotations

import pandas as pd
import pytest

from src import config
from src.data_loader import (
    build_column_mapping,
    load_damage_csv,
    load_school_zone_csv,
    normalize_column_name,
    parse_pcp_value,
    read_csv_tolerant,
)
from src.preprocessing import clean_damage_frame, parse_dates
from src.schemas import (
    COL_CATEGORY_NO,
    COL_HAZARD_TYPE,
    COL_LAT,
    COL_LON,
    DAMAGE_COLUMN_ALIASES,
    EXCL_LAT_NOT_NUMERIC,
    EXCL_OUT_OF_BBOX,
)


class TestColumnNameMapping:
    def test_bom_and_spaces_are_ignored(self):
        assert normalize_column_name("﻿ 위 도 ") == "위도"
        assert normalize_column_name("경도(WGS84)") == "경도wgs84"
        assert normalize_column_name("  분류번호\t") == "분류번호"

    def test_mapping_with_bom_whitespace_and_korean_variants(self):
        columns = [
            "﻿분류번호",
            " 위험물 유형 ",
            "수집일자",
            "위도(WGS84)",
            "경도(WGS84)",
            "노드링크ID",
            "시작노드",
            "끝노드",
            "소재지도로명주소",
            "데이터기준일자",
        ]
        mapping = build_column_mapping(columns, DAMAGE_COLUMN_ALIASES)
        standards = set(mapping.values())
        assert {"category_no", "hazard_type", "collected_at", "lat", "lon"} <= standards
        assert mapping["위도(WGS84)"] == "lat"
        assert mapping["경도(WGS84)"] == "lon"

    def test_english_headers_are_mapped(self):
        mapping = build_column_mapping(
            ["id", "hazardType", "latitude", "longitude"], DAMAGE_COLUMN_ALIASES
        )
        assert mapping["latitude"] == "lat"
        assert mapping["longitude"] == "lon"

    def test_cp949_csv_is_read(self, tmp_path):
        path = tmp_path / "cp949.csv"
        path.write_bytes("분류번호,위도,경도\n1,37.40,126.92\n".encode("cp949"))
        frame, encoding = read_csv_tolerant(path)
        assert encoding in {"cp949", "euc-kr"}
        assert list(frame.columns)[0] == "분류번호"

    def test_utf8_bom_csv_is_read(self, tmp_path):
        path = tmp_path / "bom.csv"
        path.write_bytes("﻿분류번호,위도,경도\n1,37.40,126.92\n".encode("utf-8"))
        frame, encoding = read_csv_tolerant(path)
        assert encoding == "utf-8-sig"
        assert "분류번호" in frame.columns


class TestCoordinateValidation:
    def _frame(self):
        return pd.DataFrame(
            {
                COL_CATEGORY_NO: ["1", "2", "3", "4", "5"],
                COL_HAZARD_TYPE: ["포트홀", "균열", "포트홀", "균열", "포트홀"],
                COL_LAT: ["37.4000", "37.4100", "없음", "35.1000", "37.4200"],
                COL_LON: ["126.9200", "126.9300", "126.9400", "129.0000", "126.9500"],
            }
        )

    def test_non_numeric_coordinate_is_excluded_with_reason(self):
        clean, report = clean_damage_frame(self._frame())
        assert EXCL_LAT_NOT_NUMERIC in report.excluded_counts
        assert report.excluded_counts[EXCL_LAT_NOT_NUMERIC] == 1

    def test_out_of_anyang_bbox_is_excluded_with_reason(self):
        clean, report = clean_damage_frame(self._frame())
        assert report.excluded_counts.get(EXCL_OUT_OF_BBOX) == 1

    def test_valid_rows_survive(self):
        clean, report = clean_damage_frame(self._frame())
        assert len(clean) == 3
        assert report.rows_in == 5
        assert report.excluded_total == 2

    def test_coordinates_are_numeric_after_cleaning(self):
        clean, _ = clean_damage_frame(self._frame())
        assert clean[COL_LAT].dtype.kind == "f"
        assert clean[COL_LON].dtype.kind == "f"

    def test_thousands_separator_is_tolerated(self):
        frame = pd.DataFrame({COL_LAT: ["37.40"], COL_LON: ['"126.92"']})
        clean, report = clean_damage_frame(frame)
        assert len(clean) == 1

    def test_empty_frame_does_not_raise(self):
        clean, report = clean_damage_frame(pd.DataFrame())
        assert clean.empty
        assert report.rows_in == 0


class TestDeduplication:
    def test_duplicate_counts_are_reported(self):
        frame = pd.DataFrame(
            {
                COL_CATEGORY_NO: ["1", "1", "2"],
                COL_LAT: ["37.40", "37.40", "37.41"],
                COL_LON: ["126.92", "126.92", "126.93"],
            }
        )
        clean, report = clean_damage_frame(frame)
        assert report.rows_before_dedup == 3
        assert report.rows_after_dedup == 2
        assert report.duplicates_removed == 1
        assert len(clean) == 2

    def test_dedup_falls_back_to_coordinates_when_no_id(self):
        frame = pd.DataFrame(
            {
                COL_LAT: ["37.40", "37.40"],
                COL_LON: ["126.92", "126.92"],
                COL_HAZARD_TYPE: ["포트홀", "포트홀"],
            }
        )
        clean, report = clean_damage_frame(frame)
        assert report.duplicates_removed == 1


class TestDateParsing:
    @pytest.mark.parametrize(
        "raw", ["2024-05-01", "2024.05.01", "2024/05/01", "20240501"]
    )
    def test_multiple_date_formats(self, raw):
        parsed = parse_dates(pd.Series([raw]))
        assert parsed.iloc[0].year == 2024
        assert parsed.iloc[0].month == 5

    def test_invalid_date_becomes_nat(self):
        parsed = parse_dates(pd.Series(["없음"]))
        assert pd.isna(parsed.iloc[0])


class TestMissingFilesDoNotCrash:
    def test_missing_damage_csv_without_sample_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "SAMPLE_DAMAGE_CSV_PATH", tmp_path / "no_sample.csv")
        frame, report = load_damage_csv(tmp_path / "no_such_file.csv")
        assert frame.empty
        assert report.is_sample is False
        assert any("파일이 없습니다" in m for m in report.messages)

    def test_missing_school_zone_csv_without_sample_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            config, "SAMPLE_SCHOOL_ZONE_CSV_PATH", tmp_path / "no_sample.csv"
        )
        frame, report = load_school_zone_csv(tmp_path / "no_such_file.csv")
        assert frame.empty
        assert report.is_sample is False
        assert any("파일이 없습니다" in m for m in report.messages)

    def test_missing_real_csv_falls_back_to_sample_and_flags_it(self, tmp_path):
        """실제 파일이 없으면 샘플로 대체하되 is_sample 로 반드시 구분되어야 한다."""
        if not config.SAMPLE_DAMAGE_CSV_PATH.exists():
            pytest.skip("샘플 데이터가 없는 환경")
        frame, report = load_damage_csv(tmp_path / "no_such_file.csv")
        assert report.is_sample is True
        assert any("실제" in m and "샘플" in m for m in report.messages)
        assert not frame.empty


class TestPcpParsing:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("강수없음", 0.0),
            ("", 0.0),
            ("1.0mm 미만", 0.5),
            ("3.5mm", 3.5),
            ("50.0mm 이상", 50.0),
        ],
    )
    def test_pcp_values(self, raw, expected):
        assert parse_pcp_value(raw) == expected
