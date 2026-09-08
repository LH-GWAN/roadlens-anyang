"""SQLite 저장소.

AI 분석 결과와 담당자 확인 결과를 서로 다른 필드에 저장한다.
계정정보/주민등록번호 등 개인 식별 정보는 저장하지 않는다.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from . import config
from .schemas import InspectionStatus

SCHEMA = """
CREATE TABLE IF NOT EXISTS inspections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,               -- 접수시각 (ISO8601)
    updated_at TEXT NOT NULL,
    lat REAL,
    lon REAL,
    road_address TEXT,

    anonymized_image_path TEXT,             -- 비식별 처리된 사진 경로
    original_image_path TEXT,               -- 설정이 허용될 때만 채워짐

    -- AI 분석 결과 (담당자 판단과 분리)
    ai_model_path TEXT,
    ai_available INTEGER NOT NULL DEFAULT 0,
    ai_detection_count INTEGER,
    ai_top_class TEXT,
    ai_top_confidence REAL,
    ai_damage_area_ratio REAL,
    ai_detections_json TEXT,
    ai_risk_score REAL,
    ai_risk_breakdown_json TEXT,

    -- 담당자 확인 결과
    reviewer_judgement TEXT,                -- 담당자가 입력한 판단
    reviewer_note TEXT,
    reviewer_name TEXT,                     -- 부서/직함 등 식별 최소화 문자열
    reviewed_at TEXT,

    status TEXT NOT NULL,
    privacy_json TEXT,                      -- 비식별 처리 상태 기록
    is_sample_data INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS status_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    inspection_id INTEGER NOT NULL,
    changed_at TEXT NOT NULL,
    from_status TEXT,
    to_status TEXT NOT NULL,
    note TEXT,
    FOREIGN KEY (inspection_id) REFERENCES inspections(id)
);
"""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


@contextmanager
def connect(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    path = Path(db_path) if db_path else config.DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: Path | None = None) -> None:
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)


def create_inspection(
    lat: float | None = None,
    lon: float | None = None,
    road_address: str | None = None,
    anonymized_image_path: str | None = None,
    original_image_path: str | None = None,
    ai_result: dict[str, Any] | None = None,
    risk_result: dict[str, Any] | None = None,
    privacy_result: dict[str, Any] | None = None,
    status: str = InspectionStatus.RECEIVED.value,
    is_sample_data: bool = False,
    db_path: Path | None = None,
) -> int:
    """점검 건을 등록한다. AI 결과가 없으면 관련 필드는 비워 둔다."""
    init_db(db_path)
    ai_result = ai_result or {}
    detections = ai_result.get("detections") or []
    top = max(detections, key=lambda d: d.get("confidence", 0.0)) if detections else None

    now = _now()
    with connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO inspections (
                created_at, updated_at, lat, lon, road_address,
                anonymized_image_path, original_image_path,
                ai_model_path, ai_available, ai_detection_count,
                ai_top_class, ai_top_confidence, ai_damage_area_ratio,
                ai_detections_json, ai_risk_score, ai_risk_breakdown_json,
                status, privacy_json, is_sample_data
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                now,
                now,
                lat,
                lon,
                road_address,
                anonymized_image_path,
                original_image_path,
                ai_result.get("model_path"),
                1 if ai_result.get("available") else 0,
                len(detections) if ai_result.get("available") else None,
                top.get("class_name") if top else None,
                float(top.get("confidence")) if top else None,
                ai_result.get("damage_area_ratio"),
                json.dumps(detections, ensure_ascii=False) if detections else None,
                (risk_result or {}).get("total_score"),
                json.dumps(risk_result, ensure_ascii=False) if risk_result else None,
                status,
                json.dumps(privacy_result, ensure_ascii=False) if privacy_result else None,
                1 if is_sample_data else 0,
            ),
        )
        inspection_id = int(cursor.lastrowid)
        conn.execute(
            """
            INSERT INTO status_history (inspection_id, changed_at, from_status, to_status, note)
            VALUES (?,?,?,?,?)
            """,
            (inspection_id, now, None, status, "신규 접수"),
        )
    return inspection_id


def update_status(
    inspection_id: int,
    new_status: str,
    note: str | None = None,
    db_path: Path | None = None,
) -> bool:
    """처리상태를 변경하고 이력을 남긴다."""
    if new_status not in InspectionStatus.values():
        raise ValueError(f"허용되지 않은 상태입니다: {new_status}")
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT status FROM inspections WHERE id = ?", (inspection_id,)
        ).fetchone()
        if row is None:
            return False
        now = _now()
        conn.execute(
            "UPDATE inspections SET status = ?, updated_at = ? WHERE id = ?",
            (new_status, now, inspection_id),
        )
        conn.execute(
            """
            INSERT INTO status_history (inspection_id, changed_at, from_status, to_status, note)
            VALUES (?,?,?,?,?)
            """,
            (inspection_id, now, row["status"], new_status, note),
        )
    return True


def set_reviewer_judgement(
    inspection_id: int,
    judgement: str,
    note: str | None = None,
    reviewer_name: str | None = None,
    db_path: Path | None = None,
) -> bool:
    """담당자 확인 결과를 AI 필드와 분리해 저장한다."""
    with connect(db_path) as conn:
        cursor = conn.execute(
            """
            UPDATE inspections
               SET reviewer_judgement = ?, reviewer_note = ?, reviewer_name = ?,
                   reviewed_at = ?, updated_at = ?
             WHERE id = ?
            """,
            (judgement, note, reviewer_name, _now(), _now(), inspection_id),
        )
        return cursor.rowcount > 0


def list_inspections(
    status: str | None = None, limit: int = 500, db_path: Path | None = None
) -> list[dict[str, Any]]:
    init_db(db_path)
    query = "SELECT * FROM inspections"
    params: list[Any] = []
    if status:
        query += " WHERE status = ?"
        params.append(status)
    query += " ORDER BY id DESC LIMIT ?"
    params.append(int(limit))
    with connect(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def get_inspection(inspection_id: int, db_path: Path | None = None) -> dict[str, Any] | None:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM inspections WHERE id = ?", (inspection_id,)
        ).fetchone()
    return dict(row) if row else None


def get_status_history(
    inspection_id: int, db_path: Path | None = None
) -> list[dict[str, Any]]:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM status_history WHERE inspection_id = ? ORDER BY id",
            (inspection_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def delete_inspection(inspection_id: int, db_path: Path | None = None) -> bool:
    """보유기간 경과분 삭제 등에 사용."""
    with connect(db_path) as conn:
        conn.execute(
            "DELETE FROM status_history WHERE inspection_id = ?", (inspection_id,)
        )
        cursor = conn.execute("DELETE FROM inspections WHERE id = ?", (inspection_id,))
        return cursor.rowcount > 0


def status_counts(db_path: Path | None = None) -> dict[str, int]:
    init_db(db_path)
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT status, COUNT(*) AS n FROM inspections GROUP BY status"
        ).fetchall()
    counts = {status: 0 for status in InspectionStatus.values()}
    for row in rows:
        counts[row["status"]] = int(row["n"])
    return counts
