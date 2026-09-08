"""점검 우선순위 계산.

점검 우선순위 =
  0.30 × 과거 파손 이력 밀도
+ 0.25 × 사진상 파손 정도
+ 0.20 × 어린이보호구역 인접도
+ 0.15 × 도로구간 중요도
+ 0.10 × 강수 영향

모든 입력은 0~100 으로 정규화한다. 값이 None(=데이터 없음)인 항목은 0점으로
처리하지 않고, 사용 가능한 항목들의 가중치 합이 1이 되도록 재정규화한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import RISK_COMPONENT_LABELS, RISK_WEIGHTS


@dataclass
class RiskResult:
    total_score: float
    components: dict[str, dict[str, Any]] = field(default_factory=dict)
    used_components: list[str] = field(default_factory=list)
    excluded_components: list[str] = field(default_factory=list)
    formula: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_score": self.total_score,
            "components": self.components,
            "used_components": self.used_components,
            "excluded_components": self.excluded_components,
            "formula": self.formula,
            "notes": self.notes,
        }


def clamp_score(value: float) -> float:
    """0~100 범위로 자른다."""
    if value != value:  # NaN
        return 0.0
    return float(min(100.0, max(0.0, float(value))))


def compute_risk_score(
    history_density: float | None = None,
    photo_severity: float | None = None,
    school_zone: float | None = None,
    road_importance: float | None = None,
    rainfall: float | None = None,
    weights: dict[str, float] | None = None,
) -> RiskResult:
    """항목별 점수(0~100, 없으면 None)로 총점을 계산한다.

    - 사용 가능한 항목이 하나도 없으면 총점 0, 모든 항목이 제외로 보고된다.
    - 총점은 항상 0~100 범위이다.
    """
    weights = weights or RISK_WEIGHTS
    raw_inputs: dict[str, float | None] = {
        "history_density": history_density,
        "photo_severity": photo_severity,
        "school_zone": school_zone,
        "road_importance": road_importance,
        "rainfall": rainfall,
    }

    available = {k: v for k, v in raw_inputs.items() if v is not None}
    missing = [k for k, v in raw_inputs.items() if v is None]

    result = RiskResult(total_score=0.0)
    result.excluded_components = missing

    weight_sum = sum(weights.get(k, 0.0) for k in available)
    if not available or weight_sum <= 0:
        result.formula = "사용 가능한 입력 항목이 없어 총점을 계산할 수 없습니다."
        result.notes.append(
            "모든 입력 데이터가 없어 총점 0점으로 표시합니다. "
            "이 값은 '위험도가 낮다'는 의미가 아니라 '판단할 근거가 없다'는 의미입니다."
        )
        for key in raw_inputs:
            result.components[key] = {
                "label": RISK_COMPONENT_LABELS.get(key, key),
                "score": None,
                "base_weight": weights.get(key, 0.0),
                "effective_weight": 0.0,
                "contribution": 0.0,
                "available": False,
            }
        return result

    total = 0.0
    for key in raw_inputs:
        base_weight = weights.get(key, 0.0)
        if key in available:
            score = clamp_score(available[key])
            effective = base_weight / weight_sum
            contribution = score * effective
            total += contribution
            result.components[key] = {
                "label": RISK_COMPONENT_LABELS.get(key, key),
                "score": round(score, 2),
                "base_weight": base_weight,
                "effective_weight": round(effective, 4),
                "contribution": round(contribution, 2),
                "available": True,
            }
        else:
            result.components[key] = {
                "label": RISK_COMPONENT_LABELS.get(key, key),
                "score": None,
                "base_weight": base_weight,
                "effective_weight": 0.0,
                "contribution": 0.0,
                "available": False,
            }

    result.total_score = round(clamp_score(total), 2)
    result.used_components = [k for k in raw_inputs if k in available]
    result.formula = build_formula_text(result)
    if missing:
        excluded_labels = ", ".join(RISK_COMPONENT_LABELS.get(k, k) for k in missing)
        result.notes.append(
            f"데이터가 없어 제외된 항목: {excluded_labels}. "
            f"남은 항목의 가중치 합({weight_sum:.2f})이 1이 되도록 재정규화했습니다."
        )
    return result


def build_formula_text(result: RiskResult) -> str:
    """실제로 사용된 항목만으로 계산식 문자열을 만든다."""
    parts = []
    for key, comp in result.components.items():
        if not comp["available"]:
            continue
        parts.append(
            f"{comp['effective_weight']:.3f} × {comp['label']}({comp['score']:.1f})"
        )
    if not parts:
        return "계산 불가"
    return " + ".join(parts) + f" = {result.total_score:.2f}"


def photo_severity_from_detections(
    detections: list[dict[str, Any]] | None,
    damage_area_ratio: float | None = None,
    class_weights: dict[str, float] | None = None,
) -> float | None:
    """사진 분석 결과를 0~100 의 '사진상 파손 정도' 점수로 변환한다.

    detections 가 None 이면(=분석하지 않음/모델 없음) None 을 반환하여
    우선순위 계산에서 해당 항목이 제외되도록 한다.
    빈 리스트([])는 '분석했으나 탐지 없음'이므로 0점이다.
    """
    if detections is None:
        return None

    class_weights = class_weights or {
        "D40": 1.0,   # 포트홀
        "D20": 0.8,   # 거북등 균열
        "D10": 0.6,   # 횡방향 균열
        "D00": 0.6,   # 종방향 균열
    }
    if not detections:
        return 0.0

    # 유형 가중치 × 신뢰도 의 최댓값(60점) + 탐지 개수(20점) + 면적비율(20점)
    severity = 0.0
    for det in detections:
        cls = str(det.get("class_name", det.get("class", "")))
        conf = float(det.get("confidence", 0.0))
        severity = max(severity, class_weights.get(cls, 0.5) * conf)
    type_part = severity * 60.0

    count_part = min(len(detections), 5) / 5.0 * 20.0

    ratio = damage_area_ratio if damage_area_ratio is not None else 0.0
    area_part = min(max(float(ratio), 0.0), 0.3) / 0.3 * 20.0

    return clamp_score(type_part + count_part + area_part)
