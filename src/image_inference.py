"""도로 파손 탐지 추론.

모델 파일(models/best.onnx 또는 models/best.pt)이 없으면 예측값을 만들지 않고
available=False 와 안내 메시지를 반환한다. 가짜 탐지 결과를 생성하지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import config

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]


@dataclass
class DetectionResult:
    """탐지 결과.

    available=False 이면 detections 는 None 이며(=분석하지 않음),
    우선순위 계산에서 '사진상 파손 정도' 항목이 제외된다.
    """

    available: bool = False
    reason: str = ""
    detections: list[dict[str, Any]] | None = None
    damage_area_ratio: float | None = None
    annotated_image: np.ndarray | None = None
    model_path: str | None = None
    model_kind: str | None = None
    class_names: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.detections) if self.detections else 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "reason": self.reason,
            "count": self.count,
            "detections": self.detections,
            "damage_area_ratio": self.damage_area_ratio,
            "model_path": self.model_path,
            "model_kind": self.model_kind,
        }


MODEL_MISSING_MESSAGE = (
    "모델 파일이 필요합니다. Google Colab 학습 노트북으로 학습한 뒤 "
    "best.onnx 를 아래 경로에 저장하세요.\n"
    f"  - {config.MODELS_DIR / 'best.onnx'}\n"
    f"  - (또는) {config.MODELS_DIR / 'best.pt'}  ※ .pt 사용 시 ultralytics 설치 필요"
)


# ---------------------------------------------------------------------------
# 전처리 / 후처리
# ---------------------------------------------------------------------------
def letterbox(
    image_bgr: np.ndarray, size: int = 640, color: tuple[int, int, int] = (114, 114, 114)
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """가로세로 비율을 유지한 채 size×size 로 패딩. (이미지, 배율, (패딩x, 패딩y))"""
    h, w = image_bgr.shape[:2]
    scale = min(size / h, size / w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), color, dtype=np.uint8)
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return canvas, scale, (pad_x, pad_y)


def nms(
    boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45
) -> list[int]:
    """단순 NMS. boxes 는 (N,4) xyxy."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[order[1:]] - inter
        # union 이 0 인 항목에서 0/0 경고가 나지 않도록 나눗셈 자체를 건너뛴다.
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        order = order[1:][iou <= iou_threshold]
    return keep


def decode_yolo_output(
    output: np.ndarray,
    class_names: list[str],
    scale: float,
    pad: tuple[int, int],
    original_shape: tuple[int, int],
    conf_threshold: float,
    iou_threshold: float = 0.45,
) -> list[dict[str, Any]]:
    """Ultralytics YOLOv8 ONNX 출력 (1, 4+nc, N) 을 탐지 목록으로 변환."""
    array = np.squeeze(output)
    if array.ndim != 2:
        return []
    # (4+nc, N) 형태로 맞춘다.
    if array.shape[0] < array.shape[1]:
        preds = array
    else:
        preds = array.T

    num_classes = preds.shape[0] - 4
    if num_classes <= 0:
        return []

    boxes_cxcywh = preds[:4, :].T
    class_scores = preds[4:, :].T
    class_ids = class_scores.argmax(axis=1)
    confidences = class_scores.max(axis=1)

    mask = confidences >= conf_threshold
    if not mask.any():
        return []
    boxes_cxcywh = boxes_cxcywh[mask]
    class_ids = class_ids[mask]
    confidences = confidences[mask]

    cx, cy, w, h = (
        boxes_cxcywh[:, 0],
        boxes_cxcywh[:, 1],
        boxes_cxcywh[:, 2],
        boxes_cxcywh[:, 3],
    )
    pad_x, pad_y = pad
    x1 = (cx - w / 2 - pad_x) / scale
    y1 = (cy - h / 2 - pad_y) / scale
    x2 = (cx + w / 2 - pad_x) / scale
    y2 = (cy + h / 2 - pad_y) / scale

    orig_h, orig_w = original_shape
    x1 = np.clip(x1, 0, orig_w - 1)
    y1 = np.clip(y1, 0, orig_h - 1)
    x2 = np.clip(x2, 0, orig_w - 1)
    y2 = np.clip(y2, 0, orig_h - 1)
    xyxy = np.stack([x1, y1, x2, y2], axis=1)

    keep = nms(xyxy, confidences, iou_threshold)
    detections = []
    for idx in keep:
        cls_idx = int(class_ids[idx])
        cls_name = (
            class_names[cls_idx] if 0 <= cls_idx < len(class_names) else str(cls_idx)
        )
        detections.append(
            {
                "class_id": cls_idx,
                "class_name": cls_name,
                "label_ko": config.RDD2022_CLASS_LABELS_KO.get(cls_name, cls_name),
                "confidence": float(confidences[idx]),
                "bbox": [float(v) for v in xyxy[idx]],
            }
        )
    return detections


def damage_area_ratio(
    detections: list[dict[str, Any]], image_shape: tuple[int, int]
) -> float:
    """탐지 박스들이 덮는 면적 비율(중복 제외)을 0~1 로 계산."""
    h, w = image_shape
    if h <= 0 or w <= 0 or not detections:
        return 0.0
    mask = np.zeros((h, w), dtype=bool)
    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det["bbox"]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = True
    return float(mask.sum() / (h * w))


def draw_detections(
    image_bgr: np.ndarray, detections: list[dict[str, Any]]
) -> np.ndarray:
    """탐지 결과를 이미지에 그린다(라벨은 클래스 코드 + 신뢰도)."""
    if cv2 is None:
        return image_bgr
    output = image_bgr.copy()
    palette = [(0, 165, 255), (0, 255, 255), (255, 0, 255), (0, 0, 255)]
    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det["bbox"]]
        color = palette[det.get("class_id", 0) % len(palette)]
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        label = f"{det['class_name']} {det['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(output, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            output,
            label,
            (x1 + 2, max(10, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return output


# ---------------------------------------------------------------------------
# 모델 로딩 및 추론
# ---------------------------------------------------------------------------
_ONNX_SESSION_CACHE: dict[str, Any] = {}


def _get_onnx_session(model_path: Path):
    key = str(model_path.resolve())
    if key in _ONNX_SESSION_CACHE:
        return _ONNX_SESSION_CACHE[key]
    import onnxruntime as ort

    session = ort.InferenceSession(key, providers=["CPUExecutionProvider"])
    _ONNX_SESSION_CACHE[key] = session
    return session


def _class_names_from_session(session, fallback: list[str]) -> list[str]:
    """ONNX 메타데이터의 'names' 를 읽는다(Ultralytics export 시 기록됨)."""
    try:
        meta = session.get_modelmeta().custom_metadata_map or {}
        raw = meta.get("names")
        if not raw:
            return fallback
        import ast

        parsed = ast.literal_eval(raw)
        if isinstance(parsed, dict):
            return [str(parsed[k]) for k in sorted(parsed, key=lambda x: int(x))]
        if isinstance(parsed, list):
            return [str(v) for v in parsed]
    except Exception:
        pass
    return fallback


def detect_damage(
    image_bgr: np.ndarray,
    model_path: Path | None = None,
    model_kind: str | None = None,
    conf_threshold: float = 0.25,
    imgsz: int = 640,
) -> DetectionResult:
    """도로 파손을 탐지한다. 모델이 없으면 available=False 로 사유를 반환."""
    if model_path is None or model_kind is None:
        model_path, model_kind = config.find_model_file()

    if model_path is None or not Path(model_path).exists():
        return DetectionResult(
            available=False,
            reason=MODEL_MISSING_MESSAGE,
            detections=None,
        )

    if cv2 is None:
        return DetectionResult(
            available=False,
            reason="OpenCV 가 설치되지 않아 이미지 추론을 수행할 수 없습니다.",
            detections=None,
        )

    model_path = Path(model_path)
    original_shape = image_bgr.shape[:2]

    if model_kind == "onnx":
        try:
            session = _get_onnx_session(model_path)
        except Exception as exc:
            return DetectionResult(
                available=False,
                reason=f"ONNX 모델을 불러오지 못했습니다: {exc}",
                model_path=str(model_path),
                model_kind=model_kind,
                detections=None,
            )
        class_names = _class_names_from_session(session, config.RDD2022_CLASSES)

        # 입력 크기를 모델 정의에서 읽어 고정 크기 모델에 맞춘다.
        input_meta = session.get_inputs()[0]
        shape = input_meta.shape
        if isinstance(shape, (list, tuple)) and len(shape) == 4:
            if isinstance(shape[2], int) and shape[2] > 0:
                imgsz = int(shape[2])

        padded, scale, pad = letterbox(image_bgr, imgsz)
        blob = padded[:, :, ::-1].astype(np.float32) / 255.0  # BGR->RGB, 0~1
        blob = np.transpose(blob, (2, 0, 1))[None, ...]
        try:
            outputs = session.run(None, {input_meta.name: blob})
        except Exception as exc:
            return DetectionResult(
                available=False,
                reason=f"ONNX 추론에 실패했습니다: {exc}",
                model_path=str(model_path),
                model_kind=model_kind,
                detections=None,
            )
        detections = decode_yolo_output(
            outputs[0], class_names, scale, pad, original_shape, conf_threshold
        )

    elif model_kind == "pt":
        try:
            from ultralytics import YOLO
        except ImportError:
            return DetectionResult(
                available=False,
                reason=(
                    ".pt 모델을 사용하려면 ultralytics 패키지가 필요합니다. "
                    "`pip install ultralytics` 를 실행하거나 best.onnx 를 사용하세요."
                ),
                model_path=str(model_path),
                model_kind=model_kind,
                detections=None,
            )
        try:
            model = YOLO(str(model_path))
            results = model.predict(
                image_bgr, conf=conf_threshold, imgsz=imgsz, verbose=False
            )
        except Exception as exc:
            return DetectionResult(
                available=False,
                reason=f"YOLO 추론에 실패했습니다: {exc}",
                model_path=str(model_path),
                model_kind=model_kind,
                detections=None,
            )
        class_names = [str(v) for v in getattr(model, "names", {}).values()] or (
            config.RDD2022_CLASSES
        )
        detections = []
        for res in results:
            for box in res.boxes:
                cls_idx = int(box.cls.item())
                cls_name = (
                    class_names[cls_idx] if 0 <= cls_idx < len(class_names) else str(cls_idx)
                )
                detections.append(
                    {
                        "class_id": cls_idx,
                        "class_name": cls_name,
                        "label_ko": config.RDD2022_CLASS_LABELS_KO.get(cls_name, cls_name),
                        "confidence": float(box.conf.item()),
                        "bbox": [float(v) for v in box.xyxy[0].tolist()],
                    }
                )
    else:
        return DetectionResult(
            available=False,
            reason=f"지원하지 않는 모델 형식입니다: {model_kind}",
            detections=None,
        )

    ratio = damage_area_ratio(detections, original_shape)
    return DetectionResult(
        available=True,
        detections=detections,
        damage_area_ratio=ratio,
        annotated_image=draw_detections(image_bgr, detections),
        model_path=str(model_path),
        model_kind=model_kind,
        class_names=list(class_names),
        messages=[
            "이 결과는 AI 분석 결과이며 보수 여부를 확정하지 않습니다. "
            "최종 판단은 담당자 확인 결과로 별도 기록해야 합니다."
        ],
    )
