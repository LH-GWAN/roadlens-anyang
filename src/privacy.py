"""개인정보 비식별 처리.

원칙:
- 얼굴은 OpenCV Haar cascade 로 탐지하여 모자이크 처리한다.
- 한국 번호판 전용 탐지모델이 없으므로, 기본값에서는 번호판 처리를 수행하지 않고
  "번호판은 처리되지 않았습니다" 라고 사실대로 표시한다.
- 보조 번호판 탐지기(러시아 번호판용 cascade)는 실험 옵션으로만 제공하며,
  사용하더라도 "제거 완료"라고 표시하지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]


FACE_STATUS_DONE = "처리함"
FACE_STATUS_UNAVAILABLE = "처리하지 못함"
PLATE_STATUS_NOT_PERFORMED = "미수행"
PLATE_STATUS_EXPERIMENTAL = "실험적 처리(정확도 미검증)"


@dataclass
class AnonymizationResult:
    """비식별 처리 결과. 화면에는 이 값을 그대로 표시한다."""

    image: np.ndarray | None = None
    face_count: int = 0
    face_status: str = FACE_STATUS_UNAVAILABLE
    plate_count: int = 0
    plate_status: str = PLATE_STATUS_NOT_PERFORMED
    messages: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "face_count": self.face_count,
            "face_status": self.face_status,
            "plate_count": self.plate_count,
            "plate_status": self.plate_status,
            "messages": list(self.messages),
        }


def _load_cascade(filename: str):
    if cv2 is None:
        return None
    path = cv2.data.haarcascades + filename
    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        return None
    return cascade


def mosaic_region(
    image: np.ndarray, x: int, y: int, w: int, h: int, blocks: int = 10
) -> np.ndarray:
    """지정 영역을 모자이크(축소 후 확대) 처리한다. 원본 배열을 직접 수정."""
    height, width = image.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(width, x + w), min(height, y + h)
    if x1 <= x0 or y1 <= y0:
        return image
    roi = image[y0:y1, x0:x1]
    small_w = max(1, min(blocks, x1 - x0))
    small_h = max(1, min(blocks, y1 - y0))
    small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    image[y0:y1, x0:x1] = cv2.resize(
        small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST
    )
    return image


def detect_faces(image_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    """정면 얼굴 + 측면 얼굴 cascade 로 얼굴 후보 영역을 찾는다."""
    if cv2 is None:
        return []
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    boxes: list[tuple[int, int, int, int]] = []
    for filename in (
        "haarcascade_frontalface_default.xml",
        "haarcascade_profileface.xml",
    ):
        cascade = _load_cascade(filename)
        if cascade is None:
            continue
        found = cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24)
        )
        for (x, y, w, h) in found:
            boxes.append((int(x), int(y), int(w), int(h)))
    return _merge_boxes(boxes)


def detect_plates_experimental(image_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    """보조 번호판 탐지(러시아 번호판용 cascade).

    한국 번호판으로 검증되지 않았으므로 결과를 신뢰할 수 없다.
    """
    if cv2 is None:
        return []
    cascade = _load_cascade("haarcascade_russian_plate_number.xml")
    if cascade is None:
        return []
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    found = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 10)
    )
    return _merge_boxes([(int(x), int(y), int(w), int(h)) for (x, y, w, h) in found])


def _merge_boxes(
    boxes: list[tuple[int, int, int, int]], iou_threshold: float = 0.3
) -> list[tuple[int, int, int, int]]:
    """겹치는 박스를 단순 병합해 중복 모자이크를 줄인다."""
    result: list[tuple[int, int, int, int]] = []
    for box in sorted(boxes, key=lambda b: b[2] * b[3], reverse=True):
        if all(_iou(box, kept) < iou_threshold for kept in result):
            result.append(box)
    return result


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def anonymize_image(
    image_bgr: np.ndarray,
    blur_faces: bool = True,
    experimental_plate_blur: bool = False,
    margin_ratio: float = 0.15,
) -> AnonymizationResult:
    """얼굴(및 선택적으로 번호판 후보)을 모자이크 처리한 이미지를 반환한다."""
    result = AnonymizationResult()

    if cv2 is None:
        result.messages.append(
            "OpenCV 가 설치되지 않아 비식별 처리를 수행할 수 없습니다. "
            "이 사진은 저장하지 마십시오."
        )
        result.image = image_bgr
        return result

    output = image_bgr.copy()

    if blur_faces:
        faces = detect_faces(output)
        for (x, y, w, h) in faces:
            mx, my = int(w * margin_ratio), int(h * margin_ratio)
            mosaic_region(output, x - mx, y - my, w + 2 * mx, h + 2 * my)
        result.face_count = len(faces)
        result.face_status = FACE_STATUS_DONE
        result.messages.append(
            f"얼굴 탐지기(Haar cascade)를 적용해 {len(faces)}개 영역을 모자이크 처리했습니다. "
            "탐지기 특성상 모든 얼굴이 탐지된다고 보장할 수 없습니다."
        )
    else:
        result.face_status = FACE_STATUS_UNAVAILABLE
        result.messages.append("설정에 따라 얼굴 비식별 처리를 수행하지 않았습니다.")

    if experimental_plate_blur:
        plates = detect_plates_experimental(output)
        for (x, y, w, h) in plates:
            mosaic_region(output, x, y, w, h)
        result.plate_count = len(plates)
        result.plate_status = PLATE_STATUS_EXPERIMENTAL
        result.messages.append(
            f"보조 번호판 탐지기로 {len(plates)}개 후보 영역을 모자이크 처리했습니다. "
            "이 탐지기는 한국 번호판으로 검증되지 않았으므로 "
            "번호판이 모두 제거되었다고 볼 수 없습니다."
        )
    else:
        result.plate_status = PLATE_STATUS_NOT_PERFORMED
        result.messages.append(
            "한국 번호판 전용 탐지모델이 없어 번호판 비식별 처리는 수행하지 않았습니다. "
            "번호판이 포함된 사진은 업로드 전에 직접 가려 주십시오."
        )

    result.image = output
    return result
