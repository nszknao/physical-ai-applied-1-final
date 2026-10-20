"""Detect the largest red region in an RGB camera image."""

from dataclasses import dataclass

import cv2
import numpy as np

MIN_TARGET_AREA = 25  # Connected pixels, at the 320x240 camera resolution.


@dataclass(frozen=True, kw_only=True)
class Detection:
    """Image coordinates: origin at top left, x rightward and y downward."""

    bbox: tuple[int, int, int, int]  # x, y, width, height; right/bottom exclusive.
    center: tuple[float, float]  # Centroid of the detected pixels.
    area: int  # Red pixel count, not bounding-box area or confidence.


def detect_target(rgb: np.ndarray) -> Detection | None:
    """Return the largest red component, or None when no component is large enough."""
    if (
        rgb.ndim != 3
        or rgb.shape[2] != 3
        or min(rgb.shape[:2]) == 0
        or rgb.dtype != np.uint8
    ):
        raise ValueError("Expected a nonempty HxWx3 uint8 RGB image")
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # OpenCV hue is 0..179. Red spans both ends of that interval.
    mask = cv2.inRange(hsv, (0, 120, 50), (10, 255, 255))
    mask |= cv2.inRange(hsv, (170, 120, 50), (179, 255, 255))
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return None
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[label, cv2.CC_STAT_AREA])
    if area < MIN_TARGET_AREA:
        return None
    x, y, width, height = map(int, stats[label, :4])
    return Detection(
        bbox=(x, y, width, height),
        center=tuple(map(float, centroids[label])),
        area=area,
    )


def estimate_distance(
    detection: Detection | None,
    *,
    image_width: int,
    image_height: int,
    vertical_fov: float,
    target_height: float = 0.08,
) -> float | None:
    """Approximate optical-axis range from known height; reject clipped boxes."""
    if (
        image_width <= 0
        or image_height <= 0
        or not np.isfinite(vertical_fov)
        or not 0 < vertical_fov < 180
        or not np.isfinite(target_height)
        or target_height <= 0
    ):
        raise ValueError("Invalid camera geometry or target height")
    if detection is None:
        return None
    x, y, width, height = detection.bbox
    if (
        width <= 0
        or height <= 0
        or x <= 1
        or y <= 1
        or x + width >= image_width - 1
        or y + height >= image_height - 1
    ):
        return None
    focal = image_height / (2 * np.tan(np.radians(vertical_fov) / 2))
    return float(focal * target_height / height)


def annotate_detection(
    rgb: np.ndarray,
    detection: Detection | None,
    *,
    status: str | None = None,
) -> np.ndarray:
    """Draw diagnostic marks on a copy, keeping the recognition input untouched."""
    image = rgb.copy()
    if detection is None:
        label = "HEAD | Target: not detected"
    else:
        x, y, width, height = detection.bbox
        cx, cy = detection.center
        cv2.rectangle(image, (x, y), (x + width - 1, y + height - 1), (0, 255, 0), 1)
        cv2.drawMarker(
            image, (round(cx), round(cy)), (255, 255, 0), cv2.MARKER_CROSS, 7, 1
        )
        label = f"HEAD | x={cx:.0f} y={cy:.0f} area={detection.area}px"
    # A dark outline keeps the status legible against either sky or floor.
    for color, thickness in (((0, 0, 0), 3), ((255, 255, 255), 1)):
        cv2.putText(
            image,
            label,
            (6, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            thickness,
            cv2.LINE_AA,
        )
    if status is not None:
        lines = status.splitlines()
        for index, line in enumerate(lines):
            origin = (6, image.shape[0] - 10 - 16 * (len(lines) - 1 - index))
            for color, thickness in (((0, 0, 0), 3), ((255, 255, 255), 1)):
                cv2.putText(
                    image,
                    line,
                    origin,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    color,
                    thickness,
                    cv2.LINE_AA,
                )
    return image
