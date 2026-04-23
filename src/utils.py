from __future__ import annotations

from pathlib import Path
import cv2


def ensure_dir(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)


def load_image(path: Path):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return image


def save_image(path: Path, image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def put_lines(
    image,
    lines,
    origin=(18, 28),
    line_height=24,
    color=(255, 255, 255),
    scale=0.65,
    thickness=2,
    bg=True,
):
    x0, y0 = origin
    if bg:
        h = line_height * max(1, len(lines)) + 10
        w = 0
        for line in lines:
            (tw, _), _ = cv2.getTextSize(str(line), cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
            w = max(w, tw)
        cv2.rectangle(image, (x0 - 10, y0 - 22), (x0 + w + 14, y0 - 22 + h), (0, 0, 0), -1)
        cv2.rectangle(image, (x0 - 10, y0 - 22), (x0 + w + 14, y0 - 22 + h), (255, 255, 255), 1)

    for i, line in enumerate(lines):
        y = y0 + i * line_height
        cv2.putText(
            image,
            str(line),
            (x0, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
    return image