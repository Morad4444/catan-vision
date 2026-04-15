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