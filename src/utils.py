from pathlib import Path

import cv2


def ensure_dir(directory: Path) -> None:
    """
    Create directory if it does not already exist.
    """
    directory.mkdir(parents=True, exist_ok=True)


def load_image(image_path: Path):
    """
    Load an image with OpenCV.
    Raises FileNotFoundError if the image does not exist or cannot be read.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return image


def save_image(image_path: Path, image) -> None:
    """
    Save an image with OpenCV.
    """
    success = cv2.imwrite(str(image_path), image)
    if not success:
        raise IOError(f"Could not save image to: {image_path}")



def convert_bgr_to_rgb(image_bgr):
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)