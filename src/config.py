from pathlib import Path


# Project root = folder that contains README.md, src/, data/, etc.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data folders
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
OUTPUT_DIR = DATA_DIR / "output"

# Default input images
BOARD_EMPTY_IMAGE = RAW_DIR / "board_empty.png"
BOARD_NUMBERS_IMAGE = RAW_DIR / "board_numbers.png"
BOARD_PIECES_IMAGE = RAW_DIR / "board_pieces.png"