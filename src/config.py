from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"

BOARD_EMPTY_IMAGE = DATA_DIR / "board_empty.png"
BOARD_NUMBERS_IMAGE = DATA_DIR / "board_numbers_r.png"
BOARD_PIECES_IMAGE = DATA_DIR / "board_pieces.png"
