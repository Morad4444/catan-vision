from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
OUTPUT_DIR = DATA_DIR / "output"
STATE_DIR = DATA_DIR / "state"

BOARD_EMPTY_IMAGE = RAW_DIR / "board_empty.png"
BOARD_NUMBERS_IMAGE = RAW_DIR / "board_numbers_r.png"
BOARD_PIECES_IMAGE = RAW_DIR / "board_pieces.png"

CAMERA_INDEX = 1
WINDOW_NAME = "Catan Live Demo"

NUMBER_POOL = [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12]
LOCKED_NUMBERS = {2, 6, 8, 12}

BOARD_DEBUG_DIR = OUTPUT_DIR / "board_debug"
CHIP_DEBUG_DIR = OUTPUT_DIR / "chip_debug"
PIECES_DEBUG_DIR = OUTPUT_DIR / "pieces_debug"
STATE_DEBUG_DIR = OUTPUT_DIR / "state_debug"