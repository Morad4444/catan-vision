from __future__ import annotations

from pathlib import Path
import random
import time
import json
import shutil
import cv2
import numpy as np


NUMBER_POOL = [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12]
LOCKED_NUMBERS = {2, 6, 8, 12}
EDGE_SIZE = 112

TILE_NEIGHBORS = {
    0: [1, 3, 4],
    1: [0, 2, 4, 5],
    2: [1, 5, 6],
    3: [0, 4, 7, 8],
    4: [0, 1, 3, 5, 8, 9],
    5: [1, 2, 4, 6, 9, 10],
    6: [2, 5, 10, 11],
    7: [3, 8, 12],
    8: [3, 4, 7, 9, 12, 13],
    9: [4, 5, 8, 10, 13, 14],
    10: [5, 6, 9, 11, 14, 15],
    11: [6, 10, 15],
    12: [7, 8, 13, 16],
    13: [8, 9, 12, 14, 16, 17],
    14: [9, 10, 13, 15, 17, 18],
    15: [10, 11, 14, 18],
    16: [12, 13, 17],
    17: [13, 14, 16, 18],
    18: [14, 15, 17],
}


# --------------------------------------------------
# Number layout generation
# --------------------------------------------------

def _can_place_special(number_map: dict[int, int], tile_id: int, number: int) -> bool:
    if number not in (6, 8):
        return True
    for nb in TILE_NEIGHBORS.get(tile_id, []):
        if number_map.get(nb) in (6, 8):
            return False
    return True


def generate_random_number_layout(
    centers: list,
    resource_labels: list[str],
    max_attempts: int = 5000,
) -> dict[int, int]:
    non_desert_tiles = [tile_id for tile_id, _, _ in centers if resource_labels[tile_id] != "Desert"]

    if len(non_desert_tiles) != 18:
        raise RuntimeError(f"Expected 18 non-desert tiles, got {len(non_desert_tiles)}")

    rng = random.Random()

    for _ in range(max_attempts):
        pool = NUMBER_POOL[:]
        rng.shuffle(pool)

        specials = [n for n in pool if n in (6, 8)]
        others = [n for n in pool if n not in (6, 8)]

        number_map: dict[int, int] = {}
        shuffled_tiles = non_desert_tiles[:]
        rng.shuffle(shuffled_tiles)

        placed_specials = 0
        for tile_id in shuffled_tiles:
            if placed_specials >= len(specials):
                break
            n = specials[placed_specials]
            if _can_place_special(number_map, tile_id, n):
                number_map[tile_id] = n
                placed_specials += 1

        if placed_specials != len(specials):
            continue

        remaining_tiles = [t for t in non_desert_tiles if t not in number_map]
        rng.shuffle(remaining_tiles)

        for tile_id, n in zip(remaining_tiles, others):
            number_map[tile_id] = n

        ok = True
        for tile_id, n in number_map.items():
            if n in (6, 8):
                without_self = {k: v for k, v in number_map.items() if k != tile_id}
                if not _can_place_special(without_self, tile_id, n):
                    ok = False
                    break

        if ok:
            return number_map

    raise RuntimeError("Could not generate a legal number layout.")


# --------------------------------------------------
# Edge preprocessing
# --------------------------------------------------

def _blank_edges() -> np.ndarray:
    return np.zeros((EDGE_SIZE, EDGE_SIZE), dtype=np.uint8)


def _center_crop_square(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    side = min(h, w)
    x1 = (w - side) // 2
    y1 = (h - side) // 2
    return image[y1:y1 + side, x1:x1 + side].copy()


def _circular_mask(shape: tuple[int, int], radius_scale: float = 0.88) -> np.ndarray:
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cx = w // 2
    cy = h // 2
    r = int(round(min(h, w) * 0.5 * radius_scale))
    cv2.circle(mask, (cx, cy), r, 255, -1)
    return mask


def preprocess_chip_edges(chip_patch_bgr: np.ndarray | None, output_size: int = EDGE_SIZE) -> np.ndarray | None:
    if chip_patch_bgr is None or chip_patch_bgr.size == 0:
        return None

    patch = _center_crop_square(chip_patch_bgr)
    patch = cv2.resize(patch, (output_size, output_size), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    mask = _circular_mask(gray.shape, radius_scale=0.88)
    gray = cv2.bitwise_and(gray, gray, mask=mask)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    vals = gray[mask > 0]
    if len(vals) > 0:
        mn = float(np.min(vals))
        mx = float(np.max(vals))
        if mx > mn:
            gray = ((gray.astype(np.float32) - mn) * (255.0 / (mx - mn))).clip(0, 255).astype(np.uint8)

    edges = cv2.Canny(gray, 50, 120)
    edges = cv2.bitwise_and(edges, edges, mask=mask)
    return edges


def edge_similarity(edges_a: np.ndarray | None, edges_b: np.ndarray | None) -> float:
    """
    Structure-based score, 0..1 roughly.
    Higher is better.
    """
    if edges_a is None or edges_b is None:
        return -1.0
    if edges_a.shape != edges_b.shape:
        return -1.0

    a = (edges_a > 0).astype(np.uint8) * 255
    b = (edges_b > 0).astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)
    a = cv2.dilate(a, kernel, iterations=1)
    b = cv2.dilate(b, kernel, iterations=1)

    def features(img: np.ndarray) -> dict[str, int]:
        white = int(np.count_nonzero(img))
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=8)

        comp_areas = []
        xs, ys, xe, ye = [], [], [], []
        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < 8:
                continue
            comp_areas.append(area)
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            xs.append(x)
            ys.append(y)
            xe.append(x + w)
            ye.append(y + h)

        comp_count = len(comp_areas)
        if comp_count == 0:
            bbox_w = 0
            bbox_h = 0
        else:
            bbox_w = max(xe) - min(xs)
            bbox_h = max(ye) - min(ys)

        return {
            "white": white,
            "comp_count": comp_count,
            "bbox_w": bbox_w,
            "bbox_h": bbox_h,
        }

    fa = features(a)
    fb = features(b)

    white_sim = 1.0 - abs(fa["white"] - fb["white"]) / max(1, max(fa["white"], fb["white"]))
    comp_sim = 1.0 if fa["comp_count"] == fb["comp_count"] else max(
        0.0, 1.0 - abs(fa["comp_count"] - fb["comp_count"]) / 3.0
    )
    bw_sim = 1.0 - abs(fa["bbox_w"] - fb["bbox_w"]) / max(1, max(fa["bbox_w"], fb["bbox_w"], 1))
    bh_sim = 1.0 - abs(fa["bbox_h"] - fb["bbox_h"]) / max(1, max(fa["bbox_h"], fb["bbox_h"], 1))

    score = 0.40 * white_sim + 0.30 * comp_sim + 0.15 * bw_sim + 0.15 * bh_sim
    return float(score)


# --------------------------------------------------
# Folder helpers
# --------------------------------------------------

def _temps_dir(base_dir: str | Path) -> Path:
    return Path(base_dir) / "chips_temps"


def _current_dir(base_dir: str | Path) -> Path:
    return Path(base_dir) / "chips_current"


def _edge_path(directory: Path, tile_id: int) -> Path:
    return directory / f"tile_{tile_id}_edges.png"


def _safe_imread_gray(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def save_reference_edges(chip_assignments: list[dict], base_dir: str | Path) -> None:
    temps = _temps_dir(base_dir)
    current = _current_dir(base_dir)
    temps.mkdir(parents=True, exist_ok=True)
    current.mkdir(parents=True, exist_ok=True)

    for item in chip_assignments:
        tile_id = int(item["tile_id"])
        edges = preprocess_chip_edges(item.get("chip_patch"))
        if edges is None:
            edges = _blank_edges()
        cv2.imwrite(str(_edge_path(temps, tile_id)), edges)


def save_current_edges(chip_assignments: list[dict], base_dir: str | Path) -> None:
    current = _current_dir(base_dir)
    current.mkdir(parents=True, exist_ok=True)

    for item in chip_assignments:
        tile_id = int(item["tile_id"])
        edges = preprocess_chip_edges(item.get("chip_patch"))
        if edges is None:
            edges = _blank_edges()
        cv2.imwrite(str(_edge_path(current, tile_id)), edges)


def update_reference_edges_for_tiles(tile_ids: list[int], base_dir: str | Path) -> None:
    temps = _temps_dir(base_dir)
    current = _current_dir(base_dir)
    temps.mkdir(parents=True, exist_ok=True)

    for tile_id in tile_ids:
        src = _edge_path(current, tile_id)
        dst = _edge_path(temps, tile_id)
        if src.exists():
            shutil.copy2(src, dst)


# --------------------------------------------------
# Board state
# --------------------------------------------------

def create_manual_board_state(
    chip_assignments: list[dict],
    number_map: dict[int, int],
    save_dir: str | Path | None = None,
) -> dict:
    state = {
        "created_at": time.time(),
        "number_map": {int(k): int(v) for k, v in number_map.items()},
        "last_swap_time": 0.0,
        "pending_swap": {"pair": None, "count": 0},
        "pending_refresh_tiles": set(),
        "refresh_counter": 0,
        "refresh_prev_edges": {},
    }

    if save_dir is not None:
        save_board_state(state, save_dir)
        save_reference_edges(chip_assignments, save_dir)

    return state


def save_board_state(state: dict, save_dir: str | Path) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "created_at": state["created_at"],
        "number_map": state["number_map"],
        "last_swap_time": state.get("last_swap_time", 0.0),
    }
    (save_dir / "board_state.json").write_text(json.dumps(payload, indent=2))


def load_board_state(save_dir: str | Path) -> dict:
    save_dir = Path(save_dir)
    meta = json.loads((save_dir / "board_state.json").read_text())
    return {
        "created_at": meta["created_at"],
        "number_map": {int(k): int(v) for k, v in meta["number_map"].items()},
        "last_swap_time": meta.get("last_swap_time", 0.0),
        "pending_swap": {"pair": None, "count": 0},
        "pending_refresh_tiles": set(),
        "refresh_counter": 0,
        "refresh_prev_edges": {},
    }


# --------------------------------------------------
# Analysis
# --------------------------------------------------

def analyze_chip_identities(
    chip_assignments: list[dict],
    state: dict,
    debug_dir: str | Path | None = None,
) -> list[dict]:
    base_dir = Path(debug_dir) if debug_dir is not None else Path("data/output/board_state_live")
    temps = _temps_dir(base_dir)
    current = _current_dir(base_dir)

    save_current_edges(chip_assignments, base_dir)

    reports = []

    for item in chip_assignments:
        tile_id = int(item["tile_id"])

        curr_img = _safe_imread_gray(_edge_path(current, tile_id))
        self_img = _safe_imread_gray(_edge_path(temps, tile_id))
        if curr_img is None:
            curr_img = _blank_edges()
        if self_img is None:
            self_img = _blank_edges()

        self_score = edge_similarity(curr_img, self_img)

        best_other_tile = None
        best_other_score = -1.0

        for other_item in chip_assignments:
            other_tile = int(other_item["tile_id"])
            if other_tile == tile_id:
                continue

            other_temp = _safe_imread_gray(_edge_path(temps, other_tile))
            if other_temp is None:
                other_temp = _blank_edges()

            score = edge_similarity(curr_img, other_temp)
            if score > best_other_score:
                best_other_score = score
                best_other_tile = other_tile

        improvement = best_other_score - self_score

        reports.append({
            "tile_id": tile_id,
            "current_number": state["number_map"][tile_id],
            "self_score": float(self_score),
            "best_other_tile": best_other_tile,
            "best_other_score": float(best_other_score),
            "improvement": float(improvement),
        })

    return reports


# --------------------------------------------------
# Swap detection
# --------------------------------------------------

def detect_pair_swaps(
    identity_report: list[dict],
    state: dict,
    debug_dir: str | Path,
    mismatch_threshold: float = 0.72,
    improvement_threshold: float = 0.18,
    pair_match_threshold: float = 0.82,
    require_consecutive_frames: int = 4,
    cooldown_seconds: float = 2.0,
) -> list[dict]:
    now = time.time()

    if now - state.get("last_swap_time", 0.0) < cooldown_seconds:
        return []

    pending = state.setdefault("pending_swap", {"pair": None, "count": 0})

    base_dir = Path(debug_dir)
    temps = _temps_dir(base_dir)
    current = _current_dir(base_dir)

    changed = [
        r for r in identity_report
        if r["self_score"] < mismatch_threshold and r["improvement"] > improvement_threshold
    ]

    if len(changed) != 2:
        pending["pair"] = None
        pending["count"] = 0
        return []

    a = changed[0]["tile_id"]
    b = changed[1]["tile_id"]

    if changed[0]["best_other_tile"] != b or changed[1]["best_other_tile"] != a:
        pending["pair"] = None
        pending["count"] = 0
        return []

    curr_a = _safe_imread_gray(_edge_path(current, a))
    curr_b = _safe_imread_gray(_edge_path(current, b))
    temp_a = _safe_imread_gray(_edge_path(temps, a))
    temp_b = _safe_imread_gray(_edge_path(temps, b))

    if curr_a is None or curr_b is None or temp_a is None or temp_b is None:
        pending["pair"] = None
        pending["count"] = 0
        return []

    score_ab = edge_similarity(curr_a, temp_b)
    score_ba = edge_similarity(curr_b, temp_a)

    if score_ab < pair_match_threshold or score_ba < pair_match_threshold:
        pending["pair"] = None
        pending["count"] = 0
        return []

    num_a = state["number_map"][a]
    num_b = state["number_map"][b]

    if num_a == num_b:
        pending["pair"] = None
        pending["count"] = 0
        return []

    if num_a in LOCKED_NUMBERS or num_b in LOCKED_NUMBERS:
        pending["pair"] = None
        pending["count"] = 0
        return []

    pair = tuple(sorted((a, b)))

    if pending["pair"] == pair:
        pending["count"] += 1
    else:
        pending["pair"] = pair
        pending["count"] = 1

    if pending["count"] < require_consecutive_frames:
        return []

    pending["pair"] = None
    pending["count"] = 0

    return [{
        "tile_a": a,
        "tile_b": b,
        "number_a": num_a,
        "number_b": num_b,
        "score_ab": float(score_ab),
        "score_ba": float(score_ba),
    }]


def apply_detected_swaps(state: dict, swaps: list[dict], base_dir: str | Path) -> list[dict]:
    applied = []

    for sw in swaps:
        a = sw["tile_a"]
        b = sw["tile_b"]

        if a == b:
            continue

        old_a = state["number_map"][a]
        old_b = state["number_map"][b]

        if old_a == old_b:
            continue

        state["number_map"][a], state["number_map"][b] = old_b, old_a
        applied.append(sw)

    if applied:
        state["last_swap_time"] = time.time()

        pending_tiles = state.setdefault("pending_refresh_tiles", set())
        for sw in applied:
            pending_tiles.add(sw["tile_a"])
            pending_tiles.add(sw["tile_b"])

        state["refresh_counter"] = 0
        state["refresh_prev_edges"] = {}
        save_board_state(state, base_dir)

    return applied


def refresh_pending_reference_edges(
    chip_assignments: list[dict],
    state: dict,
    base_dir: str | Path,
    stable_similarity_threshold: float = 0.94,
    require_consecutive_frames: int = 5,
    min_delay_seconds: float = 1.2,
) -> bool:
    pending_tiles: set[int] = state.get("pending_refresh_tiles", set())
    if not pending_tiles:
        return False

    if time.time() - state.get("last_swap_time", 0.0) < min_delay_seconds:
        return False

    base_dir = Path(base_dir)
    current_dir = _current_dir(base_dir)

    # Current edges are already written by analyze_chip_identities(), but writing again is harmless.
    save_current_edges(chip_assignments, base_dir)

    prev_edges: dict[int, np.ndarray] = state.setdefault("refresh_prev_edges", {})
    all_good = True

    for tile_id in pending_tiles:
        curr = _safe_imread_gray(_edge_path(current_dir, tile_id))
        if curr is None:
            curr = _blank_edges()

        prev = prev_edges.get(tile_id)
        if prev is None:
            all_good = False
        else:
            score = edge_similarity(curr, prev)
            if score < stable_similarity_threshold:
                all_good = False

        prev_edges[tile_id] = curr.copy()

    if all_good:
        state["refresh_counter"] = state.get("refresh_counter", 0) + 1
    else:
        state["refresh_counter"] = 0

    if state["refresh_counter"] < require_consecutive_frames:
        return False

    update_reference_edges_for_tiles(sorted(pending_tiles), base_dir)
    state["pending_refresh_tiles"] = set()
    state["refresh_counter"] = 0
    state["refresh_prev_edges"] = {}
    save_board_state(state, base_dir)
    return True


def print_swap_detected(swaps: list[dict]) -> None:
    for sw in swaps:
        print(f"swap detected: {sw['number_a']} <-> {sw['number_b']}")