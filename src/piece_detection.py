import cv2
import numpy as np


def estimate_tile_size_from_centers(centers):
    """
    Estimate tile side length from nearest center spacing.
    """
    pts = np.array([(x, y) for _, x, y in centers], dtype=np.float32)

    nearest = []
    for i in range(len(pts)):
        dists = []
        for j in range(len(pts)):
            if i == j:
                continue
            dists.append(np.linalg.norm(pts[i] - pts[j]))
        nearest.append(min(dists))

    median_nn = float(np.median(nearest))
    tile_size = median_nn / np.sqrt(3.0)  # Exact for hexagon grid
    return tile_size


def regular_hexagon_points(side_length, center, rotation_rad=0.0):
    """Return regular hexagon vertices in the order expected by order_hexagon_points."""
    a = float(side_length)
    h = a * np.sqrt(3.0) / 2.0
    pts = np.array(
        [
            [-a / 2.0, -h],
            [a / 2.0, -h],
            [a, 0.0],
            [a / 2.0, h],
            [-a / 2.0, h],
            [-a, 0.0],
        ],
        dtype=np.float32,
    )

    if rotation_rad != 0.0:
        c = np.cos(rotation_rad)
        s = np.sin(rotation_rad)
        rot = np.array([[c, -s], [s, c]], dtype=np.float32)
        pts = pts.dot(rot.T)

    return pts + np.array(center, dtype=np.float32)


def generate_tile_corners_from_centers(centers, tile_size):
    """
    Generate corners for each of the 19 tiles.
    Returns a list of lists: for each tile, a list of 6 (x,y) corner points.
    """
    corners = []
    rotation_rad = np.pi / 6  # 30 degrees rotation to match Catan tile orientation
    for tile_id, cx, cy in centers:
        center = np.array([cx, cy], dtype=np.float32)
        tile_corners = regular_hexagon_points(tile_size, center, rotation_rad)
        corners.append(tile_corners)
    return corners


def analyze_corner_colors(image_bgr, corners, patch_size=9):
    """
    Analyze HSV colors at each corner point and print to console.
    corners: list of lists, each sublist has 6 (x,y) points for one tile.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    for tile_idx, tile_corners in enumerate(corners):
        print(f"\nTile {tile_idx} corners:")
        for corner_idx, (x, y) in enumerate(tile_corners):
            x, y = int(round(x)), int(round(y))
            h, w = image_bgr.shape[:2]

            # Crop small patch around corner
            x1 = max(0, x - patch_size // 2)
            y1 = max(0, y - patch_size // 2)
            x2 = min(w, x + patch_size // 2)
            y2 = min(h, y + patch_size // 2)

            patch = hsv[y1:y2, x1:x2]
            if patch.size == 0:
                print(f"  Corner {corner_idx}: no patch")
                continue

            mean_hsv = np.mean(patch.reshape(-1, 3), axis=0)
            h_val, s_val, v_val = mean_hsv
            print(f"  Corner {corner_idx}: H={h_val:.1f}, S={s_val:.1f}, V={v_val:.1f}")


def draw_corner_analysis(image_bgr, corners, patch_size=10):
    """
    Draw the corners and their analysis ROIs on the image.
    Returns the annotated image.
    """
    img = image_bgr.copy()
    h, w = img.shape[:2]

    for tile_idx, tile_corners in enumerate(corners):
        for corner_idx, (x, y) in enumerate(tile_corners):
            x, y = int(round(x)), int(round(y))

            # Draw the corner point
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)  # Green dot for corner

            # Draw the ROI rectangle
            x1 = max(0, x - patch_size // 2)
            y1 = max(0, y - patch_size // 2)
            x2 = min(w, x + patch_size // 2)
            y2 = min(h, y + patch_size // 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)  # Blue rectangle for ROI

            # Label the corner
            cv2.putText(
                img,
                f"T{tile_idx}C{corner_idx}",
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    return img


