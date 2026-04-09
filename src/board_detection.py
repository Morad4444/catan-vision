import cv2
import numpy as np


def detect_board_contour(image_bgr):
    """
    Detect the Catan board using its blue outer region.
    Returns the largest contour found in the blue mask.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Blue range for the ocean border
    lower_blue = np.array([90, 80, 80], dtype=np.uint8)
    upper_blue = np.array([130, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise RuntimeError("No contours found in blue mask.")

    largest = max(contours, key=cv2.contourArea)
    return largest


def approximate_polygon(contour):
    """
    Approximate a contour to a simpler polygon.
    For the Catan board we expect 6 points.
    """
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    return approx


def polygon_to_points(polygon):
    """
    Convert OpenCV polygon shape (N,1,2) into NumPy array (N,2).
    """
    return polygon.reshape(-1, 2).astype(np.float32)


def order_hexagon_points(points):
    """
    Order 6 outer board points as:
    0 = top-left
    1 = top-right
    2 = right
    3 = bottom-right
    4 = bottom-left
    5 = left
    """
    if len(points) != 6:
        raise ValueError(f"Expected 6 points, got {len(points)}")

    pts = np.array(points, dtype=np.float32)

    # sort by y coordinate
    y_sorted = pts[np.argsort(pts[:, 1])]

    # top two, middle two, bottom two
    top_two = y_sorted[:2]
    mid_two = y_sorted[2:4]
    bottom_two = y_sorted[4:]

    # sort left/right inside each group
    top_two = top_two[np.argsort(top_two[:, 0])]
    mid_two = mid_two[np.argsort(mid_two[:, 0])]
    bottom_two = bottom_two[np.argsort(bottom_two[:, 0])]

    top_left, top_right = top_two
    left_mid, right_mid = mid_two
    bottom_left, bottom_right = bottom_two

    ordered = np.array(
        [
            top_left,      # 0
            top_right,     # 1
            right_mid,     # 2
            bottom_right,  # 3
            bottom_left,   # 4
            left_mid,      # 5
        ],
        dtype=np.float32,
    )

    return ordered


def draw_contour(image_bgr, contour, color=(0, 255, 0), thickness=3):
    """
    Draw a contour on a copy of the image.
    """
    image_copy = image_bgr.copy()
    cv2.drawContours(image_copy, [contour], -1, color, thickness)
    return image_copy


def draw_points(image_bgr, points, color=(0, 0, 255), radius=10):
    """
    Draw numbered points on the image.
    """
    image_copy = image_bgr.copy()

    for i, (x, y) in enumerate(points):
        x_i, y_i = int(x), int(y)
        cv2.circle(image_copy, (x_i, y_i), radius, color, -1)
        cv2.putText(
            image_copy,
            str(i),
            (x_i + 10, y_i - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return image_copy


def get_bounding_quad(points):
    """
    Convert the ordered outer hexagon points into a bounding rectangle.
    Returns:
    top-left, top-right, bottom-right, bottom-left
    """
    x_min = np.min(points[:, 0])
    x_max = np.max(points[:, 0])
    y_min = np.min(points[:, 1])
    y_max = np.max(points[:, 1])

    quad = np.array(
        [
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max],
        ],
        dtype=np.float32,
    )
    return quad


def warp_board_to_square(image_bgr, src_points, size=1000):
    """
    Warp the board into a square image using the bounding rectangle.
    Returns:
    warped_image, source_quad
    """
    quad = get_bounding_quad(src_points)

    dst = np.array(
        [
            [0, 0],
            [size - 1, 0],
            [size - 1, size - 1],
            [0, size - 1],
        ],
        dtype=np.float32,
    )

    H = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(image_bgr, H, (size, size))

    return warped, quad


def transform_points(points, H):
    """
    Apply homography H to a set of 2D points of shape (N,2).
    """
    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(pts, H)
    return transformed.reshape(-1, 2)


def generate_catan_tile_centers_from_hex(ordered_points):
    """
    Generate the 19 tile centers directly from the outer hexagon geometry
    on the ORIGINAL image.

    Point convention:
    0 = top-left
    1 = top-right
    2 = right
    3 = bottom-right
    4 = bottom-left
    5 = left
    """
    import numpy as np

    p0 = ordered_points[0]
    p1 = ordered_points[1]
    p2 = ordered_points[2]
    p3 = ordered_points[3]
    p4 = ordered_points[4]
    p5 = ordered_points[5]

    # Midpoints of top and bottom board edges
    top_mid = (p0 + p1) / 2.0
    bottom_mid = (p4 + p3) / 2.0

    # Center tile (tile 9) anchor
    center = (top_mid + bottom_mid) / 2.0

    # Correct center-to-center step vectors
    scale = 0.78
    # left outer vertex -> right outer vertex = 5 horizontal tile steps
    step_x = (p2 - p5) / 5.0 * scale

    # top outer edge midpoint -> bottom outer edge midpoint = 5 row steps
    step_y = (bottom_mid - top_mid) / 5.0 * scale

    row_lengths = [3, 4, 5, 4, 3]
    row_offsets = [-2, -1, 0, 1, 2]

    centers = []
    tile_id = 0

    for row_len, row_offset in zip(row_lengths, row_offsets):
        row_center = center + row_offset * step_y
        row_start = row_center - ((row_len - 1) / 2.0) * step_x

        for i in range(row_len):
            pos = row_start + i * step_x
            centers.append((tile_id, int(round(pos[0])), int(round(pos[1]))))
            tile_id += 1

    return centers

def draw_tile_centers(image_bgr, centers):
    """
    Draw tile centers and ids on the image.
    """
    image = image_bgr.copy()

    for tile_id, x, y in centers:
        cv2.circle(image, (x, y), 10, (0, 0, 255), -1)

        cv2.line(image, (x - 12, y), (x + 12, y), (255, 255, 255), 2)
        cv2.line(image, (x, y - 12), (x, y + 12), (255, 255, 255), 2)

        cv2.putText(
            image,
            str(tile_id),
            (x + 12, y - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return image