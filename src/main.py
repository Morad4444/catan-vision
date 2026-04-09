import numpy as np
import matplotlib.pyplot as plt

from config import BOARD_EMPTY_IMAGE, OUTPUT_DIR
from utils import ensure_dir, load_image, save_image, convert_bgr_to_rgb
from board_detection import (
    detect_board_contour,
    approximate_polygon,
    polygon_to_points,
    order_hexagon_points,
    draw_contour,
    draw_points,
    generate_catan_tile_centers_from_hex,
    draw_tile_centers,
)


def save_plot(image_bgr, title, output_path):
    image_rgb = convert_bgr_to_rgb(image_bgr)
    plt.figure(figsize=(8, 8))
    plt.imshow(image_rgb)
    plt.title(title)
    plt.axis("off")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def main():
    ensure_dir(OUTPUT_DIR)

    # 1) load image
    image_bgr = load_image(BOARD_EMPTY_IMAGE)

    # 2) detect board contour
    contour = detect_board_contour(image_bgr)
    polygon = approximate_polygon(contour)

    if len(polygon) != 6:
        raise RuntimeError(
            f"Expected 6 polygon points for the board, but got {len(polygon)}"
        )

    # 3) save contour image
    contour_image = draw_contour(image_bgr, polygon)
    contour_path = OUTPUT_DIR / "board_polygon.png"
    save_image(contour_path, contour_image)
    save_plot(contour_image, "Detected Board Polygon", OUTPUT_DIR / "board_polygon_plot.png")

    # 4) order outer points
    points = polygon_to_points(polygon)
    ordered_points = order_hexagon_points(points)

    points_image = draw_points(image_bgr, ordered_points, color=(0, 0, 255), radius=10)
    points_image = draw_contour(points_image, polygon)
    points_path = OUTPUT_DIR / "board_points.png"
    save_image(points_path, points_image)
    save_plot(points_image, "Ordered Outer Board Points", OUTPUT_DIR / "board_points_plot.png")

    print("\nOrdered outer hexagon points:")
    for i, (x, y) in enumerate(ordered_points):
        print(f"Point {i}: x={x:.2f}, y={y:.2f}")

    # 5) print vector from point 5 to point 2
    p5 = ordered_points[5]
    p2 = ordered_points[2]

    dx = p2[0] - p5[0]
    dy = p2[1] - p5[1]
    dist = np.sqrt(dx**2 + dy**2)

    print("\nMeasurement from outer point 5 to outer point 2:")
    print(f"Point 5: ({p5[0]:.2f}, {p5[1]:.2f})")
    print(f"Point 2: ({p2[0]:.2f}, {p2[1]:.2f})")
    print(f"dx = {dx:.2f}")
    print(f"dy = {dy:.2f}")
    print(f"distance = {dist:.2f}")

    print("\nEqual step along 5 -> 2:")
    print(f"step_dx = {dx / 4.0:.2f}")
    print(f"step_dy = {dy / 4.0:.2f}")

    # 6) generate tile centers directly on original image
    centers = generate_catan_tile_centers_from_hex(ordered_points)

    centers_image = draw_tile_centers(image_bgr, centers)
    centers_image = draw_contour(centers_image, polygon)

    centers_path = OUTPUT_DIR / "tile_centers_on_original.png"
    save_image(centers_path, centers_image)
    save_plot(
        centers_image,
        "Tile Centers on Original Image",
        OUTPUT_DIR / "tile_centers_on_original_plot.png",
    )

    print("\nGenerated tile centers:")
    for tile_id, x, y in centers:
        print(f"Tile {tile_id}: x={x}, y={y}")

    print("\nSaved files:")
    print(f"- {contour_path}")
    print(f"- {points_path}")
    print(f"- {centers_path}")


if __name__ == "__main__":
    main()