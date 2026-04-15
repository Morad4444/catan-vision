import cv2
from config import BOARD_EMPTY_IMAGE, BOARD_NUMBERS_IMAGE, BOARD_PIECES_IMAGE, OUTPUT_DIR
from utils import ensure_dir, load_image, save_image
from pathlib import Path
from number_detection import save_chip_preprocessing_debug

from board_detection import (
    detect_board_contour,
    approximate_polygon,
    polygon_to_points,
    order_hexagon_points,
    draw_contour,
    draw_points,
    generate_catan_tile_centers_from_hex,
    draw_tile_centers,
    normalize_hexagon,
    debug_board_detection,
)

from tile_classification import (
    crop_tile,
    score_tile,
    assign_resources_with_counts,
    draw_tile_labels,
)

from chip_detection import (
    detect_chips,
    assign_chips_to_tiles,
    draw_chips,
    draw_chip_tile_assignments,
    save_chip_debug_patches,
)




from piece_detection import (
    estimate_tile_size_from_centers,
    generate_tile_corners_from_centers,
    analyze_corner_colors,
    draw_corner_analysis,
)

def print_corner_hsv_values(image_bgr, points, prefix: str):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    print(f"\n{prefix}: HSV-Werte der Eckpunkte:")
    for idx, (x, y) in enumerate(points):
        xi = max(0, min(int(round(x)), hsv.shape[1] - 1))
        yi = max(0, min(int(round(y)), hsv.shape[0] - 1))
        h, s, v = hsv[yi, xi]
        print(f"  Ecke {idx}: (x={xi}, y={yi}) -> H={h}, S={s}, V={v}")


def process_board_geometry(image_bgr, prefix: str):
    contour = detect_board_contour(image_bgr)
    polygon = approximate_polygon(contour)

    if len(polygon) != 6:
        raise RuntimeError(f"{prefix}: expected 6 polygon points, got {len(polygon)}")

    points = polygon_to_points(polygon)
    ordered_points = order_hexagon_points(points)

    print_corner_hsv_values(image_bgr, ordered_points, prefix)

    hex_img = draw_contour(image_bgr, ordered_points)
    hex_img = draw_points(hex_img, ordered_points)
    save_image(OUTPUT_DIR / f"{prefix}_outer_hex.png", hex_img)

    normalized_img, normalized_points = normalize_hexagon(image_bgr, ordered_points)
    save_image(OUTPUT_DIR / f"{prefix}_normalized_hex.png", normalized_img)

    centers = generate_catan_tile_centers_from_hex(normalized_points)

    centers_img = draw_tile_centers(normalized_img, centers)
    centers_img = draw_contour(centers_img, normalized_points)
    save_image(OUTPUT_DIR / f"{prefix}_tile_centers.png", centers_img)

    return normalized_img, normalized_points, centers


def main():
    ensure_dir(OUTPUT_DIR)

    image_empty = load_image(BOARD_NUMBERS_IMAGE)
    normalized_empty, ordered_empty = debug_board_detection(image_empty, OUTPUT_DIR / "debug_empty")
    centers_empty = generate_catan_tile_centers_from_hex(ordered_empty)

    print("\nEmpty image tile centers:")
    for tile_id, x, y in centers_empty:
        print(f"Tile {tile_id}: x={x}, y={y}")

    all_scores = []
    print("\nResource scores:")
    for tile_id, x, y in centers_empty:
        tile_patch = crop_tile(normalized_empty, x, y, size=50)
        scores = score_tile(tile_patch)
        all_scores.append(scores)

        top3 = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:3]
        top_text = ", ".join([f"{name}={score:.1f}" for name, score in top3])
        print(f"Tile {tile_id}: {top_text}")

    labels = assign_resources_with_counts(all_scores)

    print("\nFinal resource labels:")
    for (tile_id, _, _), label in zip(centers_empty, labels):
        print(f"Tile {tile_id}: {label}")

    labeled_img = draw_tile_labels(
        draw_tile_centers(normalized_empty, centers_empty),
        centers_empty,
        labels,
    )
    labeled_img = draw_contour(labeled_img, ordered_empty)
    save_image(OUTPUT_DIR / "empty_resource_labels.png", labeled_img)

    image_numbers = load_image(BOARD_NUMBERS_IMAGE)
    normalized_numbers, ordered_numbers = debug_board_detection(image_numbers, OUTPUT_DIR / "debug_numbers")
    centers_numbers = generate_catan_tile_centers_from_hex(ordered_numbers)

    print("\nNumbers image tile centers:")
    for tile_id, x, y in centers_numbers:
        print(f"Tile {tile_id}: x={x}, y={y}")

    chips = detect_chips(normalized_numbers, centers_numbers)
    
    save_chip_debug_patches(chips, OUTPUT_DIR / "chip_debug")

    print("\nChip detections:")
    for item in chips:
        print(
            f"Tile {item['tile_id']}: "
            f"center=({item['tile_x']}, {item['tile_y']}), "
            f"chip=({item['chip_x']}, {item['chip_y']}), "
            f"r_detected={item['chip_r_detected']}, "
            f"r_inner={item['chip_r']}, "
            f"detected={item['detected']}"
        )

    chips_img = draw_chips(normalized_numbers, chips)
    chips_img = draw_contour(chips_img, ordered_numbers)
    save_image(OUTPUT_DIR / "numbers_chip_detections.png", chips_img)

    assignments = assign_chips_to_tiles(chips, labels)

    save_chip_preprocessing_debug(assignments, OUTPUT_DIR / "chip_preprocess_debug")



    print("\nFinal chip assignments (non-desert only):")
    for item in assignments:
        print(
            f"Tile {item['tile_id']} ({item['label']}): "
            f"chip=({item['chip_x']}, {item['chip_y']}), "
            f"r={item['chip_r']}, detected={item['detected']}"
        )

    assign_img = draw_chip_tile_assignments(normalized_numbers, assignments)
    assign_img = draw_contour(assign_img, ordered_numbers)
    
    save_image(OUTPUT_DIR / "numbers_chip_assignments.png", assign_img)

    print("\nSaved outputs in:")
    print(OUTPUT_DIR)

    # Analyze tile corners
    image_pieces = load_image(BOARD_PIECES_IMAGE)
    normalized_pieces, ordered_pieces, centers_pices = process_board_geometry(image_pieces, "empty")

    tile_size = estimate_tile_size_from_centers(centers_pices)
    tile_corners = generate_tile_corners_from_centers(centers_pices, tile_size)
    analyze_corner_colors(normalized_pieces, tile_corners)

    # Draw and save corner analysis image
    corner_analysis_img = draw_corner_analysis(normalized_pieces, tile_corners)
    corner_analysis_img = draw_contour(corner_analysis_img, ordered_pieces)
    save_image(OUTPUT_DIR / "corner_analysis.png", corner_analysis_img)

#    # Process board_pieces.png
#    image_pieces = load_image(BOARD_PIECES_IMAGE)
#    # Use the same outer points as empty for normalization, since board is the same
#    normalized_pieces, _ = normalize_hexagon(image_pieces, ordered_empty)
#    save_image(OUTPUT_DIR / f"pieces_normalized_hex.png", normalized_pieces)
#
#    # Use centers from empty, assuming same board layout
#    centers_pieces = centers_empty
#
#    # Analyze corners for pieces
#    tile_size_pieces = estimate_tile_size_from_centers(centers_pieces)
#    tile_corners_pieces = generate_tile_corners_from_centers(centers_pieces, tile_size_pieces)
#    analyze_corner_colors(normalized_pieces, tile_corners_pieces)
#
#    # Draw and save corner analysis for pieces
#    corner_analysis_pieces_img = draw_corner_analysis(normalized_pieces, tile_corners_pieces)
#    corner_analysis_pieces_img = draw_contour(corner_analysis_pieces_img, ordered_empty)
#    save_image(OUTPUT_DIR / "pieces_corner_analysis.png", corner_analysis_pieces_img)


if __name__ == "__main__":
    main()