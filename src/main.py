from config import BOARD_EMPTY_IMAGE, BOARD_NUMBERS_IMAGE, BOARD_PIECES_IMAGE, OUTPUT_DIR
from utils import ensure_dir, load_image, save_image
from pathlib import Path

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


from number_detection import (
    recognize_chip_numbers,
    constrain_recognized_numbers,
    draw_recognized_numbers,
    print_recognition_summary,
    save_number_debug_images,
    save_preprocessed_templates,
)

from piece_detection import (
    estimate_tile_size_from_centers,
    generate_tile_corners_from_centers,
    analyze_corner_colors,
    draw_corner_analysis,
)

def process_board_geometry(image_bgr, prefix: str):
    contour = detect_board_contour(image_bgr)
    polygon = approximate_polygon(contour)

    if len(polygon) != 6:
        raise RuntimeError(f"{prefix}: expected 6 polygon points, got {len(polygon)}")

    points = polygon_to_points(polygon)
    ordered_points = order_hexagon_points(points)

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

    image_empty = load_image(BOARD_EMPTY_IMAGE)
    normalized_empty, ordered_empty, centers_empty = process_board_geometry(image_empty, "empty")

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

    # Analyze tile corners
    tile_size = estimate_tile_size_from_centers(centers_empty)
    tile_corners = generate_tile_corners_from_centers(centers_empty, tile_size)
    analyze_corner_colors(normalized_empty, tile_corners)

    # Draw and save corner analysis image
    corner_analysis_img = draw_corner_analysis(normalized_empty, tile_corners)
    corner_analysis_img = draw_contour(corner_analysis_img, ordered_empty)
    save_image(OUTPUT_DIR / "corner_analysis.png", corner_analysis_img)

    labeled_img = draw_tile_labels(
        draw_tile_centers(normalized_empty, centers_empty),
        centers_empty,
        labels,
    )
    labeled_img = draw_contour(labeled_img, ordered_empty)
    save_image(OUTPUT_DIR / "empty_resource_labels.png", labeled_img)

    image_numbers = load_image(BOARD_NUMBERS_IMAGE)
    normalized_numbers, ordered_numbers, centers_numbers = process_board_geometry(image_numbers, "numbers")

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

    template_dir = Path(__file__).resolve().parent.parent / "data" / "templates" / "numbers"

    recognition_results = recognize_chip_numbers(assignments, template_dir)
    recognition_results = constrain_recognized_numbers(recognition_results)

    print_recognition_summary(recognition_results)

    numbers_img = draw_recognized_numbers(normalized_numbers, recognition_results)
    numbers_img = draw_contour(numbers_img, ordered_numbers)
    save_image(OUTPUT_DIR / "numbers_recognized.png", numbers_img)

    save_number_debug_images(recognition_results, template_dir, OUTPUT_DIR / "number_debug")
    save_preprocessed_templates(template_dir, OUTPUT_DIR / "template_debug")

    print("\nRecognized chip numbers:")
    for item in recognition_results:
        top3 = ", ".join([f"{n}:{s:.2f}" for n, s in item["number_top_scores"]])
        print(
            f"Tile {item['tile_id']} ({item['label']}) -> "
            f"pred={item['predicted_number']}, "
            f"score={item['number_score']:.3f}, "
            f"rot={item['number_rotation']:.1f}, "
            f"top3=[{top3}]"
        )

    numbers_img = draw_recognized_numbers(normalized_numbers, recognition_results)
    numbers_img = draw_contour(numbers_img, ordered_numbers)
    save_image(OUTPUT_DIR / "numbers_recognized.png", numbers_img)

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