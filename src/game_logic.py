from __future__ import annotations

from collections import defaultdict
import re


def _resource_amount_for_house(house: dict) -> int:
    """
    Normal settlement = 1 resource.
    Later, if you add cities, set house["kind"] = "city" and it gives 2.
    """
    if house.get("kind") == "city":
        return 2
    return 1


def _touching_tile_ids_from_house(house: dict) -> list[int]:
    """
    piece_detection stores labels like:
    ["T4C2", "T5C5", ...]
    This extracts the tile numbers.
    """
    tile_ids = set()

    for label in house.get("labels", []):
        match = re.match(r"T(\d+)C\d+", label)
        if match:
            tile_ids.add(int(match.group(1)))

    return sorted(tile_ids)


def calculate_resource_payout(
    dice_sum: int | None,
    number_map: dict[int, int],
    resource_labels: list[str],
    houses: list[dict],
) -> dict[str, dict[str, int]]:
    """
    Returns:
    {
        "red": {"Brick": 1},
        "blue": {"Brick": 1, "Grain": 1},
    }
    """
    payout = defaultdict(lambda: defaultdict(int))

    if dice_sum is None:
        return {}

    for house in houses:
        color = house.get("color")
        if color is None:
            continue

        amount = _resource_amount_for_house(house)

        for tile_id in _touching_tile_ids_from_house(house):
            if tile_id not in number_map:
                continue

            if number_map[tile_id] != dice_sum:
                continue

            resource = resource_labels[tile_id]

            if resource == "Desert":
                continue

            payout[color][resource] += amount

    return {
        color: dict(resources)
        for color, resources in payout.items()
    }


def format_payout_message(payout: dict[str, dict[str, int]]) -> str:
    if not payout:
        return "No resources produced."

    parts = []

    for color in sorted(payout.keys()):
        resource_parts = []

        for resource, amount in sorted(payout[color].items()):
            resource_parts.append(f"{amount} {resource}")

        parts.append(f"{color} gets " + ", ".join(resource_parts))

    return " | ".join(parts)


def get_resource_payout_message(
    dice_sum: int | None,
    number_map: dict[int, int],
    resource_labels: list[str],
    houses: list[dict],
) -> str:
    payout = calculate_resource_payout(
        dice_sum=dice_sum,
        number_map=number_map,
        resource_labels=resource_labels,
        houses=houses,
    )

    return format_payout_message(payout)