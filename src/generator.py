from pathlib import Path

import numpy as np
import yaml

# kind -> typical size (length, width, height) in meters, allowed materials
CATALOG = {
    "bed":      {"size_m": (2.0, 1.4, 0.5),  "materials": ["wood", "metal"]},
    "wardrobe": {"size_m": (0.5, 1.2, 2.0),  "materials": ["wood", "metal"]},
    "table":    {"size_m": (1.0, 0.6, 0.75), "materials": ["wood", "glass"]},
    "sofa":     {"size_m": (1.9, 0.9, 0.8),  "materials": ["wood"]},
    "desk":     {"size_m": (1.2, 0.6, 0.75), "materials": ["wood", "metal"]},
    "shelf":    {"size_m": (0.4, 1.0, 1.8),  "materials": ["wood", "metal"]},
    "chair":    {"size_m": (0.45, 0.45, 0.9), "materials": ["wood", "metal"]},
}

WALL_MATERIALS = ["concrete", "plasterboard"]
FLOOR_MATERIALS = ["concrete", "wood"]
CEILING_MATERIALS = ["concrete", "plasterboard"]

WALL_MARGIN_M = 0.05      # furniture stays this far from the walls
GAP_M = 0.10              # minimum gap between two pieces
MAX_FOOTPRINT = 0.40      # furniture may cover at most 40% of the floor


def _overlap(a, b):
    """Axis-aligned overlap test on the floor plane, with a gap."""
    ax, ay, aw, ad = a
    bx, by, bw, bd = b
    return not (ax + aw + GAP_M <= bx or bx + bw + GAP_M <= ax or
                ay + ad + GAP_M <= by or by + bd + GAP_M <= ay)


def generate_scenario(seed, grid_step_m=0.1):
    """Build a random but fully reproducible room config from a seed."""
    rng = np.random.default_rng(seed)

    L = round(float(rng.uniform(3.0, 6.0)), 1)
    W = round(float(rng.uniform(2.5, 5.0)), 1)
    H = round(float(rng.uniform(2.5, 3.0)), 1)

    cfg = {
        "seed": int(seed),
        "room": {"size_m": [L, W, H], "grid_step_m": grid_step_m},
        "surfaces": {
            "walls": str(rng.choice(WALL_MATERIALS)),
            "floor": str(rng.choice(FLOOR_MATERIALS)),
            "ceiling": str(rng.choice(CEILING_MATERIALS)),
        },
        "furniture": [],
        "radars": [],
    }

    kinds = list(CATALOG)
    n_items = int(rng.integers(2, 6))
    placed = []          # (x, y, w, d) of accepted pieces
    footprint = 0.0

    for i in range(n_items):
        kind = str(rng.choice(kinds))
        spec = CATALOG[kind]
        scale = rng.uniform(0.85, 1.15, size=3)
        w, d, h = (round(float(s * k), 2) for s, k in zip(spec["size_m"], scale))
        if rng.random() < 0.5:           # rotate 90 degrees on the floor
            w, d = d, w
        material = str(rng.choice(spec["materials"]))

        if w > L - 2 * WALL_MARGIN_M or d > W - 2 * WALL_MARGIN_M or h > H:
            continue
        if footprint + w * d > MAX_FOOTPRINT * L * W:
            continue

        for _ in range(200):             # rejection sampling
            x = round(float(rng.uniform(WALL_MARGIN_M, L - w - WALL_MARGIN_M)), 2)
            y = round(float(rng.uniform(WALL_MARGIN_M, W - d - WALL_MARGIN_M)), 2)
            if not any(_overlap((x, y, w, d), p) for p in placed):
                placed.append((x, y, w, d))
                footprint += w * d
                cfg["furniture"].append({
                    "name": f"{kind}_{i}",
                    "pos_m": [x, y],
                    "size_m": [w, d, h],
                    "material": material,
                })
                break

    z = round(min(2.2, H - 0.3), 2)
    cfg["radars"] = [
        {"name": "radar_west", "pos_m": [0.0, round(W / 2, 2), z],
         "yaw_deg": 0, "pitch_deg": -20},
        {"name": "radar_east", "pos_m": [L, round(W / 2, 2), z],
         "yaw_deg": 180, "pitch_deg": -20},
    ]
    return cfg


def save_scenario(cfg, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    for s in range(n):
        out = f"scenarios/generated/room_{s:03d}.yaml"
        save_scenario(generate_scenario(s), out)
    print(f"wrote {n} scenarios to scenarios/generated/")