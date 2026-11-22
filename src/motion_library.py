"""Step 24: tagged motion library generated from a seed.

Every item is a Sequence in the local frame (see motion.py) with its tags,
parameters and seed, so any item can be regenerated exactly. Parameter
ranges are listed in RANGES with their source or "assumption".
"""
import json
import os

import numpy as np

from src import motion as M

KINDS = ["walk", "sit", "stand", "fall", "lie_still", "breathe"]
TAGS = {
    "walk": ["locomotion", "periodic", "normal"],
    "sit": ["transition", "non-periodic", "normal"],
    "stand": ["transition", "non-periodic", "normal"],
    "fall": ["fall", "event", "non-periodic", "abnormal"],
    "lie_still": ["static", "lying", "abnormal-if-unexpected"],
    "breathe": ["static", "standing", "normal"],
}

# name: (low, high or mean/sd, source)
RANGES = {
    "height_m": ("uniform", 1.55, 1.90, "assumption: adults"),
    "walk.speed": ("normal", 1.31, 0.20, "mean: pooled usual pace of healthy adults "
                   "(Sports Med 2020, 1.31 m/s); sd: assumption; clipped 0.8-1.8"),
    "walk.duration": ("uniform", 2.0, 4.0, "assumption"),
    "sit.duration": ("normal", 2.4, 0.3, "mean/sd close to 2.36 +- 0.29 s measured for a "
                     "2 s sit-to-stand target (PMC4122698); stand-to-sit assumed equal"),
    "sit.seat_z": ("uniform", 0.40, 0.55, "assumption: bed/chair height"),
    "sit.back_m": ("uniform", 0.15, 0.30, "assumption"),
    "sit.lean_peak": ("uniform", 15.0, 35.0, "assumption"),
    "fall.direction_deg": ("choice", [0, 90, 180, 270], 15.0, "assumption: 4 directions, "
                           "+- gaussian jitter (sd 15 deg)"),
    "fall.duration": ("uniform", 0.5, 1.0, "assumption (no verified source yet)"),
    "fall.exponent": ("uniform", 1.5, 2.5, "assumption: accelerating lean"),
    "fall.knee_buckle": ("uniform", 10.0, 50.0, "assumption"),
    "fall.arm_reach": ("uniform", 0.0, 60.0, "assumption"),
    "fall.pre_s": ("uniform", 0.5, 1.5, "assumption"),
    "fall.rest_s": ("const", 2.0, 0.0, "assumption"),
    "still.duration": ("uniform", 3.0, 6.0, "assumption"),
    "breath.hz": ("uniform", 0.20, 0.33, "12-20 breaths/min at rest (arXiv 2208.00837)"),
    "breath.p2p_mm": ("uniform", 3.0, 8.0, "assumption around 4-6 mm used in a simulation paper"),
}


def _u(rng, key):
    _, lo, hi, _ = RANGES[key]
    return float(rng.uniform(lo, hi))


def sample_params(kind, rng):
    """Random parameters for one item (everything needed to rebuild it)."""
    p = {"height_m": _u(rng, "height_m"),
         "breath_hz": _u(rng, "breath.hz"), "breath_p2p_mm": _u(rng, "breath.p2p_mm"),
         "breath_phase": float(rng.uniform(0, 2 * np.pi))}
    if kind == "walk":
        _, mu, sd, _ = RANGES["walk.speed"]
        p["speed"] = float(np.clip(rng.normal(mu, sd), 0.8, 1.8))
        p["duration"] = _u(rng, "walk.duration")
    elif kind in ("sit", "stand"):
        _, mu, sd, _ = RANGES["sit.duration"]
        p["duration"] = float(np.clip(rng.normal(mu, sd), 1.5, 3.5))
        p["seat_z"] = _u(rng, "sit.seat_z")
        p["back_m"] = _u(rng, "sit.back_m")
        p["lean_peak"] = _u(rng, "sit.lean_peak")
    elif kind == "fall":
        base = float(rng.choice(RANGES["fall.direction_deg"][1]))
        p["direction_deg"] = float(base + rng.normal(0, RANGES["fall.direction_deg"][2]))
        for k in ("duration", "exponent", "knee_buckle", "arm_reach", "pre_s"):
            p[k] = _u(rng, f"fall.{k}")
        p["rest_s"] = RANGES["fall.rest_s"][1]
    else:
        p["duration"] = _u(rng, "still.duration")
    return p


def build(kind, params, fps):
    """Sequence (local frame, with breathing) from a kind and its parameters."""
    p = dict(params)
    H = p["height_m"]
    if kind == "walk":
        seq = M.walk(H, fps, p["duration"], speed=p["speed"])
    elif kind == "sit":
        seq = M.sit_down(H, fps, p["duration"], p["seat_z"], p["back_m"], p["lean_peak"])
    elif kind == "stand":
        seq = M.stand_up(H, fps, p["duration"], p["seat_z"], p["back_m"], p["lean_peak"])
    elif kind == "fall":
        seq = M.fall(H, fps, p["direction_deg"], p["duration"], p["exponent"],
                     p["knee_buckle"], p["arm_reach"], p["pre_s"], p["rest_s"])
    elif kind == "lie_still":
        seq = M.still("lie", H, fps, p["duration"])
    elif kind == "breathe":
        seq = M.still("stand", H, fps, p["duration"])
    else:
        raise ValueError(f"unknown motion kind: {kind}")
    seq = M.with_breathing(seq, p["breath_hz"], p["breath_p2p_mm"], p.get("breath_phase", 0.0))
    seq.kind = kind
    seq.params = {**seq.params, **p}
    return seq


def generate_library(seed, per_kind=10, fps=20):
    """[{id, kind, tags, seed, params, sequence}] for every kind, from a seed."""
    items = []
    for ki, kind in enumerate(KINDS):
        for i in range(per_kind):
            rng = np.random.default_rng([int(seed), ki, i])
            params = sample_params(kind, rng)
            items.append({"id": f"{kind}_{i:03d}", "kind": kind, "tags": TAGS[kind],
                          "seed": [int(seed), ki, i], "params": params,
                          "sequence": build(kind, params, fps)})
    return items


def save_library(items, outdir):
    """One .npz per item (joints, breath) plus index.json with tags and params."""
    os.makedirs(outdir, exist_ok=True)
    index = []
    for it in items:
        s = it["sequence"]
        np.savez_compressed(os.path.join(outdir, it["id"] + ".npz"),
                            joints=s.joints, breath=s.breath, fps=s.fps, height=s.height)
        index.append({"id": it["id"], "kind": it["kind"], "tags": it["tags"],
                      "seed": it["seed"], "params": it["params"],
                      "fps": s.fps, "n_frames": s.n_frames,
                      "duration_s": round(s.duration, 3), "file": it["id"] + ".npz"})
    with open(os.path.join(outdir, "index.json"), "w", encoding="utf-8") as f:
        json.dump({"joint_names": M.JOINTS, "items": index}, f, indent=1)
    return index


def load_sequence(path, kind="loaded"):
    d = np.load(path, allow_pickle=False)
    return M.Sequence(kind, float(d["fps"]), float(d["height"]), d["joints"], d["breath"])


# ------------------------------------------------------------------ scenario
def person_sequence(cfg, person, index=0):
    """Placed Sequence for a scenario person that has a `motion:` section.

    motion: {kind: walk|sit|stand|fall|lie_still|breathe, <parameter>: value...}
    Unspecified parameters are drawn from the seed (cfg seed, person index).
    """
    spec = dict(person["motion"])
    kind = spec.pop("kind")
    rng = np.random.default_rng([int(cfg["seed"]), 1000 + index])
    params = sample_params(kind, rng)
    params["height_m"] = person["height_m"]
    unknown = set(spec) - set(params)
    if unknown:
        raise ValueError(f"unknown motion parameters: {sorted(unknown)}")
    params.update(spec)
    seq = build(kind, params, cfg["fps"])
    x, y = person["pos_m"]
    return M.place(seq, x, y, person.get("yaw_deg", 0.0), person.get("support_z", 0.0))


if __name__ == "__main__":
    import sys
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    index = save_library(generate_library(seed, per_kind=10), "output/motion_library")
    print(f"saved {len(index)} sequences to output/motion_library (seed {seed})")
