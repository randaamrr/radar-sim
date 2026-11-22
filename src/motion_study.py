"""Step 24 study: which body model is the smallest that separates walking
from falling, and which body parts carry the motion?

Proxy, NOT a radar signature: radial surface velocity seen from a radar
position, weighted by area and by facing the radar (no occlusion, no range
or Doppler resolution, no noise). Real numbers come after steps 25-33.
"""
import json
import sys

import numpy as np

from src import motion as M
from src.motion_library import generate_library

RADAR = np.array([0.0, 1.5, 2.2])          # radar_west of room_basic.yaml
WINDOW_S = 0.5                              # sliding window for the Doppler spread
MODELS = {                                  # body model -> capsule names kept
    "M1 torso only": ["torso"],
    "M2 torso+head": ["torso", "head"],
    "M3 +legs": ["torso", "head", "thigh_l", "thigh_r", "shin_l", "shin_r"],
    "M4 full (10 capsules)": [c[0] for c in __import__("src.people", fromlist=["x"]).CAPSULES],
}
PART_NAMES = MODELS["M4 full (10 capsules)"]


def radial_tracks(seq):
    """Per frame step: radial velocity (m/s, + = receding) and weight, per part."""
    V = M.vertex_tracks(seq)                                     # (F, P, nV, 3)
    mid = 0.5 * (V[1:] + V[:-1])
    vel = (V[1:] - V[:-1]) * seq.fps
    rvec = mid - RADAR
    rhat = rvec / np.linalg.norm(rvec, axis=-1, keepdims=True)
    vr = (vel * rhat).sum(-1)                                    # (F-1, P, nV)
    # vertex normals of the first frame's meshes give the facing weight; the
    # topology is fixed, but normals rotate, so recompute per frame.
    w = np.zeros_like(vr)
    areas = np.array([m.area for _, m in seq.parts(0)])
    for k in range(len(vr)):
        for pi, (_, m) in enumerate(seq.parts(k + 1)):
            cos = (m.vertex_normals * (-rhat[k, pi])).sum(-1)
            w[k, pi] = areas[pi] / m.vertices.shape[0] * np.clip(cos, 0, None)
    return vr, w


def features(vr, w, keep, fps):
    """Doppler spread peak (m/s) and bulk speed peak (m/s) for the kept parts."""
    v, ww = vr[:, keep], w[:, keep]
    tot = ww.sum(axis=(1, 2)) + 1e-12
    mean = (v * ww).sum(axis=(1, 2)) / tot
    var = (((v - mean[:, None, None]) ** 2) * ww).sum(axis=(1, 2)) / tot
    n = max(1, int(WINDOW_S * fps))
    std = np.sqrt(var)
    win = np.convolve(std, np.ones(n) / n, mode="valid")
    return float(win.max()), float(np.abs(mean).max())


def dprime(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(abs(a.mean() - b.mean()) / np.sqrt(0.5 * (a.var(ddof=1) + b.var(ddof=1)) + 1e-12))


def run(n=20, seed=11, fps=20):
    lib = generate_library(seed, per_kind=n, fps=fps)
    rng = np.random.default_rng(seed)
    cases = {}
    for kind in ("walk", "fall", "breathe"):
        rows = []
        for it in [i for i in lib if i["kind"] == kind]:
            seq = M.place(it["sequence"], 2.0, 1.5, float(rng.uniform(-180, 180)))
            vr, w = radial_tracks(seq)
            rows.append((it, seq, vr, w))
        cases[kind] = rows
    names = PART_NAMES
    result = {"n_per_kind": n, "seed": seed, "fps": fps, "models": {}, "parts": {}}
    for mname, kept in MODELS.items():
        keep = [names.index(c) for c in kept]
        f = {k: np.array([features(vr, w, keep, fps) for _, _, vr, w in rows])
             for k, rows in cases.items() if k != "breathe"}
        result["models"][mname] = {
            "walk_spread_mean": float(f["walk"][:, 0].mean()),
            "fall_spread_mean": float(f["fall"][:, 0].mean()),
            "d_spread": dprime(f["walk"][:, 0], f["fall"][:, 0]),
            "walk_bulk_mean": float(f["walk"][:, 1].mean()),
            "fall_bulk_mean": float(f["fall"][:, 1].mean()),
            "d_bulk": dprime(f["walk"][:, 1], f["fall"][:, 1]),
        }
    for kind in ("walk", "fall"):
        share = np.zeros(len(names))
        for _, _, vr, w in cases[kind]:
            tot = w.sum(axis=(1, 2)) + 1e-12
            mean = (vr * w).sum(axis=(1, 2)) / tot
            contrib = (((vr - mean[:, None, None]) ** 2) * w).sum(axis=2)   # (F-1, P)
            share += contrib.sum(axis=0) / contrib.sum()
        result["parts"][kind] = {nm: float(x / len(cases[kind])) for nm, x in zip(names, share)}
    # Breathing: torso surface speed against walking
    b = []
    for it, seq, vr, w in cases["breathe"]:
        b.append(np.abs(vr[:, names.index("torso")]).max())
    p = [it["params"] for it, *_ in cases["breathe"]]
    analytic = [np.pi * q["breath_hz"] * q["breath_p2p_mm"] * 1e-3 for q in p]
    result["breathing"] = {"torso_peak_radial_speed_mm_s": float(np.mean(b) * 1e3),
                           "analytic_pi_f_p2p_mm_s": float(np.mean(analytic) * 1e3),
                           "walk_peak_bulk_speed_mm_s": result["models"]["M4 full (10 capsules)"]["walk_bulk_mean"] * 1e3}
    # time series for the figure
    result["_series"] = {k: [(seq.joints[:, M.IDX["head"], 2].tolist(),) for _, seq, _, _ in rows[:5]]
                         for k, rows in cases.items() if k in ("walk", "fall")}
    return result


def report(r):
    print(f"proxy study: {r['n_per_kind']} walks vs {r['n_per_kind']} falls, seed {r['seed']}, "
          f"radar at {RADAR.tolist()} (NOT a radar signature)")
    print(f"{'model':24} {'walk spread':>12} {'fall spread':>12} {'d_spread':>9} {'walk bulk':>10} {'fall bulk':>10} {'d_bulk':>7}")
    for m, v in r["models"].items():
        print(f"{m:24} {v['walk_spread_mean']:12.2f} {v['fall_spread_mean']:12.2f} {v['d_spread']:9.2f} "
              f"{v['walk_bulk_mean']:10.2f} {v['fall_bulk_mean']:10.2f} {v['d_bulk']:7.2f}")
    for kind in ("walk", "fall"):
        top = sorted(r["parts"][kind].items(), key=lambda kv: -kv[1])
        print(f"variance share by part, {kind}: " + ", ".join(f"{n} {100*s:.0f}%" for n, s in top))
    b = r["breathing"]
    print(f"breathing torso peak radial speed {b['torso_peak_radial_speed_mm_s']:.1f} mm/s "
          f"(pi*f*p2p = {b['analytic_pi_f_p2p_mm_s']:.1f} mm/s); walking bulk peak {b['walk_peak_bulk_speed_mm_s']:.0f} mm/s")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    res = run(n)
    report(res)
    res.pop("_series")
    with open("output/motion_study.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
