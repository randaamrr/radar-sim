import copy
import numpy as np

from src.room import build_room
from src.materials import apply_materials


def range_profile(room, radar_pos, bin_m=0.05, r_max=8.0):
    """Received power vs range for point scatterers (power ~ strength / R^4)."""
    d = np.linalg.norm(room["points"] - np.asarray(radar_pos, dtype=float), axis=1)
    power = room["strength"] / np.maximum(d, 0.1) ** 4
    edges = np.arange(0.0, r_max + bin_m, bin_m)
    hist, _ = np.histogram(d, bins=edges, weights=power)
    return hist


def room_at_step(cfg, step):
    c = copy.deepcopy(cfg)
    c["room"]["grid_step_m"] = step
    return apply_materials(build_room(c))

def smoothed_profile(room, radar_pos, range_res_m, r_max=8.0):
    """Range profile blurred to the radar's range resolution.

    A real radar cannot separate scatterers closer than c/(2B), so the
    fair comparison between two geometry samplings is at that resolution.
    """
    fine = range_profile(room, radar_pos, bin_m=0.01, r_max=r_max)
    sigma_bins = max(range_res_m / 0.01 / 2.355, 0.5)  # FWHM -> sigma
    half = int(4 * sigma_bins)
    k = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma_bins) ** 2)
    k /= k.sum()
    return np.convolve(fine, k, mode="same") 

def grid_convergence(cfg, steps=(0.4, 0.2, 0.1, 0.05), ref_step=0.025,
                     range_res_m=0.15):
    """Relative error of each grid step vs a fine reference, measured at
    the radar's range resolution (default 15 cm = 1 GHz bandwidth)."""
    ref = room_at_step(cfg, ref_step)
    ref_profiles = [smoothed_profile(ref, r["pos_m"], range_res_m)
                    for r in cfg["radars"]]

    results = []
    for step in steps:
        room = room_at_step(cfg, step)
        errs = []
        for r, q in zip(cfg["radars"], ref_profiles):
            p = smoothed_profile(room, r["pos_m"], range_res_m)
            errs.append(np.linalg.norm(p - q) / np.linalg.norm(q))
        results.append({
            "step_m": step,
            "n_points": int(len(room["points"])),
            "rel_error": float(np.mean(errs)),
        })
    return results


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path
    from src.scenario import load_scenario

    cfg = load_scenario("scenarios/room_basic.yaml")
    print(f"{'step (m)':>9} {'points':>9} {'err @15cm':>10} {'err @5cm':>9}")
    r15 = grid_convergence(cfg, range_res_m=0.15)
    r5 = grid_convergence(cfg, range_res_m=0.05)
    for a, b in zip(r15, r5):
        print(f"{a['step_m']:>9} {a['n_points']:>9} {a['rel_error']:>10.3f} {b['rel_error']:>9.3f}")
    res = r15

    Path("output").mkdir(exist_ok=True)
    fig, ax = plt.subplots()
    ax.loglog([r["step_m"] for r in res], [r["rel_error"] for r in res], "o-")
    ax.set_xlabel("grid step (m)")
    ax.set_ylabel("relative range-profile error")
    ax.set_title("Grid step convergence")
    ax.grid(True, which="both", alpha=0.3)
    fig.savefig("output/grid_convergence.png", dpi=150)
    print("saved output/grid_convergence.png")