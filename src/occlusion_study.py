import numpy as np

from src.scenario import load_scenario
from src.surface import scene_samples, radar_view, direct_power, removed_by_range
from src.compare import smoothed_profile

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    cfg = load_scenario("scenarios/room_basic.yaml")
    room = scene_samples(cfg)
    Path("output").mkdir(exist_ok=True)

    n = len(cfg["radars"])
    fig, axes = plt.subplots(n, 1, sharex=True, figsize=(7, 3.5 * n))
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, cfg["radars"]):
        pos = r["pos_m"]
        a = direct_power(room, pos)
        facing = radar_view(room, cfg, pos, facing=True, occlusion=False)
        both = radar_view(room, cfg, pos, facing=True, occlusion=True)
        seen = (both["strength"] > 0).mean() * 100
        print(f"{r['name']}: facing only {10*np.log10(direct_power(facing, pos)/a):.1f} dB, "
              f"facing+occlusion {10*np.log10(direct_power(both, pos)/a):.1f} dB, "
              f"samples contributing {seen:.0f}%")
        for lo, hi, pct in removed_by_range(room, cfg, pos):
            print(f"    {lo}-{hi} m: occlusion removes {pct:.1f}% of power")
        pa = smoothed_profile(room, pos, 0.15)
        pc = smoothed_profile(both, pos, 0.15)
        x = np.arange(len(pa)) * 0.01
        peak = pa.max()
        ax.semilogy(x, np.maximum(pa / peak, 1e-8), label="no occlusion")
        ax.semilogy(x, np.maximum(pc / peak, 1e-8), label="facing + occlusion")
        ax.set_ylim(1e-6, 2)
        ax.set_xlim(0, 8)
        ax.set_title(r["name"])
        ax.set_ylabel("power (normalized)")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
    axes[-1].set_xlabel("range (m)")
    fig.tight_layout()
    fig.savefig("output/occlusion.png", dpi=150)
    print("saved output/occlusion.png")