import numpy as np

from src.scenario import load_scenario
from src.surface import scene_samples, radar_view, direct_power
from src.multipath import first_order_ghosts, _smooth
from src.ghosts import first_order_ghosts_view


def profiles(strength_room, ranges, powers, pos, res=0.15, bin_m=0.01, r_max=10.0):
    edges = np.arange(0.0, r_max + bin_m, bin_m)
    d = np.maximum(np.linalg.norm(strength_room["points"] - np.asarray(pos), axis=1), 0.1)
    direct, _ = np.histogram(d, bins=edges, weights=strength_room["strength"] / d ** 4)
    ghost, _ = np.histogram(ranges, bins=edges, weights=powers)
    x = 0.5 * (edges[:-1] + edges[1:])
    return x, _smooth(direct, res, bin_m), _smooth(ghost, res, bin_m)


def dominance_pct(x, direct, ghost):
    """% of range bins (0.5-6 m) where the ghost is stronger than the direct."""
    m = (x >= 0.5) & (x <= 6.0) & (np.maximum(direct, ghost) > 1e-6 * direct.max())
    return 100.0 * float((ghost[m] > direct[m]).mean())


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

        # A: old model (no incidence angle, no occlusion)
        ra, pa = first_order_ghosts(room, cfg, pos)
        da = direct_power(room, pos)
        xa, dir_a, gh_a = profiles(room, ra, pa, pos)

        # B: incidence angle only
        rb, pb = first_order_ghosts_view(room, cfg, pos, occlusion=False)
        vb = radar_view(room, cfg, pos, occlusion=False)

        # C: incidence angle + occlusion
        rc, pc = first_order_ghosts_view(room, cfg, pos)
        vc = radar_view(room, cfg, pos)
        xc, dir_c, gh_c = profiles(vc, rc, pc, pos)

        db = lambda g, d: 10 * np.log10(g / d)
        print(f"{r['name']}: ghost/direct  old {db(pa.sum(), da):.1f} dB | "
              f"angle {db(pb.sum(), direct_power(vb, pos)):.1f} dB | "
              f"angle+occlusion {db(pc.sum(), direct_power(vc, pos)):.1f} dB")
        print(f"    ghost stronger than direct in: old {dominance_pct(xa, dir_a, gh_a):.0f}% "
              f"| angle+occlusion {dominance_pct(xc, dir_c, gh_c):.0f}% of range bins")

        peak = dir_c.max()
        ax.semilogy(xc, np.maximum(dir_c / peak, 1e-8), label="direct")
        ax.semilogy(xc, np.maximum(gh_c / peak, 1e-8), label="ghost (1st order)")
        ax.set_ylim(1e-6, 2)
        ax.set_xlim(0, 8)
        ax.set_title(r["name"] + " (angle + occlusion)")
        ax.set_ylabel("power (normalized)")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
    axes[-1].set_xlabel("apparent range (m)")
    fig.tight_layout()
    fig.savefig("output/ghosts_occ.png", dpi=150)
    print("saved output/ghosts_occ.png")