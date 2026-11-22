"""Step 23/24 visual check: frames of the scenario motions + curves.

python -m src.motion_demo [scenario.yaml]   (writes output/motion_frames.png
and output/motion_curves.png)
"""
import sys

import numpy as np
import pyvista as pv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import motion as M
from src.scenario import load_scenario
from src.motion_library import person_sequence, generate_library
from src.viewer_surfaces import build_surface_scene, to_pv


def frames_png(cfg, path, every_s=0.4):
    pl = pv.Plotter(off_screen=True, window_size=(1400, 900))
    cfg_static = dict(cfg, people=[])
    build_surface_scene(cfg_static, pl)
    for pi, person in enumerate(cfg["people"]):
        seq = person_sequence(cfg, person, pi)
        ks = list(range(0, seq.n_frames, max(1, int(every_s * seq.fps)))) + [seq.n_frames - 1]
        cmap = plt.get_cmap("viridis" if pi == 0 else "plasma")
        for n, k in enumerate(ks):
            color = cmap(n / max(1, len(ks) - 1))[:3]
            for _, m in seq.parts(k):
                pl.add_mesh(to_pv(m), color=color, opacity=0.85, smooth_shading=True)
    pl.camera_position = [(7.5, -4.5, 5.0), (2.0, 1.5, 0.8), (0, 0, 1)]
    pl.show(screenshot=path)


def curves_png(path):
    lib = generate_library(11, per_kind=5, fps=20)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for it in lib:
        s, k = it["sequence"], it["kind"]
        t = np.arange(s.n_frames) / s.fps
        if k in ("walk", "fall", "sit", "stand"):
            ax[0].plot(t, s.joints[:, M.IDX["head"], 2], color={"walk": "tab:blue", "fall": "tab:red",
                       "sit": "tab:green", "stand": "tab:olive"}[k], alpha=0.6,
                       label=k if it["id"].endswith("000") else None)
        if k == "walk":
            ax[1].plot(t, s.joints[:, M.IDX["ankle_l"], 0] - s.joints[:, M.IDX["ankle_r"], 0],
                       alpha=0.6, color="tab:blue")
        if k == "breathe":
            ax[2].plot(t, s.breath * 1e3, alpha=0.7)
    ax[0].set(title="head height (m)", xlabel="s"); ax[0].legend()
    ax[1].set(title="walk: left minus right ankle along x (m)", xlabel="s")
    ax[2].set(title="breathing: torso radius change (mm)", xlabel="s")
    fig.tight_layout(); fig.savefig(path, dpi=110)


if __name__ == "__main__":
    cfg = load_scenario(sys.argv[1] if len(sys.argv) > 1 else "scenarios/room_motion.yaml")
    frames_png(cfg, "output/motion_frames.png")
    curves_png("output/motion_curves.png")
    print("saved output/motion_frames.png and output/motion_curves.png")
