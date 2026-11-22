import sys

import numpy as np
import pyvista as pv

from src.scenario import load_scenario, radar_direction
from src.meshes import furniture_meshes
from src.people import cfg_people
from src.surface import scene_samples, radar_view

MATERIAL_COLORS = {
    "concrete": "silver",
    "wood": "saddlebrown",
    "glass": "lightskyblue",
    "plasterboard": "wheat",
    "metal": "slategray",
}


def to_pv(mesh):
    """trimesh triangle mesh -> pyvista PolyData."""
    faces = np.hstack([np.full((len(mesh.faces), 1), 3), mesh.faces]).ravel()
    return pv.PolyData(np.asarray(mesh.vertices, dtype=float), faces)


def quad(origin, u, v):
    """Flat rectangle from a corner and two edge vectors."""
    o = np.asarray(origin, dtype=float)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    pts = np.array([o, o + u, o + u + v, o + v])
    return pv.PolyData(pts, np.array([4, 0, 1, 2, 3]))


def shell_quads(cfg):
    """Room surfaces as (name, quad, material, opacity)."""
    L, W, H = cfg["room"]["size_m"]
    s = cfg["surfaces"]
    return [
        ("floor", quad((0, 0, 0), (L, 0, 0), (0, W, 0)), s["floor"], 0.9),
        ("ceiling", quad((0, 0, H), (L, 0, 0), (0, W, 0)), s["ceiling"], 0.05),
        ("wall_x0", quad((0, 0, 0), (0, W, 0), (0, 0, H)), s["walls"], 0.12),
        ("wall_x1", quad((L, 0, 0), (0, W, 0), (0, 0, H)), s["walls"], 0.12),
        ("wall_y0", quad((0, 0, 0), (L, 0, 0), (0, 0, H)), s["walls"], 0.12),
        ("wall_y1", quad((0, W, 0), (L, 0, 0), (0, 0, H)), s["walls"], 0.12),
    ]


def visibility_overlay(cfg, radar_name):
    """Sample points (nudged off the surface) and a mask of those that
    contribute to this radar (facing it and not occluded)."""
    radar = next(r for r in cfg["radars"] if r["name"] == radar_name)
    room = scene_samples(cfg)
    view = radar_view(room, cfg, radar["pos_m"])
    pts = room["points"] + 0.01 * room["normal"]
    return pts, view["strength"] > 0


def build_surface_scene(cfg, plotter, radar_name=None):
    for _, q, mat, opacity in shell_quads(cfg):
        plotter.add_mesh(q, color=MATERIAL_COLORS.get(mat, "white"),
                         opacity=opacity)

    L, W, H = cfg["room"]["size_m"]
    plotter.add_mesh(pv.Box(bounds=(0, L, 0, W, 0, H)), style="wireframe",
                     color="black", line_width=1)

    for name, mat, mesh in furniture_meshes(cfg):
        plotter.add_mesh(to_pv(mesh), color=MATERIAL_COLORS.get(mat, "white"),
                         opacity=0.95, show_edges=True, edge_color="black")
        top = np.array([mesh.bounds[:, 0].mean(), mesh.bounds[:, 1].mean(),
                        mesh.bounds[1, 2] + 0.1])
        plotter.add_point_labels([top], [name], font_size=10, point_size=1,
                                 shape=None, always_visible=True)

    for name, parts in cfg_people(cfg):
        for _, mesh in parts:
            plotter.add_mesh(to_pv(mesh), color="navajowhite",
                             smooth_shading=True)
        top = max(m.bounds[1, 2] for _, m in parts) + 0.1
        cx = np.mean([m.bounds[:, :2].mean(axis=0) for _, m in parts], axis=0)
        plotter.add_point_labels([[cx[0], cx[1], top]], [name], font_size=10,
                                 point_size=1, shape=None, always_visible=True)

    for r in cfg["radars"]:
        pos = np.array(r["pos_m"], dtype=float)
        d = radar_direction(r["yaw_deg"], r["pitch_deg"])
        plotter.add_mesh(pv.Arrow(start=pos, direction=d, scale=0.8), color="red")
        plotter.add_point_labels([pos + np.array([0, 0, 0.2])], [r["name"]],
                                 font_size=12, point_size=1, shape=None,
                                 always_visible=True)

    if radar_name:
        pts, seen = visibility_overlay(cfg, radar_name)
        if seen.any():
            plotter.add_mesh(pv.PolyData(pts[seen]), color="limegreen",
                             point_size=4, label=f"seen by {radar_name}")
        if (~seen).any():
            plotter.add_mesh(pv.PolyData(pts[~seen]), color="crimson",
                             point_size=4, label=f"hidden / facing away")
        plotter.add_legend()
    plotter.add_axes()


def show(path="scenarios/room_basic.yaml", radar_name=None, png=None):
    cfg = load_scenario(path)
    plotter = pv.Plotter(off_screen=bool(png))
    build_surface_scene(cfg, plotter, radar_name)
    if png:
        plotter.show(screenshot=png)
        print(f"saved {png}")
    else:
        plotter.show()


if __name__ == "__main__":
    args = sys.argv[1:]
    show(args[0] if len(args) > 0 else "scenarios/room_basic.yaml",
         args[1] if len(args) > 1 else None,
         args[2] if len(args) > 2 else None)