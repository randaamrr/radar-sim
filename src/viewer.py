import numpy as np
import pyvista as pv

from src.scenario import load_scenario, radar_direction
from src.room import build_room
from src.materials import apply_materials

COLORS = {
    "concrete": "dimgray",
    "wood": "saddlebrown",
    "glass": "lightblue",
    "plasterboard": "wheat",
    "metal": "silver",
}


def build_scene(cfg, plotter):
    """Add room points (colored by material) and radar arrows to a plotter."""
    room = apply_materials(build_room(cfg))
    for mat in np.unique(room["material"]):
        mask = room["material"] == mat
        cloud = pv.PolyData(room["points"][mask])
        plotter.add_mesh(
            cloud,
            color=COLORS.get(mat, "white"),
            point_size=6,
            render_points_as_spheres=False,
            label=mat,
        )

    for r in cfg["radars"]:
        pos = np.array(r["pos_m"], dtype=float)
        d = radar_direction(r["yaw_deg"], r["pitch_deg"])
        arrow = pv.Arrow(start=pos, direction=d, scale=0.8)
        plotter.add_mesh(arrow, color="red")
        plotter.add_point_labels([pos + np.array([0, 0, 0.15])], [r["name"]],
                                 font_size=12, point_size=1, shape=None,
                                 always_visible=True)

    L, W, H = cfg["room"]["size_m"]
    plotter.add_mesh(pv.Box(bounds=(0, L, 0, W, 0, H)),
                     style="wireframe", color="black", line_width=1)
    plotter.add_legend()
    plotter.add_axes()


def show(path="scenarios/room_basic.yaml"):
    cfg = load_scenario(path)
    plotter = pv.Plotter()
    build_scene(cfg, plotter)
    plotter.show()


if __name__ == "__main__":
    show()
