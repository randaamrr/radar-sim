import pyvista as pv

from src.scenario import load_scenario
from src.viewer_surfaces import build_surface_scene, to_pv
from src.people import person_meshes, validate_placement

# pose, height, x, y, yaw_deg, support_z
PEOPLE = [
    ("stand", 1.7, 1.3, 2.3, 0, 0.0),      # standing on the floor
    ("sit", 1.7, 1.9, 0.8, 0, 0.5),        # sitting on the bed edge
    ("lie", 1.7, 2.8, 1.0, 90, 0.0),       # lying on the floor
]
COLORS = ["navajowhite", "lightsalmon", "plum"]

if __name__ == "__main__":
    cfg = load_scenario("scenarios/room_basic.yaml")
    plotter = pv.Plotter()
    build_surface_scene(cfg, plotter)
    for color, (pose, h, x, y, yaw, sz) in zip(COLORS, PEOPLE):
        parts = person_meshes(pose, h, x, y, yaw, sz)
        print(pose, "->", validate_placement(cfg, parts) or "valid")
        for _, m in parts:
            plotter.add_mesh(to_pv(m), color=color, smooth_shading=True)
    plotter.show()