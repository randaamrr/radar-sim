import pyvista as pv
from src.scenario import load_scenario
from src.viewer import build_scene


def test_scene_builds_offscreen():
    cfg = load_scenario("scenarios/room_basic.yaml")
    p = pv.Plotter(off_screen=True)
    build_scene(cfg, p)
    # 3 materials points (concrete, wood) + 2 arrows + box + labels
    assert len(p.renderer.actors) >= 5
    p.close()