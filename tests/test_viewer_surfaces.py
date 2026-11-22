import numpy as np
import pyvista as pv
from src.scenario import load_scenario
from src.meshes import furniture_meshes
from src.viewer_surfaces import (to_pv, quad, shell_quads, visibility_overlay,
                                 build_surface_scene)


def test_to_pv_matches_mesh():
    cfg = load_scenario("scenarios/room_basic.yaml")
    _, _, mesh = furniture_meshes(cfg)[0]
    poly = to_pv(mesh)
    assert poly.n_cells == len(mesh.faces)
    assert np.allclose(np.array(poly.bounds).reshape(3, 2).T, mesh.bounds)


def test_shell_quads_cover_the_room_area():
    cfg = load_scenario("scenarios/room_basic.yaml")
    L, W, H = cfg["room"]["size_m"]
    total = sum(q.area for _, q, _, _ in shell_quads(cfg))
    assert np.isclose(total, 2 * (L * W + L * H + W * H))
    assert np.isclose(quad((0, 0, 0), (4, 0, 0), (0, 3, 0)).area, 12.0)


def test_overlay_splits_seen_and_hidden():
    cfg = load_scenario("scenarios/room_basic.yaml")
    pts, seen = visibility_overlay(cfg, "radar_west")
    assert len(pts) == len(seen)
    assert 0.5 < seen.mean() < 0.8       # about 66% in the occlusion study


def test_surface_scene_builds_with_and_without_overlay():
    cfg = load_scenario("scenarios/room_basic.yaml")
    for radar in (None, "radar_east"):
        p = pv.Plotter(off_screen=True)
        build_surface_scene(cfg, p, radar)
        # 6 surfaces + box outline + furniture + radars (+ overlay)
        assert len(p.renderer.actors) >= 6 + 1 + len(cfg["furniture"]) + 2
        p.close()


def test_screenshot_is_written(tmp_path):
    cfg = load_scenario("scenarios/room_basic.yaml")
    out = tmp_path / "scene.png"
    p = pv.Plotter(off_screen=True)
    build_surface_scene(cfg, p, "radar_west")
    p.show(screenshot=str(out))
    assert out.exists() and out.stat().st_size > 1000