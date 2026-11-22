import numpy as np
from src.scenario import load_scenario
from src.surface import scene_samples
from src.multipath import first_order_ghosts
from src.ghosts import first_order_ghosts_view, segments_clear
from src.meshes import occluder_scene
from tests.test_surface import CFG, RADAR


def test_reduces_to_old_model_when_disabled():
    cfg = load_scenario("scenarios/room_basic.yaml")
    room = scene_samples(cfg)
    pos = cfg["radars"][0]["pos_m"]
    r0, p0 = first_order_ghosts(room, cfg, pos)
    r1, p1 = first_order_ghosts_view(room, cfg, pos, facing=False, occlusion=False)
    assert len(r0) == len(r1)
    assert np.isclose(p0.sum(), p1.sum())
    assert np.allclose(np.sort(r0), np.sort(r1))


def test_each_effect_removes_ghost_power():
    cfg = load_scenario("scenarios/room_basic.yaml")
    room = scene_samples(cfg)
    pos = cfg["radars"][0]["pos_m"]
    _, p_old = first_order_ghosts_view(room, cfg, pos, facing=False, occlusion=False)
    _, p_ang = first_order_ghosts_view(room, cfg, pos, occlusion=False)
    _, p_all = first_order_ghosts_view(room, cfg, pos)
    assert p_all.sum() < p_ang.sum() < p_old.sum()


def test_hidden_target_has_no_ghost():
    # a point on the far wall in the box's shadow: direct path is blocked
    room = {"points": np.array([[3.95, 1.5, 0.5]]),
            "strength": np.array([1.0]),
            "normal": np.array([[-1.0, 0.0, 0.0]])}
    r_on, _ = first_order_ghosts_view(room, CFG, RADAR, facing=False, occlusion=True)
    r_off, _ = first_order_ghosts_view(room, CFG, RADAR, facing=False, occlusion=False)
    assert len(r_on) == 0 and len(r_off) > 0


def test_segments_clear():
    scene = occluder_scene(CFG)
    a = np.array([RADAR, RADAR])
    b = np.array([[3.5, 1.5, 0.5], [3.5, 0.2, 0.5]])
    assert list(segments_clear(scene, a, b)) == [False, True]