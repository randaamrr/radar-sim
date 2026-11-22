import numpy as np
from src.scenario import load_scenario
from src.surface import (scene_samples, radar_view, direct_power,
                         surface_normals, removed_by_range)

CFG = {
    "seed": 1,
    "room": {"size_m": [4.0, 3.0, 2.7], "grid_step_m": 0.1},
    "surfaces": {"walls": "concrete", "floor": "concrete", "ceiling": "concrete"},
    "furniture": [{"name": "box", "pos_m": [1.5, 1.0],
                   "size_m": [1.0, 1.0, 1.0], "material": "wood"}],
    "radars": [],
}
RADAR = [0.0, 1.5, 0.5]


def test_normals_are_unit_and_correct():
    room = scene_samples(load_scenario("scenarios/room_basic.yaml"))
    assert np.allclose(np.linalg.norm(room["normal"], axis=1), 1.0)
    floor = room["surface"] == "floor"
    assert np.allclose(room["normal"][floor], [0, 0, 1])
    assert np.allclose(surface_normals(np.array(["table_3_top"])), [[0, 0, 1]])


def test_box_casts_a_shadow_on_the_far_wall():
    room = scene_samples(CFG)
    view = radar_view(room, CFG, RADAR)
    p = room["points"]
    far = room["surface"] == "wall_x1"
    shadow = far & (p[:, 1] > 0.5) & (p[:, 1] < 2.5) & (p[:, 2] < 1.5)
    lit = far & (p[:, 1] < 0.06)
    assert shadow.sum() > 0 and (view["strength"][shadow] == 0).all()
    assert lit.sum() > 0 and (view["strength"][lit] > 0).all()


def test_faces_toward_and_away_from_radar():
    room = scene_samples(CFG)
    view = radar_view(room, CFG, RADAR)
    assert (view["strength"][room["surface"] == "box_left"] > 0).all()
    assert (view["strength"][room["surface"] == "box_right"] == 0).all()


def test_power_decreases_with_each_effect():
    cfg = load_scenario("scenarios/room_basic.yaml")
    room = scene_samples(cfg)
    pos = cfg["radars"][0]["pos_m"]
    a = direct_power(room, pos)
    b = direct_power(radar_view(room, cfg, pos, occlusion=False), pos)
    c = direct_power(radar_view(room, cfg, pos), pos)
    assert c < b <= a


def test_occlusion_matters_far_not_near():
    room = scene_samples(CFG)
    res = removed_by_range(room, CFG, RADAR, edges=(0, 1, 2, 6))
    assert res[0][2] == 0.0      # 0-1 m: nothing is behind the box yet
    assert res[2][2] > 0.0       # 2-6 m: the shadow of the box