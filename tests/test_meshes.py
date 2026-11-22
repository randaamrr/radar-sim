import numpy as np
from src.scenario import load_scenario
from src.meshes import (room_shell_mesh, furniture_meshes,
                        occluder_scene, visible)

CFG = {
    "room": {"size_m": [4.0, 3.0, 2.7], "grid_step_m": 0.1},
    "furniture": [{"name": "box", "pos_m": [1.5, 1.0],
                   "size_m": [1.0, 1.0, 1.0], "material": "wood"}],
}


def test_room_shell_is_closed_with_right_bounds():
    cfg = load_scenario("scenarios/room_basic.yaml")
    shell = room_shell_mesh(cfg)
    assert shell.is_watertight
    assert np.allclose(shell.bounds, [[0, 0, 0], [4.0, 3.0, 2.7]])


def test_furniture_meshes_match_config():
    cfg = load_scenario("scenarios/room_basic.yaml")
    items = furniture_meshes(cfg)
    assert len(items) == len(cfg["furniture"])
    for (name, mat, m), it in zip(items, cfg["furniture"]):
        assert m.is_watertight
        assert np.isclose(m.volume, np.prod(it["size_m"]))


def test_object_blocks_what_is_behind_it():
    scene = occluder_scene(CFG)
    radar = [0.0, 1.5, 0.5]
    behind = [3.5, 1.5, 0.5]       # straight line goes through the box
    aside = [3.5, 0.2, 0.5]        # line passes beside the box
    front_face = [1.5, 1.5, 0.5]   # the face the radar looks at
    v = visible(scene, radar, [behind, aside, front_face])
    assert list(v) == [False, True, True]


def test_no_furniture_everything_visible():
    assert visible(None, [0, 0, 1], [[1, 1, 1], [2, 2, 2]]).all()