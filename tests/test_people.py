import copy
import numpy as np
from src.scenario import load_scenario
from src.people import person_meshes, merged, validate_placement
from src.meshes import visible
from tests.test_surface import CFG


def _cfg():
    return load_scenario("scenarios/room_basic.yaml")


def test_standing_height_and_parts():
    parts = person_meshes("stand", 1.7, 2.0, 1.5)
    assert len(parts) == 10 and all(m.is_watertight for _, m in parts)
    b = merged(parts).bounds
    assert np.isclose(b[0, 2], 0.0, atol=1e-9)
    assert np.isclose(b[1, 2] - b[0, 2], 1.7, rtol=0.02)


def test_yaw_rotates_the_body():
    ex = lambda yaw: np.ptp(merged(person_meshes("sit", 1.7, 2, 2, yaw, 0.5)).bounds, axis=0)
    e0, e90 = ex(0), ex(90)
    assert e0[0] > e0[1]          # facing +x: longer along x (legs forward)
    assert e90[1] > e90[0]        # facing +y: longer along y


def test_valid_placements():
    cfg = _cfg()
    cases = [
        ("stand", 1.3, 2.3, 0, 0.0),     # on the floor
        ("stand", 1.0, 0.8, 0, 0.5),     # standing on the bed
        ("sit", 1.9, 0.8, 0, 0.5),       # on the bed edge, legs over the side
        ("lie", 2.8, 1.0, 90, 0.0),      # lying on the floor
    ]
    for pose, x, y, yaw, sz in cases:
        problems = validate_placement(cfg, person_meshes(pose, 1.7, x, y, yaw, sz))
        assert problems == [], (pose, problems)


def test_invalid_placements():
    cfg = _cfg()
    # standing on the wardrobe: head would be above the ceiling
    p = validate_placement(cfg, person_meshes("stand", 1.7, 3.75, 0.7, 0, 2.0))
    assert any("outside" in s for s in p)
    # sitting in the middle of the bed: the shins go through the mattress block
    p = validate_placement(cfg, person_meshes("sit", 1.7, 1.0, 0.8, 0, 0.5))
    assert any("bed" in s for s in p)
    # sitting on the floor: no room for the lower legs under a seat pose
    p = validate_placement(cfg, person_meshes("sit", 1.7, 1.3, 2.3, 0, 0.0))
    assert any("outside" in s for s in p)
    # too close to the wall
    p = validate_placement(cfg, person_meshes("stand", 1.7, 0.05, 2.5, 0, 0.0))
    assert any("outside" in s for s in p)


def test_person_blocks_line_of_sight():
    cfg = copy.deepcopy(CFG)
    cfg["furniture"] = []
    scene = merged(person_meshes("stand", 1.7, 2.0, 1.5))
    radar = [0.0, 1.5, 1.0]
    v = visible(scene, radar, [[3.5, 1.5, 1.0], [3.5, 0.2, 1.0]])
    assert list(v) == [False, True]