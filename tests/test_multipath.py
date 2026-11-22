import numpy as np
from src.scenario import load_scenario
from src.multipath import mirror, first_order_ghosts

RADAR = [0.0, 1.5, 2.2]  # mounted on wall_x0


def _one_point_room(p):
    return {"points": np.array([p], dtype=float), "strength": np.array([1.0])}


def test_mirror():
    assert np.allclose(mirror([1, 1, 1], 0, 0.0), [-1, 1, 1])
    assert np.allclose(mirror([1, 1, 1], 2, 2.7), [1, 1, 4.4])


def test_known_floor_ghost_range():
    cfg = load_scenario("scenarios/room_basic.yaml")
    p = [2.0, 1.5, 1.0]
    ranges, _ = first_order_ghosts(_one_point_room(p), cfg, RADAR)
    d = np.linalg.norm(np.subtract(p, RADAR))
    d_img = np.linalg.norm(np.subtract(p, [0.0, 1.5, -2.2]))  # floor image
    assert any(np.isclose(r, (d + d_img) / 2) for r in ranges)


def test_ghosts_are_farther_and_weaker_than_direct():
    cfg = load_scenario("scenarios/room_basic.yaml")
    p = [2.0, 1.5, 1.0]
    d = np.linalg.norm(np.subtract(p, RADAR))
    ranges, powers = first_order_ghosts(_one_point_room(p), cfg, RADAR)
    assert (ranges >= d).all()
    assert (powers < 1.0 / d ** 4).all()


def test_radar_own_wall_skipped():
    cfg = load_scenario("scenarios/room_basic.yaml")
    ranges, _ = first_order_ghosts(_one_point_room([2.0, 1.5, 1.0]), cfg, RADAR)
    assert len(ranges) == 5  # 6 planes minus the one the radar is mounted on