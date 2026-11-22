import copy
import numpy as np
from src.scenario import load_scenario
from src.surface import scene_samples
from src.ghosts import first_order_ghosts_view
from src.ghosts2 import second_order_ghosts_view
from tests.test_surface import CFG


def test_known_floor_ceiling_path():
    cfg = copy.deepcopy(CFG)
    cfg["furniture"] = []
    radar = [0.0, 1.5, 1.0]
    room = {"points": np.array([[2.0, 1.5, 1.0]]), "strength": np.array([1.0]),
            "normal": np.array([[-1.0, 0.0, 0.0]])}
    r, p = second_order_ghosts_view(room, cfg, radar, facing=False, occlusion=False)
    d = 2.0
    d2 = np.sqrt(2.0 ** 2 + 5.4 ** 2)        # double image of the radar (floor, ceiling)
    assert any(np.isclose(x, (d + d2) / 2) for x in r)
    assert (p > 0).all() and np.isfinite(p).all()


def test_second_order_weaker_than_first_and_effects_reduce_power():
    cfg = copy.deepcopy(load_scenario("scenarios/room_basic.yaml"))
    cfg["room"]["grid_step_m"] = 0.2
    room = scene_samples(cfg)
    pos = cfg["radars"][0]["pos_m"]
    f = first_order_ghosts_view(room, cfg, pos)[1].sum()
    s_ang = second_order_ghosts_view(room, cfg, pos, occlusion=False)[1].sum()
    s_all = second_order_ghosts_view(room, cfg, pos)[1].sum()
    assert 0 < s_all < s_ang
    assert s_all < f


def test_no_ghosts_when_target_hidden():
    room = {"points": np.array([[3.95, 1.5, 0.5]]), "strength": np.array([1.0]),
            "normal": np.array([[-1.0, 0.0, 0.0]])}
    r, _ = second_order_ghosts_view(room, CFG, [0.0, 1.5, 0.5],
                                    facing=False, occlusion=True)
    assert len(r) == 0