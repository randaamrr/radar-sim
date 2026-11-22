import numpy as np
from src.scenario import load_scenario
from src.compare import grid_convergence, room_at_step


def test_total_strength_independent_of_step():
    # Refining the grid must not change total reflected strength,
    # only how it is sampled.
    cfg = load_scenario("scenarios/room_basic.yaml")
    coarse = room_at_step(cfg, 0.4)
    fine = room_at_step(cfg, 0.1)
    assert np.isclose(coarse["strength"].sum(), fine["strength"].sum())


def test_error_shrinks_with_finer_grid():
    cfg = load_scenario("scenarios/room_basic.yaml")
    res = grid_convergence(cfg, steps=(0.4, 0.1), ref_step=0.05)
    assert res[0]["n_points"] < res[1]["n_points"]
    assert res[0]["rel_error"] > res[1]["rel_error"]
def test_default_step_is_fine_enough_for_15cm_resolution():
    cfg = load_scenario("scenarios/room_basic.yaml")
    res = grid_convergence(cfg, steps=(cfg["room"]["grid_step_m"],),
                           range_res_m=0.15)
    assert res[0]["rel_error"] < 0.10