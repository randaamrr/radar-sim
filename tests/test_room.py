import numpy as np
from src.scenario import load_scenario
from src.room import build_shell


def test_shell_point_count_and_area():
    cfg = load_scenario("scenarios/room_basic.yaml")
    shell = build_shell(cfg)
    # 4x3 -> 20x15 = 300 نقطة للأرضية، ونفس الشيء للسقف
    assert (shell["surface"] == "floor").sum() == 1200
    assert (shell["surface"] == "ceiling").sum() == 1200
    # مجموع المساحات لازم يساوي مساحة الغرفة الكلية
    L, W, H = cfg["room"]["size_m"]
    total = 2 * (L * W + L * H + W * H)
    assert np.isclose(shell["area"].sum(), total)


def test_points_inside_room():
    cfg = load_scenario("scenarios/room_basic.yaml")
    shell = build_shell(cfg)
    L, W, H = cfg["room"]["size_m"]
    p = shell["points"]
    assert p.min() >= 0
    assert p[:, 0].max() <= L and p[:, 1].max() <= W and p[:, 2].max() <= H