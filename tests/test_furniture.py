import numpy as np
from src.scenario import load_scenario
from src.room import build_shell, build_furniture, build_room


def _box_area(w, d, h):
    # وش علوي + 4 جنبات (من غير القاع)
    return w * d + 2 * w * h + 2 * d * h


def test_furniture_area_matches_boxes():
    cfg = load_scenario("scenarios/room_basic.yaml")
    furn = build_furniture(cfg)
    expected = sum(_box_area(*it["size_m"]) for it in cfg["furniture"])
    assert np.isclose(furn["area"].sum(), expected)


def test_each_item_has_points():
    cfg = load_scenario("scenarios/room_basic.yaml")
    furn = build_furniture(cfg)
    for it in cfg["furniture"]:
        n = np.char.startswith(furn["surface"], it["name"] + "_").sum()
        assert n > 0, f"{it['name']} ملوش نقط"


def test_furniture_points_inside_room():
    cfg = load_scenario("scenarios/room_basic.yaml")
    furn = build_furniture(cfg)
    L, W, H = cfg["room"]["size_m"]
    p = furn["points"]
    assert p.min() >= 0
    assert p[:, 0].max() <= L and p[:, 1].max() <= W and p[:, 2].max() <= H


def test_build_room_combines_everything():
    cfg = load_scenario("scenarios/room_basic.yaml")
    room = build_room(cfg)
    n = len(build_shell(cfg)["points"]) + len(build_furniture(cfg)["points"])
    assert len(room["points"]) == n
    assert len(room["points"]) == len(room["area"]) == len(room["material"])