import copy
import numpy as np
from src.scenario import load_scenario
from src.generator import save_scenario, _overlap
from src.surface import scene_samples
from src.small_objects import add_small_objects, evaluate
from tests.test_surface import CFG, RADAR


def test_small_objects_valid_and_reproducible(tmp_path):
    base = load_scenario("scenarios/room_basic.yaml")
    a = add_small_objects(base, 6, seed=3)
    assert a == add_small_objects(base, 6, seed=3)
    assert len(a["furniture"]) > len(base["furniture"])
    path = tmp_path / "s.yaml"
    save_scenario(a, str(path))
    load_scenario(str(path))          # raises if anything is outside the room
    boxes = [(f["pos_m"][0], f["pos_m"][1], f["size_m"][0], f["size_m"][1])
             for f in a["furniture"]]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            assert not _overlap(boxes[i], boxes[j])


def test_no_change_gives_zero():
    res = evaluate(CFG, CFG, [RADAR])[0]
    assert all(v == 0.0 for v in res["l2"].values())
    assert np.allclose(res["window_change_pct"], 0.0)


def test_tiny_object_matters_less_than_big_furniture():
    mug = copy.deepcopy(CFG)
    mug["furniture"].append({"name": "mug", "pos_m": [0.5, 0.5],
                             "size_m": [0.1, 0.1, 0.1], "material": "wood"})
    nobig = copy.deepcopy(CFG)
    nobig["furniture"] = []
    small = evaluate(CFG, mug, [RADAR])[0]["l2"][0.15]
    big = evaluate(CFG, nobig, [RADAR])[0]["l2"][0.15]
    assert 0 < small < big


def test_room_without_furniture_builds():
    empty = copy.deepcopy(CFG)
    empty["furniture"] = []
    assert len(scene_samples(empty)["points"]) > 1000