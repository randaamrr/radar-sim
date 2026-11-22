import numpy as np
from src.scenario import load_scenario
from src.room import build_room
from src.materials import apply_materials
from src.generator import generate_scenario, save_scenario, _overlap


def test_same_seed_same_room():
    assert generate_scenario(7) == generate_scenario(7)


def test_different_seeds_differ():
    sizes = {tuple(generate_scenario(s)["room"]["size_m"]) for s in range(10)}
    assert len(sizes) > 5


def test_many_seeds_pass_validation(tmp_path):
    # the generator output must be accepted by the same validator as
    # hand-written files
    for seed in range(50):
        path = tmp_path / f"r{seed}.yaml"
        save_scenario(generate_scenario(seed), str(path))
        cfg = load_scenario(str(path))
        assert len(cfg["radars"]) == 2


def test_no_overlap_and_footprint_limit():
    for seed in range(50):
        cfg = generate_scenario(seed)
        L, W, _ = cfg["room"]["size_m"]
        boxes = [(f["pos_m"][0], f["pos_m"][1], f["size_m"][0], f["size_m"][1])
                 for f in cfg["furniture"]]
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                assert not _overlap(boxes[i], boxes[j])
        assert sum(b[2] * b[3] for b in boxes) <= 0.40 * L * W + 1e-9


def test_generated_room_builds_points(tmp_path):
    path = tmp_path / "r.yaml"
    save_scenario(generate_scenario(3), str(path))
    room = apply_materials(build_room(load_scenario(str(path))))
    assert len(room["points"]) > 1000
    assert (room["strength"] > 0).all()