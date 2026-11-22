import pytest
from src.scenario import load_scenario, ScenarioError


def test_valid_file_loads():
    cfg = load_scenario("scenarios/room_basic.yaml")
    assert cfg["room"]["size_m"] == [4.0, 3.0, 2.7]


def test_furniture_outside_room_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "seed: 1\nroom: {size_m: [4,3,2.7], grid_step_m: 0.2}\n"
        "surfaces: {walls: concrete, floor: concrete, ceiling: concrete}\n"
        "furniture:\n  - {name: bed, pos_m: [3.5, 0], size_m: [2,1,0.5], material: wood}\n",
        encoding="utf-8",
    )
    with pytest.raises(ScenarioError):
        load_scenario(str(bad))