import numpy as np
import pytest
from src.scenario import load_scenario, radar_direction, ScenarioError

BASE = (
    "seed: 1\nroom: {size_m: [4,3,2.7], grid_step_m: 0.2}\n"
    "surfaces: {walls: concrete, floor: concrete, ceiling: concrete}\n"
    "furniture: []\n"
)


def _load_text(tmp_path, extra):
    f = tmp_path / "s.yaml"
    f.write_text(BASE + extra, encoding="utf-8")
    return load_scenario(str(f))


def test_two_radars_loaded():
    cfg = load_scenario("scenarios/room_basic.yaml")
    assert [r["name"] for r in cfg["radars"]] == ["radar_west", "radar_east"]


def test_radar_outside_room_rejected(tmp_path):
    extra = "radars:\n  - {name: r, pos_m: [5, 1, 1], yaw_deg: 0, pitch_deg: 0}\n"
    with pytest.raises(ScenarioError):
        _load_text(tmp_path, extra)


def test_bad_pitch_rejected(tmp_path):
    extra = "radars:\n  - {name: r, pos_m: [1, 1, 1], yaw_deg: 0, pitch_deg: 120}\n"
    with pytest.raises(ScenarioError):
        _load_text(tmp_path, extra)


def test_direction_vectors():
    assert np.allclose(radar_direction(0, 0), [1, 0, 0])
    assert np.allclose(radar_direction(180, 0), [-1, 0, 0], atol=1e-9)
    d = radar_direction(0, -20)
    assert d[2] < 0 and np.isclose(np.linalg.norm(d), 1.0)