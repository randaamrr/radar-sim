import numpy as np
import pytest
from src.scenario import load_scenario, KNOWN_MATERIALS
from src.room import build_room
from src.materials import MATERIALS, reflectance, apply_materials


def test_table_covers_all_known_materials():
    assert set(MATERIALS) == set(KNOWN_MATERIALS)


def test_reflectance_values_and_order():
    assert np.isclose(reflectance("concrete"), 0.156, atol=0.005)
    assert reflectance("metal") == 1.0
    assert (reflectance("wood") < reflectance("plasterboard")
            < reflectance("concrete") < reflectance("glass") < 1.0)


def test_apply_materials_shapes_and_values():
    cfg = load_scenario("scenarios/room_basic.yaml")
    room = apply_materials(build_room(cfg))
    n = len(room["points"])
    assert len(room["reflectance"]) == len(room["strength"]) == n
    assert (room["strength"] > 0).all()
    assert np.allclose(room["strength"], room["area"] * room["reflectance"])


def test_unknown_material_raises():
    with pytest.raises(KeyError):
        reflectance("unobtainium")