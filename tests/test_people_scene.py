import copy

import numpy as np
import pytest
import trimesh

from src.scenario import load_scenario, ScenarioError
from src.surface import scene_samples, radar_view
from src.room import build_room
from src.materials import body_reflectance
from src.people import cfg_people
from src.meshes import occluder_scene
from src.export import export_scene, load_exported
from src.ghosts import first_order_ghosts_view
from tests.test_surface import CFG

PATH = "scenarios/room_people.yaml"


def _cfg():
    return load_scenario(PATH)


def _bad(**person):
    cfg = load_scenario("scenarios/room_basic.yaml")
    base = {"name": "x", "pose": "stand", "height_m": 1.7, "pos_m": [1.3, 2.3]}
    base.update(person)
    cfg["people"] = [base]
    return cfg


def _validate(cfg):
    from src.scenario import _validate_people
    _validate_people(cfg)


def test_scenario_with_people_loads_with_defaults():
    cfg = _cfg()
    assert [p["name"] for p in cfg["people"]] == ["p_stand", "p_sit", "p_lie"]
    assert load_scenario("scenarios/room_basic.yaml")["people"] == []


@pytest.mark.parametrize("change", [
    {"pose": "dance"},
    {"height_m": 3.0},
    {"pos_m": [0.05, 2.5]},                 # inside the wall
    {"pos_m": [3.6, 0.7]},                  # inside the wardrobe
    {"support_z": 0.9, "pos_m": [1.0, 0.8]},  # floating above the bed
    {"yaw_deg": 400},
])
def test_invalid_people_are_rejected(change):
    with pytest.raises(ScenarioError):
        _validate(_bad(**change))


def test_duplicate_names_and_overlapping_bodies_are_rejected():
    cfg = _bad()
    cfg["people"].append(dict(cfg["people"][0]))
    with pytest.raises(ScenarioError, match="duplicate"):
        _validate(cfg)
    cfg = _bad()
    cfg["people"].append(dict(cfg["people"][0], name="y", pos_m=[1.35, 2.3]))
    with pytest.raises(ScenarioError, match="overlaps"):
        _validate(cfg)


def test_no_people_changes_nothing():
    cfg = load_scenario("scenarios/room_basic.yaml")
    assert len(scene_samples(cfg)["points"]) == len(build_room(cfg)["points"])


def test_people_samples_are_consistent():
    cfg = _cfg()
    room = scene_samples(cfg)
    n = len(room["points"])
    for key in ("normal", "area", "strength", "reflectance", "material", "surface"):
        assert len(room[key]) == n
    assert np.allclose(np.linalg.norm(room["normal"], axis=1), 1.0)
    body = room["material"] == "skin"
    assert body.any() and np.isclose(room["reflectance"][body], body_reflectance()).all()
    assert np.allclose(room["strength"], room["area"] * room["reflectance"])


def test_skin_reflectance_is_inside_the_published_range():
    # Wu et al. 2015: 34-42% of normally incident power at 60 GHz
    assert 0.34 <= body_reflectance() <= 0.42


def test_samples_lie_on_the_outer_skin_with_outward_normals():
    cfg = _cfg()
    room = scene_samples(cfg)
    for name, parts in cfg_people(cfg):
        own = np.char.startswith(room["surface"], name + "_")
        pts, nrm = room["points"][own], room["normal"][own]
        # nothing sampled inside another capsule of the same body
        for part, mesh in parts:
            mine = room["surface"][own] == f"{name}_{part}"
            for other, om in parts:
                if other != part:
                    assert not om.contains(pts[mine]).any()
            # normals point out of their own capsule
            assert not mesh.contains(pts[mine] + 0.01 * nrm[mine]).any()
            assert mesh.contains(pts[mine] - 0.01 * nrm[mine]).all()


def test_body_area_is_plausible_and_below_the_capsule_total():
    cfg = _cfg()
    room = scene_samples(cfg)
    for name, parts in cfg_people(cfg):
        own = np.char.startswith(room["surface"], name + "_")
        total = sum(m.area for _, m in parts)
        got = room["area"][own].sum()
        assert 0.6 * total < got < total        # overlaps removed
        assert 1.2 < got < 2.4                  # m^2, adult-sized body


def test_samples_are_deterministic():
    a, b = scene_samples(_cfg()), scene_samples(_cfg())
    assert np.array_equal(a["points"], b["points"])


def test_person_hides_wall_samples_from_the_radar():
    cfg = copy.deepcopy(CFG)
    cfg["furniture"] = []
    cfg["people"] = [{"name": "p", "pose": "stand", "height_m": 1.7,
                      "pos_m": [2.0, 1.5], "yaw_deg": 0, "support_z": 0.0}]
    room = scene_samples(cfg)
    view = radar_view(room, cfg, [0.0, 1.5, 1.0])
    p = room["points"]
    shadow = ((room["surface"] == "wall_x1") & (abs(p[:, 1] - 1.5) < 0.15)
              & (abs(p[:, 2] - 1.0) < 0.3))
    assert shadow.any() and (view["strength"][shadow] == 0).all()


def test_furniture_hides_the_legs_of_a_person_but_not_the_head():
    cfg = copy.deepcopy(CFG)             # box x 1.5-2.5, y 1-2, height 1.0
    cfg["people"] = [{"name": "p", "pose": "stand", "height_m": 1.7,
                      "pos_m": [3.2, 1.5], "yaw_deg": 0, "support_z": 0.0}]
    room = scene_samples(cfg)
    view = radar_view(room, cfg, [0.0, 1.5, 0.5])
    shin = np.char.endswith(room["surface"], "_shin_l") | np.char.endswith(room["surface"], "_shin_r")
    head = np.char.endswith(room["surface"], "_head")
    assert (view["strength"][shin] == 0).all()
    assert (view["strength"][head] > 0).any()


def test_person_is_not_blocked_by_itself():
    cfg = copy.deepcopy(CFG)
    cfg["furniture"] = []
    cfg["people"] = [{"name": "p", "pose": "stand", "height_m": 1.7,
                      "pos_m": [2.0, 1.5], "yaw_deg": 0, "support_z": 0.0}]
    room = scene_samples(cfg)
    radar = np.array([0.0, 1.5, 1.2])
    body = room["material"] == "skin"
    vec = radar - room["points"][body]
    cos = (room["normal"][body] * vec).sum(1) / np.linalg.norm(vec, axis=1)
    facing = cos > 0.3
    seen = radar_view(room, cfg, radar)["strength"][body] > 0
    assert facing.sum() > 20
    assert seen[facing].mean() > 0.8      # arms hide part of the torso side


def test_export_v3_contains_people(tmp_path):
    cfg = _cfg()
    meta = export_scene(cfg, str(tmp_path / "s.npz"))
    d = load_exported(str(tmp_path / "s.npz"))
    assert meta["format_version"] == 4
    assert list(d["people_names"]) == ["p_stand", "p_sit", "p_lie"]
    assert d["people_faces"].max() < len(d["people_vertices"])
    assert d["people_face_part"].max() < len(d["part_names"])
    assert set(d["people_face_person"]) == {0, 1, 2}
    furn_only = trimesh.util.concatenate(
        [m for _, _, m in __import__("src.meshes", fromlist=["x"]).furniture_meshes(cfg)])
    assert len(d["occluder_faces"]) == len(furn_only.faces) + len(d["people_faces"])


def test_ghosts_still_run_with_people():
    cfg = _cfg()
    room = scene_samples(cfg)
    r, p = first_order_ghosts_view(room, cfg, cfg["radars"][0]["pos_m"])
    assert len(r) == len(p) > 0 and np.isfinite(p).all()
