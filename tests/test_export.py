import numpy as np
from src.scenario import load_scenario
from src.room import build_room
from src.materials import apply_materials
from src.export import export_scene, load_exported
import copy
import trimesh
from tests.test_surface import CFG

def test_roundtrip_matches_scene(tmp_path):
    cfg = load_scenario("scenarios/room_basic.yaml")
    path = tmp_path / "scene.npz"
    meta = export_scene(cfg, str(path))

    original = apply_materials(build_room(cfg))
    loaded = load_exported(str(path))

    assert meta["n_points"] == len(original["points"])
    assert np.array_equal(loaded["points"], original["points"])
    assert np.allclose(loaded["strength"], original["strength"])
    assert list(loaded["material"]) == list(original["material"])


def test_radars_exported(tmp_path):
    cfg = load_scenario("scenarios/room_basic.yaml")
    path = tmp_path / "scene.npz"
    export_scene(cfg, str(path))
    loaded = load_exported(str(path))
    assert list(loaded["radar_names"]) == ["radar_west", "radar_east"]
    assert loaded["radar_pos_m"].shape == (2, 3)


def test_same_seed_same_file(tmp_path):
    cfg = load_scenario("scenarios/room_basic.yaml")
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    export_scene(cfg, str(a))
    export_scene(cfg, str(b))
    la, lb = load_exported(str(a)), load_exported(str(b))
    assert np.array_equal(la["points"], lb["points"])


def test_v4_has_normals_and_meshes(tmp_path):
    cfg = load_scenario("scenarios/room_basic.yaml")
    path = tmp_path / "scene.npz"
    meta = export_scene(cfg, str(path))
    loaded = load_exported(str(path))
    assert meta["format_version"] == 4
    assert loaded["normal"].shape == loaded["points"].shape
    assert np.allclose(np.linalg.norm(loaded["normal"], axis=1), 1.0)
    mesh = trimesh.Trimesh(loaded["occluder_vertices"], loaded["occluder_faces"])
    expected = sum(np.prod(f["size_m"]) for f in cfg["furniture"])
    assert np.isclose(mesh.volume, expected)


def test_export_room_without_furniture(tmp_path):
    cfg = copy.deepcopy(CFG)
    cfg["furniture"] = []
    path = tmp_path / "empty.npz"
    export_scene(cfg, str(path))
    loaded = load_exported(str(path))
    assert loaded["occluder_faces"].shape == (0, 3)
    assert len(loaded["furniture_names"]) == 0