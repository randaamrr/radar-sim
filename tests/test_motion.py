import copy
import json

import numpy as np
import pytest

from src import motion as M
from src.people import joints_for, RT
from src.scenario import load_scenario, ScenarioError, _validate_people
from src.motion_library import (generate_library, save_library, load_sequence,
                                person_sequence, build, KINDS, TAGS)

H = 1.7
FPS = 20
SEGMENTS = [("chest", "pelvis", M.CHEST), ("neck", "pelvis", M.NECK), ("head", "pelvis", M.HEAD),
            ("elbow_l", "shoulder_l", M.UPPER_ARM), ("wrist_l", "elbow_l", M.FOREARM),
            ("knee_r", "hip_r", M.THIGH), ("ankle_r", "knee_r", M.SHIN)]


def _cfg():
    return load_scenario("scenarios/room_motion.yaml")


def _all_sequences():
    return [M.walk(H, FPS, 2.0), M.sit_down(H, FPS), M.stand_up(H, FPS),
            M.fall(H, FPS, 0), M.fall(H, FPS, 90), M.fall(H, FPS, 180),
            M.still("lie", H, FPS, 1.0), M.still("stand", H, FPS, 1.0)]


def test_fk_reproduces_the_static_poses():
    for name, pose in (("stand", M.STAND), ("sit", M.SIT)):
        J = M.skeleton(H, pose)
        ref = joints_for(name, H)
        for jn in M.JOINTS:
            assert np.allclose(J[M.IDX[jn]] - J[M.IDX["pelvis"]],
                               ref[jn] - ref["pelvis"], atol=1e-9), (name, jn)


def test_lying_pose_has_the_same_segment_lengths_as_the_reference():
    J = M.skeleton(H, M.LIE)
    ref = joints_for("lie", H)
    d = lambda A, a, b: np.linalg.norm(A[a] - A[b])
    for a, b in (("head", "pelvis"), ("ankle_l", "hip_l"), ("wrist_r", "shoulder_r")):
        assert np.isclose(d({k: J[M.IDX[k]] for k in M.JOINTS}, a, b), d(ref, a, b), atol=1e-9)


@pytest.mark.parametrize("seq", _all_sequences(), ids=lambda s: s.kind)
def test_limb_lengths_never_change(seq):
    for child, parent, frac in SEGMENTS:
        d = np.linalg.norm(seq.joints[:, M.IDX[child]] - seq.joints[:, M.IDX[parent]], axis=1)
        assert np.allclose(d, frac * H, atol=1e-9), child


@pytest.mark.parametrize("seq", _all_sequences(), ids=lambda s: s.kind)
def test_every_frame_has_the_same_mesh_topology(seq):
    a, b = seq.parts(0), seq.parts(seq.n_frames - 1)
    for (_, ma), (_, mb) in zip(a, b):
        assert ma.vertices.shape == mb.vertices.shape and ma.is_watertight


def test_walk_speed_stride_and_ground_contact():
    seq = M.walk(H, FPS, 4.0, speed=1.31)
    pel = seq.joints[:, M.IDX["pelvis"]]
    speed = np.linalg.norm(pel[-1, :2] - pel[0, :2]) / seq.duration
    assert np.isclose(speed, 1.31, rtol=1e-6)
    low = [min(m.bounds[0, 2] for _, m in seq.parts(k)) for k in range(0, seq.n_frames, 5)]
    assert np.allclose(low, 0.0, atol=2e-3)               # a foot is always on the floor (mesh tessellation < 2 mm)
    assert np.ptp(pel[:, 2]) < 0.08                        # vertical bob below 8 cm
    lead = seq.joints[:, M.IDX["ankle_l"], 0] - seq.joints[:, M.IDX["ankle_r"], 0]
    crossings = int((np.diff(np.sign(lead)) != 0).sum())
    expected = 2 * seq.params["f_cycle"] * seq.duration
    assert abs(crossings - expected) <= 1.5                # legs alternate


def test_sit_down_goes_from_stand_to_the_sit_pose_and_stand_up_reverses_it():
    s = M.sit_down(H, FPS, 2.4, seat_z=0.5, back_m=0.25)
    j0, j1 = s.joints[0], s.joints[-1]
    ref = joints_for("stand", H)
    for jn in M.JOINTS:
        assert np.allclose(j0[M.IDX[jn]] - j0[0], ref[jn] - ref["pelvis"], atol=1e-9)
    ref = joints_for("sit", H)
    for jn in M.JOINTS:
        assert np.allclose(j1[M.IDX[jn]] - j1[0], ref[jn] - ref["pelvis"], atol=1e-9)
    assert np.isclose(j1[0, 2], 0.5 + RT * H)
    assert np.isclose(j1[0, 0] - j0[0, 0], -0.25)
    low = [min(m.bounds[0, 2] for _, m in s.parts(k)) for k in range(s.n_frames)]
    assert min(low) >= -2e-3                                   # nothing goes below the floor
    u = M.stand_up(H, FPS, 2.4, seat_z=0.5, back_m=0.25)
    assert np.allclose(u.joints[0, 0, :2], 0.0)
    assert np.allclose(u.joints[:, 1:, :] - u.joints[:, :1, :],
                       (s.joints[::-1, 1:, :] - s.joints[::-1, :1, :]))


def test_fall_lowers_the_head_monotonically_and_ends_lying():
    for az in (0, 90, 180, 270):
        s = M.fall(H, FPS, az, duration=0.8, pre_s=1.0, rest_s=2.0)
        head = s.joints[:, M.IDX["head"], 2]
        i0, i1 = int(1.0 * FPS), int(1.8 * FPS)
        assert np.all(np.diff(head[i0:i1 + 1]) <= 1e-9)
        assert np.allclose(head[:i0 + 1], head[0])                  # stands still first
        assert np.allclose(head[i1:], head[-1])                     # then lies still
        top = max(m.bounds[1, 2] for _, m in s.parts(s.n_frames - 1))
        assert top < 0.55          # lying: body thickness (side-lying is thicker than supine)
        assert head[0] > 1.4


def test_fall_direction_matches_the_requested_azimuth():
    for az, vec in ((0, (1, 0)), (180, (-1, 0)), (90, (0, 1)), (270, (0, -1))):
        s = M.fall(H, FPS, az)
        d = s.joints[-1, M.IDX["head"], :2] - s.joints[-1, M.IDX["ankle_l"], :2]
        d = d / np.linalg.norm(d)
        assert np.dot(d, vec) > 0.95, az


def test_fall_pivots_about_the_feet():
    s = M.fall(H, FPS, 0)
    ank = s.joints[:, [M.IDX["ankle_l"], M.IDX["ankle_r"]], :2].mean(axis=1)
    assert np.allclose(ank, ank[0], atol=1e-9)


def test_breathing_amplitude_rate_and_speed():
    base = M.still("stand", H, 20, 20.0)
    seq = M.with_breathing(base, hz=0.25, p2p_mm=5.0)
    assert np.isclose(np.ptp(seq.breath), 5e-3, rtol=0.02)
    spec = np.abs(np.fft.rfft(seq.breath))
    f = np.fft.rfftfreq(len(seq.breath), 1 / 20)
    assert np.isclose(f[spec.argmax()], 0.25, atol=0.05)
    peak_speed = np.abs(np.diff(seq.breath)).max() * 20
    assert np.isclose(peak_speed, np.pi * 0.25 * 5e-3, rtol=0.05)   # pi * f * p2p
    r0 = seq.parts(0)[0][1].bounds
    rk = seq.parts(int(np.argmax(seq.breath)))[0][1].bounds
    assert rk[1, 0] - r0[1, 0] > 0.9 * 2.5e-3                       # torso surface moved ~2.5 mm


def test_scenario_motions_are_valid_and_fps_comes_from_the_scenario():
    cfg = _cfg()
    assert cfg["fps"] == 20
    seq = person_sequence(cfg, cfg["people"][1], 1)
    assert seq.fps == 20 and np.isclose(seq.duration, 1.0 + 0.8 + 2.0)
    assert M.validate_sequence(cfg, seq) == []
    cfg2 = copy.deepcopy(cfg)
    cfg2["fps"] = 10
    assert person_sequence(cfg2, cfg2["people"][1], 1).n_frames == 39
    assert load_scenario("scenarios/room_basic.yaml")["fps"] == 20


def test_walking_into_the_table_is_caught():
    cfg = _cfg()
    seq = M.place(M.walk(H, FPS, 2.0), 2.5, 1.0, 90.0)       # towards the table at y >= 2.0
    problems = M.validate_sequence(cfg, seq)
    assert any("table" in p for p in problems)
    seq_ok = M.place(M.walk(H, FPS, 1.0), 2.75, 0.4, 90.0)   # free corridor between bed and wardrobe
    assert M.validate_sequence(cfg, seq_ok) == []


def test_bad_motion_sections_are_rejected():
    cfg = _cfg()
    for motion in ({"kind": "moonwalk"}, {"kind": "walk", "speed_mps": 1.0},
                   {"kind": "walk", "duration": 3.0}):            # 1.3 m/s * 3 s hits the wall/table
        c = copy.deepcopy(cfg)
        c["people"] = [copy.deepcopy(c["people"][1])]
        c["people"][0]["motion"] = motion
        c["people"][0]["pos_m"] = [2.75, 0.4]
        c["people"][0]["yaw_deg"] = 90
        with pytest.raises(ScenarioError):
            _validate_people(c)


def test_library_has_all_kinds_tags_and_is_deterministic():
    a, b = generate_library(7, per_kind=4, fps=FPS), generate_library(7, per_kind=4, fps=FPS)
    assert {it["kind"] for it in a} == set(KINDS)
    for x, y in zip(a, b):
        assert np.array_equal(x["sequence"].joints, y["sequence"].joints)
        assert x["tags"] == TAGS[x["kind"]]
    c = generate_library(8, per_kind=4, fps=FPS)
    assert not np.array_equal(a[0]["sequence"].joints, c[0]["sequence"].joints)


def test_falls_differ_between_people():
    lib = [it for it in generate_library(3, per_kind=12, fps=FPS) if it["kind"] == "fall"]
    p = lambda k: np.array([it["params"][k] for it in lib])
    assert len(set(np.round(p("duration"), 3))) == 12
    assert p("duration").min() >= 0.5 and p("duration").max() <= 1.0
    assert np.ptp(p("knee_buckle")) > 15 and len(set(np.round(p("direction_deg") // 45))) >= 3


def test_library_roundtrip_on_disk(tmp_path):
    lib = generate_library(5, per_kind=2, fps=FPS)
    index = save_library(lib, str(tmp_path))
    assert len(index) == len(lib) and (tmp_path / "index.json").exists()
    meta = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
    assert meta["joint_names"] == M.JOINTS
    it = lib[5]
    s = load_sequence(str(tmp_path / (it["id"] + ".npz")))
    assert np.array_equal(s.joints, it["sequence"].joints)
    assert np.array_equal(s.breath, it["sequence"].breath)
    rebuilt = build(it["kind"], it["params"], FPS)
    assert np.array_equal(rebuilt.joints, s.joints)
