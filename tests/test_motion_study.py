import numpy as np

from src import motion as M
from src.motion_study import radial_tracks, features, dprime, PART_NAMES, RADAR


def test_dprime_is_zero_for_equal_groups_and_grows_with_the_gap():
    a = np.array([1.0, 1.1, 0.9, 1.0])
    assert dprime(a, a) < 1e-6
    assert dprime(a, a + 1.0) > 5


def test_breathing_radial_speed_matches_pi_f_p2p():
    seq = M.place(M.with_breathing(M.still("stand", 1.7, 20, 8.0), 0.25, 5.0), 2.0, 1.5, 0.0)
    vr, w = radial_tracks(seq)
    torso = PART_NAMES.index("torso")
    peak = np.abs(vr[:, torso]).max()
    assert np.isclose(peak, np.pi * 0.25 * 5e-3, rtol=0.35)   # radial component of a normal motion


def test_a_walker_has_a_much_larger_bulk_speed_than_a_breather():
    walk = M.place(M.walk(1.7, 20, 2.0), 1.0, 1.5, 0.0)       # walks away from the radar
    vr, w = radial_tracks(walk)
    spread, bulk = features(vr, w, list(range(len(PART_NAMES))), 20)
    assert 0.8 < bulk < 1.8 and spread > 0.1
