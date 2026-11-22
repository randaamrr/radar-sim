import copy
import glob

import numpy as np
import pytest

from src.radar_config import (C, RadarConfig, RadarConfigError, config_from_spec,
                              radar_config_for, PRESETS)
from src.scenario import load_scenario, ScenarioError, _validate_radar_config
from src.export import export_scene, load_exported


def test_range_resolution_is_c_over_2b():
    assert np.isclose(RadarConfig({}).range_resolution_m, C / 2e9)
    assert np.isclose(RadarConfig({"bandwidth_hz": 4e9}).range_resolution_m, 0.0375, atol=1e-4)


def test_max_range_is_n_samples_times_resolution_and_halves_for_real_adc():
    rc = RadarConfig({})
    assert np.isclose(rc.max_range_m, rc.n_samples * rc.range_resolution_m)
    real = RadarConfig({"adc_mode": "real"})
    assert np.isclose(real.max_range_m, rc.max_range_m / 2)
    # the beat frequency of a target at max range is the sample rate (I/Q) or half of it (real)
    assert np.isclose(rc.beat_frequency_max_hz, rc.sample_rate_hz)
    assert np.isclose(real.beat_frequency_max_hz, real.sample_rate_hz / 2)
    # beat frequency law f_b = 2 S R / c
    assert np.isclose(2 * rc.slope_hz_per_s * 1.0 / C, 2 * (rc.bandwidth_hz / rc.chirp_duration_s) / C)


def test_velocity_limits_and_their_identity():
    rc = RadarConfig({})
    tp = rc.n_tx * (rc.chirp_duration_s + rc.idle_s)
    assert np.isclose(rc.chirp_period_s, tp)
    assert np.isclose(rc.max_velocity_mps, rc.wavelength_m / (4 * tp))
    assert np.isclose(rc.velocity_resolution_mps, rc.wavelength_m / (2 * rc.n_chirps * tp))
    assert np.isclose(rc.max_velocity_mps / rc.velocity_resolution_mps, rc.n_chirps / 2)
    one = RadarConfig({"array": {"n_tx": 1}})
    assert np.isclose(one.max_velocity_mps, 2 * rc.max_velocity_mps)     # TDM halves it for 2 TX
    assert np.isclose(rc.frame_time_s, rc.n_chirps * tp)


def test_wavelength_uses_the_centre_frequency():
    rc = RadarConfig({"f0_hz": 77e9, "bandwidth_hz": 4e9})
    assert np.isclose(rc.f_center_hz, 79e9) and np.isclose(rc.wavelength_m, C / 79e9)


def test_angle_resolution_and_field_of_view():
    assert np.isclose(RadarConfig({}).angle_resolution_deg, np.degrees(2 / 8))
    big = RadarConfig({"array": {"n_tx": 3, "n_rx": 4}})
    assert np.isclose(big.angle_resolution_deg, np.degrees(2 / 12))
    assert RadarConfig({}).max_angle_deg == 90.0
    wide = RadarConfig({"array": {"rx_spacing_wl": 1.0}})
    assert np.isclose(wide.max_angle_deg, 30.0)


@pytest.mark.parametrize("bad", [
    {"bandwidth_hz": -1}, {"bandwidth_hz": 0}, {"f0_hz": 5}, {"chirp_duration_s": 0},
    {"idle_s": -1e-6}, {"n_chirps": 1}, {"n_chirps": 12.5}, {"n_samples": 4},
    {"adc_mode": "analog"}, {"array": {"n_rx": 0}}, {"array": {"rx_spacing_wl": 5.0}},
    {"unknown_field": 1}, {"array": {"n_banana": 1}}, {"bandwidth_hz": 100e9},
])
def test_invalid_configs_are_rejected(bad):
    with pytest.raises(RadarConfigError):
        RadarConfig(bad)


def test_presets_and_overrides_merge():
    assert config_from_spec(None).bandwidth_hz == 1e9
    wide = config_from_spec({"preset": "wide_4ghz", "n_chirps": 64})
    assert wide.bandwidth_hz == 4e9 and wide.n_samples == 256 and wide.n_chirps == 64
    assert wide.n_tx == 2 and wide.n_rx == 4                     # array kept from defaults
    tiny = config_from_spec({"preset": "tiny"})
    assert tiny.n_virtual == 2 and tiny.rx_spacing_wl == 0.5
    with pytest.raises(RadarConfigError):
        config_from_spec({"preset": "nope"})


def test_default_design_covers_the_second_order_ghosts():
    # 2nd-order ghost apparent range in this room reaches 8.49 m (ghost2 study)
    assert RadarConfig({}).max_range_m > 8.5


def test_every_scenario_loads_and_each_radar_gets_a_config():
    for path in ["scenarios/room_basic.yaml", "scenarios/room_people.yaml",
                 "scenarios/room_motion.yaml", "scenarios/room_radar.yaml"] + glob.glob("scenarios/generated/*.yaml"):
        cfg = load_scenario(path)
        for r in cfg["radars"]:
            assert radar_config_for(cfg, r).frame_time_s <= 1 / cfg["fps"]


def test_per_radar_override_in_the_scenario():
    cfg = load_scenario("scenarios/room_radar.yaml")
    west, east = (radar_config_for(cfg, r) for r in cfg["radars"])
    assert np.isclose(west.range_resolution_m, 0.15, atol=1e-3)
    assert np.isclose(east.range_resolution_m, 0.0375, atol=1e-4)
    assert east.n_chirps == 64 and west.n_chirps == 128


def test_scenario_rejects_a_frame_longer_than_the_scene_step():
    cfg = load_scenario("scenarios/room_basic.yaml")
    cfg["fps"] = 100                      # 10 ms < 15.36 ms frame
    with pytest.raises(ScenarioError, match="frame time"):
        _validate_radar_config(cfg)


def test_scenario_rejects_a_range_that_does_not_reach_the_room_corners():
    cfg = load_scenario("scenarios/room_basic.yaml")
    cfg["radar_config"] = {"n_samples": 16}        # 2.4 m max range
    with pytest.raises(ScenarioError, match="max range"):
        _validate_radar_config(cfg)
    cfg["radar_config"] = {"bandwidth_hz": -1}
    with pytest.raises(ScenarioError):
        _validate_radar_config(cfg)


def test_export_v4_carries_the_radar_config(tmp_path):
    cfg = load_scenario("scenarios/room_radar.yaml")
    meta = export_scene(cfg, str(tmp_path / "s.npz"))
    got = load_exported(str(tmp_path / "s.npz"))["meta"]
    assert meta["format_version"] == got["format_version"] == 4
    assert got["fps"] == 20
    east = got["radar_config"]["radar_east"]
    assert east["config"]["bandwidth_hz"] == 4e9
    assert np.isclose(east["limits"]["range_resolution_m"], 0.0375, atol=1e-4)
    # the stored config rebuilds the same limits
    assert RadarConfig(east["config"]).limits() == east["limits"]
