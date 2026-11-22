"""Step 25: radar configuration and the limits it implies.

Everything the signal simulator needs comes from one dict (`radar_config:`
in the scenario, optionally overridden per radar). The limits below are
closed-form FMCW relations; step 26 must reproduce them with a simulated
target (that is the check that the simulator obeys its own config).

Conventions and assumptions (also written in decision_note_radar.md):
  * f0_hz is the START frequency of the sweep; the centre frequency
    f0 + B/2 defines the wavelength.
  * The ADC samples the whole chirp (n_samples over chirp_duration_s). Real
    chips sample only the linear part of the ramp, which lowers the usable B.
  * adc_mode "complex" (I/Q) by default; "real" halves the maximum range.
  * Several TX use time-division MIMO: TX fire one after another, so the
    chirp period seen by one virtual channel is n_tx * (chirp + idle).
  * The virtual array is a uniform linear array along azimuth with spacing
    rx_spacing_wl (TX spaced n_rx * rx_spacing_wl apart).
  * Angle resolution is the common approximation lambda / (N_virtual * d).
"""
import copy
import numpy as np

C = 299_792_458.0   # speed of light in vacuum, m/s

# Reference design (an assumption, to be swept in step 31).
DEFAULTS = {
    "f0_hz": 60.0e9,              # 60 GHz band; skin reflectance was computed at 60 GHz
    "bandwidth_hz": 1.0e9,        # range resolution 15 cm = the room-track assumption
    "chirp_duration_s": 40e-6,
    "idle_s": 20e-6,
    "n_chirps": 128,              # per TX, per frame
    "n_samples": 64,              # per chirp: unambiguous range 9.6 m > 8.5 m (2nd-order ghosts)
    "adc_mode": "complex",
    "array": {"n_tx": 2, "n_rx": 4, "rx_spacing_wl": 0.5},
}

PRESETS = {
    "baseline_1ghz": {},
    "wide_4ghz": {"bandwidth_hz": 4.0e9, "n_samples": 256},      # 3.75 cm resolution
    "tiny": {"bandwidth_hz": 1.0e9, "n_chirps": 32, "array": {"n_tx": 1, "n_rx": 2}},
}


class RadarConfigError(ValueError):
    pass


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


class RadarConfig:
    """Validated radar configuration with derived limits as properties."""

    FIELDS = ("f0_hz", "bandwidth_hz", "chirp_duration_s", "idle_s", "n_chirps",
              "n_samples", "adc_mode", "array")

    def __init__(self, d):
        d = _merge(DEFAULTS, d)
        unknown = set(d) - set(self.FIELDS)
        if unknown:
            raise RadarConfigError(f"unknown radar_config fields: {sorted(unknown)}")
        arr = d["array"]
        unknown = set(arr) - {"n_tx", "n_rx", "rx_spacing_wl"}
        if unknown:
            raise RadarConfigError(f"unknown array fields: {sorted(unknown)}")
        self.f0_hz = float(d["f0_hz"])
        self.bandwidth_hz = float(d["bandwidth_hz"])
        self.chirp_duration_s = float(d["chirp_duration_s"])
        self.idle_s = float(d["idle_s"])
        self.n_chirps = d["n_chirps"]
        self.n_samples = d["n_samples"]
        self.adc_mode = d["adc_mode"]
        self.n_tx, self.n_rx = arr["n_tx"], arr["n_rx"]
        self.rx_spacing_wl = float(arr["rx_spacing_wl"])
        self._validate()

    def _validate(self):
        for name in ("n_chirps", "n_samples", "n_tx", "n_rx"):
            v = getattr(self, name)
            if not isinstance(v, (int, np.integer)) or isinstance(v, bool) or v < 1:
                raise RadarConfigError(f"{name} must be a positive integer")
        if self.n_samples < 8 or self.n_chirps < 2:
            raise RadarConfigError("need n_samples >= 8 and n_chirps >= 2")
        if not 1e9 <= self.f0_hz <= 300e9:
            raise RadarConfigError("f0_hz must be in [1 GHz, 300 GHz]")
        if not 0 < self.bandwidth_hz <= self.f0_hz:
            raise RadarConfigError("bandwidth_hz must be in (0, f0_hz]")
        if self.chirp_duration_s <= 0 or self.idle_s < 0:
            raise RadarConfigError("chirp_duration_s must be > 0 and idle_s >= 0")
        if self.adc_mode not in ("complex", "real"):
            raise RadarConfigError("adc_mode must be 'complex' or 'real'")
        if not 0.1 <= self.rx_spacing_wl <= 2.0:
            raise RadarConfigError("rx_spacing_wl must be in [0.1, 2.0] wavelengths")

    # ---- derived quantities -------------------------------------------------
    @property
    def f_center_hz(self):
        return self.f0_hz + self.bandwidth_hz / 2.0

    @property
    def wavelength_m(self):
        return C / self.f_center_hz

    @property
    def slope_hz_per_s(self):
        return self.bandwidth_hz / self.chirp_duration_s

    @property
    def sample_rate_hz(self):
        return self.n_samples / self.chirp_duration_s

    @property
    def range_resolution_m(self):
        return C / (2.0 * self.bandwidth_hz)

    @property
    def max_range_m(self):
        r = self.sample_rate_hz * C / (2.0 * self.slope_hz_per_s)
        return r if self.adc_mode == "complex" else r / 2.0

    @property
    def chirp_period_s(self):
        """Time between two chirps of the same virtual channel (TDM-MIMO)."""
        return self.n_tx * (self.chirp_duration_s + self.idle_s)

    @property
    def frame_time_s(self):
        return self.n_chirps * self.chirp_period_s

    @property
    def velocity_resolution_mps(self):
        return self.wavelength_m / (2.0 * self.n_chirps * self.chirp_period_s)

    @property
    def max_velocity_mps(self):
        return self.wavelength_m / (4.0 * self.chirp_period_s)

    @property
    def n_virtual(self):
        return self.n_tx * self.n_rx

    @property
    def angle_resolution_deg(self):
        """Broadside resolution lambda / (N_virtual * d), in degrees."""
        return float(np.degrees(1.0 / (self.n_virtual * self.rx_spacing_wl)))

    @property
    def max_angle_deg(self):
        """Half field of view before grating lobes: asin(1 / (2 d))."""
        s = 1.0 / (2.0 * self.rx_spacing_wl)
        return 90.0 if s >= 1.0 else float(np.degrees(np.arcsin(s)))

    @property
    def beat_frequency_max_hz(self):
        """Beat frequency of a target at max_range_m."""
        return 2.0 * self.slope_hz_per_s * self.max_range_m / C

    def limits(self):
        return {
            "f_center_ghz": self.f_center_hz / 1e9,
            "wavelength_mm": self.wavelength_m * 1e3,
            "slope_mhz_per_us": self.slope_hz_per_s / 1e12,
            "sample_rate_mhz": self.sample_rate_hz / 1e6,
            "range_resolution_m": self.range_resolution_m,
            "max_range_m": self.max_range_m,
            "chirp_period_us": self.chirp_period_s * 1e6,
            "frame_time_ms": self.frame_time_s * 1e3,
            "velocity_resolution_mps": self.velocity_resolution_mps,
            "max_velocity_mps": self.max_velocity_mps,
            "n_virtual": self.n_virtual,
            "angle_resolution_deg": self.angle_resolution_deg,
            "max_angle_deg": self.max_angle_deg,
        }

    def to_dict(self):
        return {"f0_hz": self.f0_hz, "bandwidth_hz": self.bandwidth_hz,
                "chirp_duration_s": self.chirp_duration_s, "idle_s": self.idle_s,
                "n_chirps": int(self.n_chirps), "n_samples": int(self.n_samples),
                "adc_mode": self.adc_mode,
                "array": {"n_tx": int(self.n_tx), "n_rx": int(self.n_rx),
                          "rx_spacing_wl": self.rx_spacing_wl}}


def config_from_spec(spec):
    """RadarConfig from a user section: optional `preset`, then overrides."""
    spec = dict(spec or {})
    preset = spec.pop("preset", "baseline_1ghz")
    if preset not in PRESETS:
        raise RadarConfigError(f"unknown preset: {preset} (known: {sorted(PRESETS)})")
    return RadarConfig(_merge(PRESETS[preset], spec))


def radar_config_for(cfg, radar):
    """Effective RadarConfig of one radar: scenario section, then its own override."""
    spec = _merge(cfg.get("radar_config") or {}, radar.get("radar_config"))
    return config_from_spec(spec)


def farthest_in_room(cfg, pos):
    L, W, H = cfg["room"]["size_m"]
    corners = np.array([[x, y, z] for x in (0, L) for y in (0, W) for z in (0, H)])
    return float(np.linalg.norm(corners - np.asarray(pos, dtype=float), axis=1).max())


def check_scenario(cfg):
    """Raise RadarConfigError if a radar's config contradicts the scenario.

    * the frame (all chirps) must fit in one scene step 1/fps;
    * the maximum range must cover the farthest corner of the room
      (otherwise direct echoes alias).
    Returns {radar name: RadarConfig}.
    """
    out = {}
    for r in cfg["radars"]:
        rc = radar_config_for(cfg, r)
        if rc.frame_time_s > 1.0 / cfg["fps"]:
            raise RadarConfigError(
                f"{r['name']}: frame time {rc.frame_time_s * 1e3:.2f} ms does not fit "
                f"in 1/fps = {1e3 / cfg['fps']:.2f} ms")
        far = farthest_in_room(cfg, r["pos_m"])
        if rc.max_range_m < far:
            raise RadarConfigError(
                f"{r['name']}: max range {rc.max_range_m:.2f} m is below the farthest "
                f"room corner ({far:.2f} m)")
        out[r["name"]] = rc
    return out
