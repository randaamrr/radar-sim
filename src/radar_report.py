"""Step 25: print the limits of every radar in a scenario and compare presets.

python -m src.radar_report [scenario.yaml]   (also writes output/radar_limits.md)
"""
import sys

import numpy as np

from src.scenario import load_scenario
from src.radar_config import (PRESETS, RadarConfig, config_from_spec, radar_config_for,
                              farthest_in_room)
from src.surface import scene_samples
from src.ghosts import first_order_ghosts_view
from src.ghosts2 import second_order_ghosts_view

ROWS = [("f_center_ghz", "centre frequency (GHz)", "{:.2f}"),
        ("wavelength_mm", "wavelength (mm)", "{:.3f}"),
        ("slope_mhz_per_us", "chirp slope (MHz/us)", "{:.1f}"),
        ("sample_rate_mhz", "ADC sample rate (MHz)", "{:.2f}"),
        ("range_resolution_m", "range resolution (m)", "{:.4f}"),
        ("max_range_m", "max range (m)", "{:.2f}"),
        ("chirp_period_us", "chirp period per channel (us)", "{:.0f}"),
        ("frame_time_ms", "frame time (ms)", "{:.2f}"),
        ("velocity_resolution_mps", "velocity resolution (m/s)", "{:.3f}"),
        ("max_velocity_mps", "max velocity (m/s)", "{:.2f}"),
        ("n_virtual", "virtual channels", "{:d}"),
        ("angle_resolution_deg", "angle resolution at broadside (deg)", "{:.1f}"),
        ("max_angle_deg", "half field of view (deg)", "{:.0f}")]


def table(columns):
    """columns: {name: limits dict} -> markdown table."""
    names = list(columns)
    lines = ["| limit | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for key, label, fmt in ROWS:
        lines.append(f"| {label} | " + " | ".join(fmt.format(columns[n][key]) for n in names) + " |")
    return "\n".join(lines)


def coverage(cfg):
    """Farthest direct echo and farthest ghost apparent range per radar."""
    room = scene_samples(cfg)
    out = {}
    for r in cfg["radars"]:
        a, _ = first_order_ghosts_view(room, cfg, r["pos_m"], facing=False, occlusion=False)
        b, _ = second_order_ghosts_view(room, cfg, r["pos_m"], facing=False, occlusion=False)
        out[r["name"]] = {"farthest_corner_m": farthest_in_room(cfg, r["pos_m"]),
                          "ghost1_max_m": float(a.max()), "ghost2_max_m": float(b.max())}
    return out


def main(path="scenarios/room_basic.yaml"):
    cfg = load_scenario(path)
    per_radar = {r["name"]: radar_config_for(cfg, r).limits() for r in cfg["radars"]}
    text = [f"# Radar limits for {path}", "", f"scene fps = {cfg['fps']}", "", table(per_radar), ""]
    presets = {n: config_from_spec({"preset": n}).limits() for n in PRESETS}
    text += ["## Presets", "", table(presets), ""]
    cov = coverage(cfg)
    text += ["## Range coverage (direct echoes and ghosts, no occlusion)", "",
             "| radar | farthest corner (m) | 1st-order ghost max (m) | 2nd-order ghost max (m) | max range (m) | covers 2nd-order ghosts |",
             "|---|---|---|---|---|---|"]
    for r in cfg["radars"]:
        c, lim = cov[r["name"]], per_radar[r["name"]]
        ok = lim["max_range_m"] >= c["ghost2_max_m"]
        text.append(f"| {r['name']} | {c['farthest_corner_m']:.2f} | {c['ghost1_max_m']:.2f} | "
                    f"{c['ghost2_max_m']:.2f} | {lim['max_range_m']:.2f} | {'yes' if ok else 'NO'} |")
    out = "\n".join(text)
    print(out)
    with open("output/radar_limits.md", "w", encoding="utf-8") as f:
        f.write(out + "\n")


if __name__ == "__main__":
    main(*(sys.argv[1:2]))
