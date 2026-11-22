# Radar limits for scenarios\room_radar.yaml

scene fps = 20

| limit | radar_west | radar_east |
|---|---|---|
| centre frequency (GHz) | 60.50 | 62.00 |
| wavelength (mm) | 4.955 | 4.835 |
| chirp slope (MHz/us) | 25.0 | 100.0 |
| ADC sample rate (MHz) | 1.60 | 6.40 |
| range resolution (m) | 0.1499 | 0.0375 |
| max range (m) | 9.59 | 9.59 |
| chirp period per channel (us) | 120 | 120 |
| frame time (ms) | 15.36 | 7.68 |
| velocity resolution (m/s) | 0.161 | 0.315 |
| max velocity (m/s) | 10.32 | 10.07 |
| virtual channels | 8 | 8 |
| angle resolution at broadside (deg) | 14.3 | 14.3 |
| half field of view (deg) | 90 | 90 |

## Presets

| limit | baseline_1ghz | wide_4ghz | tiny |
|---|---|---|---|
| centre frequency (GHz) | 60.50 | 62.00 | 60.50 |
| wavelength (mm) | 4.955 | 4.835 | 4.955 |
| chirp slope (MHz/us) | 25.0 | 100.0 | 25.0 |
| ADC sample rate (MHz) | 1.60 | 6.40 | 1.60 |
| range resolution (m) | 0.1499 | 0.0375 | 0.1499 |
| max range (m) | 9.59 | 9.59 | 9.59 |
| chirp period per channel (us) | 120 | 120 | 60 |
| frame time (ms) | 15.36 | 15.36 | 1.92 |
| velocity resolution (m/s) | 0.161 | 0.157 | 1.290 |
| max velocity (m/s) | 10.32 | 10.07 | 20.65 |
| virtual channels | 8 | 8 | 2 |
| angle resolution at broadside (deg) | 14.3 | 14.3 | 57.3 |
| half field of view (deg) | 90 | 90 | 90 |

## Range coverage (direct echoes and ghosts, no occlusion)

| radar | farthest corner (m) | 1st-order ghost max (m) | 2nd-order ghost max (m) | max range (m) | covers 2nd-order ghosts |
|---|---|---|---|---|---|
| radar_west | 4.81 | 5.56 | 8.49 | 9.59 | yes |
| radar_east | 4.81 | 5.56 | 8.49 | 9.59 | yes |
