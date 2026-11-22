# Decision note: Room representation (Marwan track, Phase 1)

## Question
How should the room and objects be represented for radar simulation:
voxel/point grid, or meshes (Blender)? How should wall multipath be included?

## Options
1. Point-scatterer grid sampled on room and furniture surfaces (NumPy)
2. Meshes (Open3D / Blender) sampled to points at runtime
3. Multipath: none / image method (1st, 2nd order) / ray tracing

## Criteria
Accuracy of the range profile at the radar's range resolution,
number of points (cost), reproducibility from a YAML file, independence
from heavy tools.

## Choice
- Surface point grid, default step 0.1 m, generated in pure NumPy.
- Multipath by the image method, first order in the default pipeline.
- Blender is not used in Phase 1.

## Evidence
Grid convergence (error vs a 0.025 m reference, range profile smoothed
to the radar resolution, mean over 2 radars):

| step (m) | points | err @15 cm | err @5 cm |
|---|---|---|---|
| 0.4  | 500   | 0.840 | 1.025 |
| 0.2  | 1969  | 0.497 | 1.147 |
| 0.1  | 7856  | 0.035 | 0.460 |
| 0.05 | 31360 | 0.020 | 0.086 |

At 15 cm range resolution (1 GHz bandwidth) step 0.1 m gives 3.5% error.
At 5 cm resolution it needs 0.05 m (8.6%). Step is therefore tied to the
radar resolution: roughly equal to the range resolution gives <10% error
at both tested points (to be confirmed with a third resolution).

Multipath (image method, 2 radars on opposite walls):
- 1st order ghost/direct total power: -18.8 dB (west), -18.7 dB (east)
- 2nd order (approximate lower bound): -24.5 dB for both
- Ghosts appear as discrete peaks; at about 1.5, 2.3 and 4.1 m apparent
  range the ghost peak exceeds the direct return.
Conclusion: multipath cannot be ignored; image method is the cheapest
sufficient model; ray tracing is not yet justified.

Reflectance: Fresnel normal-incidence power reflectance from relative
permittivity (ITU-R P.2040 values, to be verified against the document).

## Limitations
- The reference is a finer grid, not a real mesh. Valid for flat
  surfaces; must be revisited for curved bodies (people).
- Normal incidence, lossless, no angle dependence in reflectance.
- Multipath ignores occlusion (Yasmine's track); 2nd order approximate.
- Minimum range clamp of 0.1 m affects near-radar points (blind zone).
- Results are relative comparisons inside the simulator, not real-room
  accuracy.

## Revisit if
- Meshes are needed for people/curved objects.
- Yasmine's ray-tracing check shows image-method error above target.
- A different bandwidth is chosen (re-derive the step from resolution).