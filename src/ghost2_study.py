import copy

import numpy as np

from src.scenario import load_scenario
from src.surface import scene_samples, radar_view, direct_power
from src.ghosts import first_order_ghosts_view
from src.ghosts2 import second_order_ghosts_view

if __name__ == "__main__":
    cfg = load_scenario("scenarios/room_basic.yaml")
    cfg = copy.deepcopy(cfg)
    cfg["room"]["grid_step_m"] = 0.2      # ratios are integrals: coarse grid is enough
    room = scene_samples(cfg)
    db = lambda a, b: 10 * np.log10(a / b)

    print("ghost/direct total power (grid step 0.2 m)")
    for r in cfg["radars"]:
        pos = r["pos_m"]
        d_ang = direct_power(radar_view(room, cfg, pos, occlusion=False), pos)
        d_all = direct_power(radar_view(room, cfg, pos), pos)
        f_ang = first_order_ghosts_view(room, cfg, pos, occlusion=False)[1].sum()
        f_all = first_order_ghosts_view(room, cfg, pos)[1].sum()
        s_ang = second_order_ghosts_view(room, cfg, pos, occlusion=False)[1].sum()
        s_all = second_order_ghosts_view(room, cfg, pos)[1].sum()
        print(f"{r['name']}:")
        print(f"    1st order: angle {db(f_ang, d_ang):.1f} dB | angle+occlusion {db(f_all, d_all):.1f} dB")
        print(f"    2nd order: angle {db(s_ang, d_ang):.1f} dB | angle+occlusion {db(s_all, d_all):.1f} dB")
        print(f"    2nd vs 1st (angle+occlusion): {db(s_all, f_all):.1f} dB")