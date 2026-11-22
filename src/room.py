import numpy as np


def _plane_points(origin, u_vec, v_vec, step):
    u_len = np.linalg.norm(u_vec)
    v_len = np.linalg.norm(v_vec)
    nu = max(1, round(u_len / step))
    nv = max(1, round(v_len / step))
    u_dir = np.array(u_vec) / u_len
    v_dir = np.array(v_vec) / v_len
    du = u_len / nu
    dv = v_len / nv
    iu, iv = np.meshgrid(np.arange(nu), np.arange(nv), indexing="ij")
    pts = (
        np.array(origin)
        + ((iu.ravel() + 0.5) * du)[:, None] * u_dir
        + ((iv.ravel() + 0.5) * dv)[:, None] * v_dir
    )
    return pts, du * dv


def build_shell(cfg):
    L, W, H = cfg["room"]["size_m"]
    step = cfg["room"]["grid_step_m"]
    surf = cfg["surfaces"]

    planes = {
        "floor": ((0, 0, 0), (L, 0, 0), (0, W, 0), surf["floor"]),
        "ceiling": ((0, 0, H), (L, 0, 0), (0, W, 0), surf["ceiling"]),
        "wall_x0": ((0, 0, 0), (0, W, 0), (0, 0, H), surf["walls"]),
        "wall_x1": ((L, 0, 0), (0, W, 0), (0, 0, H), surf["walls"]),
        "wall_y0": ((0, 0, 0), (L, 0, 0), (0, 0, H), surf["walls"]),
        "wall_y1": ((0, W, 0), (L, 0, 0), (0, 0, H), surf["walls"]),
    }

    all_pts, all_area, all_mat, all_name = [], [], [], []
    for name, (origin, u, v, mat) in planes.items():
        pts, area = _plane_points(origin, u, v, step)
        all_pts.append(pts)
        all_area.append(np.full(len(pts), area))
        all_mat.extend([mat] * len(pts))
        all_name.extend([name] * len(pts))

    return {
        "points": np.vstack(all_pts),
        "area": np.concatenate(all_area),
        "material": np.array(all_mat),
        "surface": np.array(all_name),
    }


def build_furniture(cfg):
    step = cfg["room"]["grid_step_m"]

    all_pts, all_area, all_mat, all_name = [], [], [], []
    for item in cfg["furniture"]:
        x, y = item["pos_m"]
        w, d, h = item["size_m"]
        mat = item["material"]
        name = item["name"]

        faces = {
            "top": ((x, y, h), (w, 0, 0), (0, d, 0)),
            "front": ((x, y, 0), (w, 0, 0), (0, 0, h)),
            "back": ((x, y + d, 0), (w, 0, 0), (0, 0, h)),
            "left": ((x, y, 0), (0, d, 0), (0, 0, h)),
            "right": ((x + w, y, 0), (0, d, 0), (0, 0, h)),
        }
        for face, (origin, u, v) in faces.items():
            pts, area = _plane_points(origin, u, v, step)
            all_pts.append(pts)
            all_area.append(np.full(len(pts), area))
            all_mat.extend([mat] * len(pts))
            all_name.extend([f"{name}_{face}"] * len(pts))

    if not all_pts:
        return {
            "points": np.zeros((0, 3)),
            "area": np.zeros(0),
            "material": np.array([], dtype=str),
            "surface": np.array([], dtype=str),
        }

    return {
        "points": np.vstack(all_pts),
        "area": np.concatenate(all_area),
        "material": np.array(all_mat),
        "surface": np.array(all_name),
    }


def build_room(cfg):
    shell = build_shell(cfg)
    furn = build_furniture(cfg)
    return {k: np.concatenate([shell[k], furn[k]]) for k in shell}