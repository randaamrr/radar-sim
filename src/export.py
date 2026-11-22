import json
import numpy as np

from src.surface import scene_samples
from src.meshes import occluder_scene
from src.people import cfg_people, CAPSULES
from src.radar_config import radar_config_for

FORMAT_VERSION = 4   # v2: normals + occluder; v3: people; v4: radar_config + limits in meta


def export_scene(cfg, path):
    """Write the shared scene file: surface samples, normals, radars, meshes."""
    room = scene_samples(cfg)
    radars = cfg["radars"]

    scene = occluder_scene(cfg)
    if scene is None:
        verts = np.zeros((0, 3))
        faces = np.zeros((0, 3), dtype=np.int64)
    else:
        verts = np.asarray(scene.vertices, dtype=float)
        faces = np.asarray(scene.faces, dtype=np.int64)

    furn = cfg["furniture"]

    # People: one merged mesh with a person index and a part index per face.
    part_names = [n for n, *_ in CAPSULES]
    pv, pf, f_person, f_part = [], [], [], []
    offset = 0
    for k, (_, parts) in enumerate(cfg_people(cfg)):
        for (part, m) in parts:
            pv.append(np.asarray(m.vertices, dtype=float))
            pf.append(np.asarray(m.faces, dtype=np.int64) + offset)
            f_person.append(np.full(len(m.faces), k, dtype=np.int64))
            f_part.append(np.full(len(m.faces), part_names.index(part), dtype=np.int64))
            offset += len(m.vertices)
    people = cfg.get("people", [])
    meta = {
        "format_version": FORMAT_VERSION,
        "seed": cfg["seed"],
        "room_size_m": cfg["room"]["size_m"],
        "grid_step_m": cfg["room"]["grid_step_m"],
        "n_points": int(len(room["points"])),
        "fps": cfg.get("fps", 20),
        "radar_config": {r["name"]: {"config": radar_config_for(cfg, r).to_dict(),
                                      "limits": radar_config_for(cfg, r).limits()}
                         for r in radars},
    }
    np.savez_compressed(
        path,
        points=room["points"],
        normal=room["normal"],
        area=room["area"],
        reflectance=room["reflectance"],
        strength=room["strength"],
        material=room["material"],
        surface=room["surface"],
        radar_names=np.array([r["name"] for r in radars], dtype=str),
        radar_pos_m=np.array([r["pos_m"] for r in radars], dtype=float).reshape(-1, 3),
        radar_yaw_deg=np.array([r["yaw_deg"] for r in radars], dtype=float),
        radar_pitch_deg=np.array([r["pitch_deg"] for r in radars], dtype=float),
        furniture_names=np.array([f["name"] for f in furn], dtype=str),
        furniture_materials=np.array([f["material"] for f in furn], dtype=str),
        people_names=np.array([p["name"] for p in people], dtype=str),
        people_poses=np.array([p["pose"] for p in people], dtype=str),
        people_height_m=np.array([p["height_m"] for p in people], dtype=float),
        people_pos_m=np.array([p["pos_m"] for p in people], dtype=float).reshape(-1, 2),
        people_yaw_deg=np.array([p["yaw_deg"] for p in people], dtype=float),
        people_support_z=np.array([p["support_z"] for p in people], dtype=float),
        people_vertices=np.vstack(pv) if pv else np.zeros((0, 3)),
        people_faces=np.vstack(pf) if pf else np.zeros((0, 3), dtype=np.int64),
        people_face_person=np.concatenate(f_person) if f_person else np.zeros(0, dtype=np.int64),
        people_face_part=np.concatenate(f_part) if f_part else np.zeros(0, dtype=np.int64),
        part_names=np.array(part_names, dtype=str),
        occluder_vertices=verts,
        occluder_faces=faces,
        meta=json.dumps(meta),
    )
    return meta


def load_exported(path):
    data = np.load(path, allow_pickle=False)
    out = {k: data[k] for k in data.files if k != "meta"}
    out["meta"] = json.loads(str(data["meta"]))
    return out