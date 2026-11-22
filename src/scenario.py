import yaml
import numpy as np

KNOWN_MATERIALS = {"concrete", "wood", "glass", "metal", "plasterboard"}


class ScenarioError(ValueError):
    pass


def load_scenario(path):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ScenarioError("ملف السيناريو فارغ أو غير صالح")

    for key in ("seed", "room", "surfaces", "furniture"):
        if key not in cfg:
            raise ScenarioError(f"الحقل ناقص: {key}")

    room = cfg["room"]
    surfaces = cfg["surfaces"]
    furniture = cfg["furniture"]

    if not isinstance(room, dict) or not isinstance(surfaces, dict) or not isinstance(furniture, list):
        raise ScenarioError("نوع البيانات في room أو surfaces أو furniture غير صحيح")

    size = room.get("size_m")
    step = room.get("grid_step_m")
    if not isinstance(size, list) or len(size) != 3 or any(s <= 0 for s in size):
        raise ScenarioError("size_m لازم 3 أرقام موجبة")
    if step is None or step <= 0:
        raise ScenarioError("grid_step_m لازم يكون موجب")

    for surf, mat in surfaces.items():
        if mat not in KNOWN_MATERIALS:
            raise ScenarioError(f"خامة مجهولة في {surf}: {mat}")

    for item in furniture:
        x, y = item["pos_m"]
        w, d, h = item["size_m"]
        if item["material"] not in KNOWN_MATERIALS:
            raise ScenarioError(f"خامة مجهولة في {item['name']}")
        if x < 0 or y < 0 or x + w > size[0] or y + d > size[1] or h > size[2]:
            raise ScenarioError(f"{item['name']} طالع بره الغرفة")

    cfg.setdefault("fps", 20)          # frames per second (assumption: radar frame rate)
    if not isinstance(cfg["fps"], (int, float)) or not 1 <= cfg["fps"] <= 200:
        raise ScenarioError("fps must be a number in [1, 200]")

    cfg.setdefault("radars", [])
    names = set()
    for r in cfg["radars"]:
        for key in ("name", "pos_m", "yaw_deg", "pitch_deg"):
            if key not in r:
                raise ScenarioError(f"radar field missing: {key}")
        if r["name"] in names:
            raise ScenarioError(f"duplicate radar name: {r['name']}")
        names.add(r["name"])
        x, y, z = r["pos_m"]
        if not (0 <= x <= size[0] and 0 <= y <= size[1] and 0 <= z <= size[2]):
            raise ScenarioError(f"{r['name']} is outside the room")
        if not -180 <= r["yaw_deg"] <= 180:
            raise ScenarioError(f"{r['name']}: yaw_deg must be in [-180, 180]")
        if not -90 <= r["pitch_deg"] <= 90:
            raise ScenarioError(f"{r['name']}: pitch_deg must be in [-90, 90]")

    _validate_people(cfg)
    _validate_radar_config(cfg)
    return cfg


def _validate_radar_config(cfg):
    from src.radar_config import RadarConfigError, check_scenario, config_from_spec
    try:
        config_from_spec(cfg.get("radar_config"))
        check_scenario(cfg)
    except RadarConfigError as e:
        raise ScenarioError(str(e))


def _validate_people(cfg):
    from src.people import (POSES, HEIGHT_RANGE_M, person_meshes,
                            validate_placement, check_support, bodies_overlap)
    cfg.setdefault("people", [])
    if not isinstance(cfg["people"], list):
        raise ScenarioError("people must be a list")
    names, bodies = set(), []
    for p in cfg["people"]:
        for key in ("name", "pose", "height_m", "pos_m"):
            if key not in p:
                raise ScenarioError(f"person field missing: {key}")
        name = p["name"]
        if name in names:
            raise ScenarioError(f"duplicate person name: {name}")
        names.add(name)
        p.setdefault("yaw_deg", 0.0)
        p.setdefault("support_z", 0.0)
        if p["pose"] not in POSES:
            raise ScenarioError(f"{name}: pose must be one of {POSES}")
        lo, hi = HEIGHT_RANGE_M
        if not lo <= p["height_m"] <= hi:
            raise ScenarioError(f"{name}: height_m must be in [{lo}, {hi}]")
        if not -180 <= p["yaw_deg"] <= 180:
            raise ScenarioError(f"{name}: yaw_deg must be in [-180, 180]")
        x, y = p["pos_m"]
        parts = person_meshes(p["pose"], p["height_m"], x, y,
                              p["yaw_deg"], p["support_z"])
        problems = validate_placement(cfg, parts)
        if not check_support(cfg, x, y, p["support_z"]):
            problems.append("nothing supports the body at support_z")
        for other, other_parts in bodies:
            if bodies_overlap(parts, other_parts):
                problems.append(f"body overlaps {other}")
        if problems:
            raise ScenarioError(f"{name}: " + "; ".join(problems))
        bodies.append((name, parts))
        if "motion" in p:
            _validate_motion(cfg, p, len(bodies) - 1)


def _validate_motion(cfg, person, index):
    from src.motion_library import KINDS, person_sequence
    from src.motion import validate_sequence
    motion = person["motion"]
    if not isinstance(motion, dict) or motion.get("kind") not in KINDS:
        raise ScenarioError(f"{person['name']}: motion.kind must be one of {KINDS}")
    try:
        seq = person_sequence(cfg, person, index)
    except ValueError as e:
        raise ScenarioError(f"{person['name']}: {e}")
    problems = validate_sequence(cfg, seq)
    if problems:
        raise ScenarioError(f"{person['name']}: " + "; ".join(problems[:3]))


def radar_direction(yaw_deg, pitch_deg):
    """Unit boresight vector. yaw 0 -> +x, yaw 90 -> +y, negative pitch -> down."""
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)
    return np.array([
        np.cos(pitch) * np.cos(yaw),
        np.cos(pitch) * np.sin(yaw),
        np.sin(pitch),
    ])
