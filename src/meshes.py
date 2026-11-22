import numpy as np
import trimesh


def room_shell_mesh(cfg):
    """Room as an inward-facing box (walls, floor, ceiling)."""
    L, W, H = cfg["room"]["size_m"]
    box = trimesh.creation.box(extents=[L, W, H])
    box.apply_translation([L / 2, W / 2, H / 2])
    box.invert()
    return box


def furniture_meshes(cfg):
    """One box mesh per furniture item: list of (name, material, mesh)."""
    out = []
    for it in cfg["furniture"]:
        x, y = it["pos_m"]
        w, d, h = it["size_m"]
        m = trimesh.creation.box(extents=[w, d, h])
        m.apply_translation([x + w / 2, y + d / 2, h / 2])
        out.append((it["name"], it["material"], m))
    return out


def occluder_scene(cfg):
    """Furniture and people merged into one mesh (the room shell never
    blocks a line of sight between two points inside the room)."""
    from src.people import cfg_people
    meshes = [m for _, _, m in furniture_meshes(cfg)]
    for _, parts in cfg_people(cfg):
        meshes.extend(m for _, m in parts)
    if not meshes:
        return None
    return trimesh.util.concatenate(meshes)


RAY_BATCH = 500    # rays per ray-casting call; keeps peak memory small on any PC


def blocked_segments(scene, starts, ends, eps=1e-3):
    """True where the segment start->end hits the scene before reaching its end.

    Rays are cast in batches of RAY_BATCH: one call for thousands of rays
    against a mesh with many triangles can need hundreds of MB.
    """
    starts = np.asarray(starts, dtype=float)
    ends = np.asarray(ends, dtype=float)
    blocked = np.zeros(len(starts), dtype=bool)
    if scene is None or len(starts) == 0:
        return blocked
    vec = ends - starts
    dist = np.linalg.norm(vec, axis=1)
    dirs = vec / np.maximum(dist, 1e-9)[:, None]
    for lo in range(0, len(starts), RAY_BATCH):
        hi = lo + RAY_BATCH
        loc, idx_ray, _ = scene.ray.intersects_location(
            starts[lo:hi], dirs[lo:hi], multiple_hits=True)
        if len(idx_ray) == 0:
            continue
        hit_d = np.linalg.norm(loc - starts[lo:hi][idx_ray], axis=1)
        blocked[lo + idx_ray[hit_d < dist[lo:hi][idx_ray] - eps]] = True
    return blocked


def visible(scene, origin, targets, eps=1e-3):
    """True for each target point the radar at `origin` can see directly."""
    targets = np.asarray(targets, dtype=float)
    if scene is None:
        return np.ones(len(targets), dtype=bool)
    origins = np.tile(np.asarray(origin, dtype=float), (len(targets), 1))
    return ~blocked_segments(scene, origins, targets, eps)
