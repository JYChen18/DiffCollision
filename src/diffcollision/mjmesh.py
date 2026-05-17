import mujoco
import trimesh
from diffcollision.io import DCMesh
from diffcollision.utils import DCTensorSpec


def get_mesh_from_mjmodel(model: mujoco.MjModel, ts: DCTensorSpec = DCTensorSpec()):
    link_meshes_visual = {}
    link_meshes_collision = {}
    for idx in range(model.ngeom):
        geom_type = model.geom_type[idx]
        mesh = None
        if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
            mesh_id = model.geom_dataid[idx]
            if mesh_id >= 0:
                vert_adr = model.mesh_vertadr[mesh_id]
                vert_num = model.mesh_vertnum[mesh_id]
                face_adr = model.mesh_faceadr[mesh_id]
                face_num = model.mesh_facenum[mesh_id]
                verts = model.mesh_vert[vert_adr : vert_adr + vert_num].copy()
                faces = model.mesh_face[face_adr : face_adr + face_num].copy()
                mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            mesh = trimesh.creation.box(extents=model.geom_size[idx] * 2)
        elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            radius = model.geom_size[idx][0]
            height = model.geom_size[idx][1] * 2
            mesh = trimesh.creation.cylinder(radius=radius, height=height)
        elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
            mesh = trimesh.creation.icosphere(radius=model.geom_size[idx][0])
        elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
            radius = model.geom_size[idx][0]
            height = model.geom_size[idx][1] * 2
            mesh = trimesh.creation.capsule(radius=radius, height=height)

        if mesh is None:
            continue

        T_geom = trimesh.transformations.quaternion_matrix(model.geom_quat[idx])
        T_geom[:3, 3] = model.geom_pos[idx]
        mesh.apply_transform(T_geom)

        body_id = model.geom_bodyid[idx]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        contype = model.geom_contype[idx]
        conaffinity = model.geom_conaffinity[idx]
        is_visual = contype == 0 and conaffinity == 0
        if is_visual:
            link_meshes_visual.setdefault(name, []).append(mesh)
        else:
            link_meshes_collision.setdefault(name, []).append(mesh)

    for name, meshes in link_meshes_visual.items():
        link_meshes_visual[name] = trimesh.util.concatenate(meshes)

    link_to_dcmesh = {}
    for name in link_meshes_collision.keys():
        fm_lst = link_meshes_collision[name]
        if name in link_meshes_visual.keys():
            cm = link_meshes_visual[name]
        else:
            cm = trimesh.util.concatenate(fm_lst)
        link_to_dcmesh[name] = DCMesh.from_trimesh(cm, fm_lst, ts)

    return link_to_dcmesh


def get_exclude_pairs_from_mjmodel(model):
    exclude_pairs = []
    for sig in model.exclude_signature:
        sig = int(sig)
        body1 = sig >> 16
        body2 = sig & 0xFFFF

        name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body1)
        name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2)
        exclude_pairs.append((name1, name2))
        exclude_pairs.append((name2, name1))

    return exclude_pairs


def _body_name(model: mujoco.MjModel, body_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(body_id))


def _geom_pair_is_compatible(model: mujoco.MjModel, geom1: int, geom2: int) -> bool:
    contype1 = int(model.geom_contype[geom1])
    contype2 = int(model.geom_contype[geom2])
    conaffinity1 = int(model.geom_conaffinity[geom1])
    conaffinity2 = int(model.geom_conaffinity[geom2])
    return (contype1 & conaffinity2) != 0 or (contype2 & conaffinity1) != 0


def _add_collision_pair(
    collision_pair_margins: dict[tuple[int, int], float],
    body1: int,
    body2: int,
    margin: float,
) -> None:
    if body1 == body2:
        return
    pair = tuple(sorted((int(body1), int(body2))))
    collision_pair_margins[pair] = max(collision_pair_margins.get(pair, 0.0), margin)


def get_collision_pair_margins_from_mjmodel(model):
    """Return MuJoCo body-id collision pairs and their contact margins.

    DiffCollision stores one collision mesh per MuJoCo body, so this mirrors
    MuJoCo's geom-level contact filtering at body-pair granularity: a body pair
    is kept when at least one geom pair between those bodies can be checked. The
    returned margin is the max effective margin over geom pairs for that body
    pair, which is conservative for the aggregated body mesh representation.
    """
    body_to_geoms = {}
    for geom_id in range(model.ngeom):
        contype = int(model.geom_contype[geom_id])
        conaffinity = int(model.geom_conaffinity[geom_id])
        if contype == 0 and conaffinity == 0:
            continue
        body_id = int(model.geom_bodyid[geom_id])
        body_to_geoms.setdefault(body_id, []).append(geom_id)

    body_ids = sorted(body_to_geoms.keys())
    exclude_pairs = set(get_exclude_pairs_from_mjmodel(model))
    collision_pair_margins = {}
    explicit_pair_margins = {}
    explicit_geom_pairs = {
        tuple(sorted((int(model.pair_geom1[pair_id]), int(model.pair_geom2[pair_id]))))
        for pair_id in range(model.npair)
    }

    filter_parent = not (
        int(model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT)
    )

    for idx1, body1 in enumerate(body_ids):
        for body2 in body_ids[idx1 + 1 :]:
            name1 = _body_name(model, body1)
            name2 = _body_name(model, body2)
            if (name1, name2) in exclude_pairs:
                continue
            if int(model.body_weldid[body1]) == int(model.body_weldid[body2]):
                continue
            if filter_parent and (
                int(model.body_parentid[body1]) == body2
                or int(model.body_parentid[body2]) == body1
            ):
                continue
            for geom1 in body_to_geoms[body1]:
                for geom2 in body_to_geoms[body2]:
                    geom_pair = tuple(sorted((geom1, geom2)))
                    if geom_pair in explicit_geom_pairs:
                        continue
                    if not _geom_pair_is_compatible(model, geom1, geom2):
                        continue
                    margin = float(model.geom_margin[geom1] + model.geom_margin[geom2])
                    _add_collision_pair(
                        collision_pair_margins, body1, body2, margin
                    )

    for pair_id in range(model.npair):
        geom1 = int(model.pair_geom1[pair_id])
        geom2 = int(model.pair_geom2[pair_id])
        body1 = int(model.geom_bodyid[geom1])
        body2 = int(model.geom_bodyid[geom2])
        name1 = _body_name(model, body1)
        name2 = _body_name(model, body2)
        if (name1, name2) not in exclude_pairs:
            _add_collision_pair(
                explicit_pair_margins, body1, body2, float(model.pair_margin[pair_id])
            )

    for (body1, body2), margin in explicit_pair_margins.items():
        _add_collision_pair(collision_pair_margins, body1, body2, margin)
    return collision_pair_margins


def get_collision_pairs_from_mjmodel(model):
    return set(get_collision_pair_margins_from_mjmodel(model).keys())
