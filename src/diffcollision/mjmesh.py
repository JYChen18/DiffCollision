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
