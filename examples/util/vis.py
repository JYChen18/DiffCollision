# Standard Library
from dataclasses import dataclass
import os
import sys
import logging

# Third Party
import numpy as np
import trimesh
import torch

from .rotation import torch_quaternion_rotate_points, torch_matrix_to_quaternion
from diffcollision import DCDebugDict

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


@dataclass
class Material:
    color: list
    name: str = "1"
    metallic: float = 0.0
    roughness: float = 0.5


class UsdStage:
    def __init__(
        self,
        name: str = "curobo_stage.usd",
        base_frame: str = "/world",
        timestep: int | None = None,
        dt: float = 0.02,
        interp_step: float = 1,
    ):
        self.stage = Usd.Stage.CreateNew(name)
        UsdGeom.SetStageUpAxis(self.stage, "Z")
        UsdGeom.SetStageMetersPerUnit(self.stage, 1)
        UsdPhysics.SetStageKilogramsPerUnit(self.stage, 1)
        xform = self.stage.DefinePrim(base_frame, "Xform")
        self.stage.SetDefaultPrim(xform)
        self.dt = dt
        self.interp_step = interp_step
        if timestep is not None:
            self.stage.SetStartTimeCode(0)
            self.stage.SetEndTimeCode((timestep - 1) * self.interp_step)
            self.stage.SetTimeCodesPerSecond((24))

    def add_subroot(self, root: str = "/world", sub_root: str = "obstacles"):
        xform = self.stage.DefinePrim(os.path.join(root, sub_root), "Xform")
        UsdGeom.Xformable(xform).AddScaleOp()
        xform.GetAttribute("xformOp:scale").Set(Gf.Vec3f([100.0, 100.0, 100.0]))
        UsdGeom.Xformable(xform).AddTranslateOp()
        xform.GetAttribute("xformOp:translate").Set(Gf.Vec3f([100.0, 100.0, 100.0]))

    def add_mesh_lst(
        self,
        mesh_lst: list[trimesh.Trimesh],
        name_lst: list[str],
        pose_lst: list[list],
        visible_time_lst: list[tuple[float, float]],
        material_lst: Material | None = None,
        base_frame: str = "/world",
        obstacles_frame: str = "obstacles",
    ):
        self.add_subroot(base_frame, obstacles_frame)
        full_path = os.path.join(base_frame, obstacles_frame)

        prim_path = [
            self.add_mesh(m, name, full_path, visible_time=vt, material=material)
            for vt, name, m, material in zip(
                visible_time_lst, name_lst, mesh_lst, material_lst
            )
        ]

        for i, i_val in enumerate(prim_path):
            curr_prim = self.stage.GetPrimAtPath(i_val)
            form = UsdGeom.Xformable(curr_prim).GetOrderedXformOps()

            for t, p in enumerate(pose_lst[i]):
                position = Gf.Vec3f(p[0], p[1], p[2])
                quat = Gf.Quatf(p[3], *p[4:-1])
                scale = Gf.Vec3f(p[-1], p[-1], p[-1])
                real_t = (t + visible_time_lst[i][0]) * self.interp_step
                form[0].Set(time=real_t, value=position)
                form[1].Set(time=real_t, value=quat)
                form[2].Set(time=real_t, value=scale)
        return

    def add_mesh(
        self,
        mesh: trimesh.Trimesh,
        mesh_name: str,
        base_frame: str = "/world/obstacles",
        visible_time: tuple[float, float] | None = None,
        material: Material | None = None,
    ):
        root_path = os.path.join(
            base_frame,
            "o"
            + mesh_name.replace(".", "_")
            .replace("é", "e")
            .replace("+", "_")
            .replace(":", "_")
            .replace("-", "_"),
        )
        obj_geom = UsdGeom.Mesh.Get(self.stage, root_path)
        if not obj_geom:
            obj_geom = UsdGeom.Mesh.Define(self.stage, root_path)
            verts, faces = mesh.vertices, mesh.faces
            obj_geom.CreatePointsAttr(verts)
            obj_geom.CreateFaceVertexCountsAttr([3 for _ in range(len(faces))])
            obj_geom.CreateFaceVertexIndicesAttr(np.ravel(faces).tolist())
            obj_geom.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
            a = UsdGeom.Xformable(obj_geom)  #
            a.AddTranslateOp()
            a.AddOrientOp()
            a.AddScaleOp()
            obj_geom.GetVisibilityAttr().Set("invisible", Usd.TimeCode(0))

        if visible_time is not None:
            obj_geom.GetVisibilityAttr().Set("inherited", Usd.TimeCode(visible_time[0]))
            obj_geom.GetVisibilityAttr().Set("invisible", Usd.TimeCode(visible_time[1]))

        if material is not None:
            obj_prim = self.stage.GetPrimAtPath(root_path)
            self.add_material(
                "material_" + material.name,
                root_path,
                material.color,
                obj_prim,
                material.roughness,
                material.metallic,
            )

        return root_path

    def write_stage_to_file(self, file_path: str, flatten: bool = False):
        if flatten:
            usd_str = self.stage.Flatten().ExportToString()
        else:
            usd_str = self.stage.GetRootLayer().ExportToString()
        with open(file_path, "w") as f:
            f.write(usd_str)

    def add_material(
        self,
        name: str,
        obj_path: str,
        color: list[float],
        obj_prim: Usd.Prim,
        roughness: float,
        metallic: float,
    ):
        mat_path = os.path.join(obj_path, name)
        material_usd = UsdShade.Material.Define(self.stage, mat_path)
        pbrShader = UsdShade.Shader.Define(
            self.stage, os.path.join(mat_path, "PbrShader")
        )
        pbrShader.CreateIdAttr("UsdPreviewSurface")
        pbrShader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        pbrShader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
        pbrShader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(color[:3])
        )
        pbrShader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(color[:3])
        )
        pbrShader.CreateInput("baseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(color[:3])
        )

        pbrShader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(color[3])
        material_usd.CreateSurfaceOutput().ConnectToSource(
            pbrShader.ConnectableAPI(), "surface"
        )
        obj_prim.GetPrim().ApplyAPI(UsdShade.MaterialBindingAPI)
        UsdShade.MaterialBindingAPI(obj_prim).Bind(material_usd)
        return material_usd

    def save(self):
        self.stage.Save()


MATERIAL_CLS = {
    "blue": Material(color=(0.55, 0.75, 0.95, 0.8), name="blue"),
    "orange": Material(color=(0.99, 0.75, 0.52, 0.8), name="orange"),
    "green": Material(color=(0, 1, 0, 1.0), name="green"),
    "red": Material(color=(1, 0.0, 0.0, 1.0), name="red"),
    "purple": Material(color=(0.75, 0.65, 0.85, 1.0), name="purple"),
}


def vis_usd(
    vis_dict: DCDebugDict,
    vis_ids: list,
    save_folder: str,
    name2material: dict = {
        "mesh1": "blue",
        "mesh2": "orange",
        "tp1": "green",
        "tp2": "green",
        "wp1": "red",
        "wp2": "red",
    },
):
    data_length = len(vis_dict.wp1)
    tm_lst = [m.coarse_mesh for m in vis_dict.meshes]
    for k, v in vis_dict.__dict__.items():
        if "mesh" not in k and len(v) > 0:
            setattr(vis_dict, k, torch.stack(v, dim=0))

    q = torch_matrix_to_quaternion(vis_dict.transforms[..., :3, :3])
    t = vis_dict.transforms[..., :3, 3]
    n_t, _, n_m = q.shape[:3]
    assert n_m == len(vis_dict.meshes)

    scale = max([np.linalg.norm(tm.bounds[0] - tm.bounds[1]) for tm in tm_lst])
    sphere_mesh = trimesh.primitives.Sphere(radius=0.025 * scale)
    help2 = torch.tensor([[1, 0, 0, 0, 1]]).expand(n_t, -1)

    for i in vis_ids:
        save_path = os.path.join(save_folder, f"{i}.usd")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        usd_stage = UsdStage(save_path, timestep=data_length, dt=0.01)

        mesh_lst = [m.coarse_mesh for m in vis_dict.meshes]
        pose_lst = (
            torch.cat([t[:, i], q[:, i], torch.ones((n_t, n_m, 1))], dim=-1)
            .transpose(0, 1)
            .tolist()
        )
        name_lst = [f"mesh{j}" for j in range(1, n_m + 1)]
        time_lst = [(0, data_length)] * n_m
        material_lst = [MATERIAL_CLS[name2material[n]] for n in name_lst]

        for key, val in vis_dict.__dict__.items():
            if len(val) == 0 or key not in name2material:
                continue
            if len(val.shape) == 5:
                s1, s2, s3, s4, s5 = val.shape[:]
                val = val.reshape(s1, s2, s3 * s4, s5)
            n_pp = val.shape[-2]
            material_lst += [MATERIAL_CLS[name2material[key]]] * n_pp
            time_lst += [(0, data_length)] * n_pp
            mesh_lst += [sphere_mesh] * n_pp
            for j in range(n_pp):
                name_lst.append(key + str(j))
                pose_lst.append(torch.cat([val[:, i, j], help2], dim=-1).tolist())
        usd_stage.add_mesh_lst(mesh_lst, name_lst, pose_lst, time_lst, material_lst)
        usd_stage.write_stage_to_file(save_path)
    logging.info(f"Saved USD to folder {os.path.abspath(save_folder)}")
    return
