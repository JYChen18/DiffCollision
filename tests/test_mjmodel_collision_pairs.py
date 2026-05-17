import mujoco
import torch

from diffcollision import DiffCollision
from diffcollision.mjmesh import (
    get_collision_pair_margins_from_mjmodel,
    get_collision_pairs_from_mjmodel,
)


def _model(xml: str) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(xml)


def _pair_names(model: mujoco.MjModel, diffcoll: DiffCollision) -> set[tuple[str, str]]:
    return {
        (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(idx1)),
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(idx2)),
        )
        for idx1, idx2 in diffcoll.cfg._mesh_pair_ids
    }


def _body_pair_ids(model: mujoco.MjModel, name1: str, name2: str) -> tuple[int, int]:
    return tuple(
        sorted(
            (
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name1),
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name2),
            )
        )
    )


def test_build_from_mjmodel_keeps_compatible_non_adjacent_body_pairs():
    model = _model(
        """
        <mujoco>
          <worldbody>
            <body name="a" pos="-1 0 0">
              <joint type="free"/>
              <geom name="ga" type="box" size=".1 .1 .1"/>
            </body>
            <body name="b" pos="1 0 0">
              <joint type="free"/>
              <geom name="gb" type="box" size=".1 .1 .1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )

    diffcoll = DiffCollision.build_from_mjmodel(model)

    assert _pair_names(model, diffcoll) == {("a", "b")}


def test_build_from_mjmodel_exposes_body_ids_not_compact_mesh_indices():
    model = _model(
        """
        <mujoco>
          <worldbody>
            <body name="empty"/>
            <body name="a" pos="-1 0 0">
              <joint type="free"/>
              <geom name="ga" type="box" size=".1 .1 .1"/>
            </body>
            <body name="b" pos="1 0 0">
              <joint type="free"/>
              <geom name="gb" type="box" size=".1 .1 .1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )

    diffcoll = DiffCollision.build_from_mjmodel(model)
    body_id_a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "a")
    body_id_b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "b")

    assert diffcoll.cfg._mesh_ids.tolist() == [body_id_a, body_id_b]
    assert diffcoll.cfg._mesh_pair_ids.tolist() == [[body_id_a, body_id_b]]
    assert diffcoll.cfg._mesh_pair_indices.tolist() == [[0, 1]]


def test_forward_cpidx_uses_body_ids():
    model = _model(
        """
        <mujoco>
          <worldbody>
            <body name="empty"/>
            <body name="a">
              <joint type="free"/>
              <geom name="ga" type="box" size=".1 .1 .1"/>
            </body>
            <body name="b">
              <joint type="free"/>
              <geom name="gb" type="box" size=".1 .1 .1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )

    diffcoll = DiffCollision.build_from_mjmodel(model)
    transforms = torch.eye(4).view(1, 1, 4, 4).repeat(
        1, len(diffcoll.cfg._mesh_ids), 1, 1
    )
    result = diffcoll.forward(transforms)

    body_id_a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "a")
    body_id_b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "b")
    assert result.cpidx[0, 0].tolist() == [body_id_a, body_id_b]


def test_build_from_mjmodel_reads_pair_margin_tensor_from_mujoco_geoms():
    model = _model(
        """
        <mujoco>
          <worldbody>
            <body name="a" pos="-1 0 0">
              <joint type="free"/>
              <geom name="ga" type="box" size=".1 .1 .1" margin=".03"/>
            </body>
            <body name="b" pos="1 0 0">
              <joint type="free"/>
              <geom name="gb" type="box" size=".1 .1 .1" margin=".07"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )

    diffcoll = DiffCollision.build_from_mjmodel(model)

    assert diffcoll.cfg.margin.shape == (1,)
    assert diffcoll.cfg._margin.shape == (1,)
    assert torch.allclose(diffcoll.cfg._margin, torch.tensor([0.10]))


def test_explicit_geom_pair_uses_pair_margin():
    model = _model(
        """
        <mujoco>
          <worldbody>
            <body name="a" pos="-1 0 0">
              <joint type="free"/>
              <geom name="ga" type="box" size=".1 .1 .1" margin=".03"/>
            </body>
            <body name="b" pos="1 0 0">
              <joint type="free"/>
              <geom name="gb" type="box" size=".1 .1 .1" margin=".07"/>
            </body>
          </worldbody>
          <contact>
            <pair geom1="ga" geom2="gb" margin=".04"/>
          </contact>
        </mujoco>
        """
    )

    pair_margins = get_collision_pair_margins_from_mjmodel(model)

    assert pair_margins[_body_pair_ids(model, "a", "b")] == 0.04


def test_parent_child_body_pairs_are_filtered_by_default():
    model = _model(
        """
        <mujoco>
          <worldbody>
            <body name="parent">
              <joint type="free"/>
              <geom name="gp" type="box" size=".1 .1 .1"/>
              <body name="child" pos=".3 0 0">
                <joint type="hinge" axis="0 0 1"/>
                <geom name="gc" type="box" size=".1 .1 .1"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """
    )

    assert _body_pair_ids(model, "parent", "child") not in get_collision_pairs_from_mjmodel(model)


def test_parent_child_body_pairs_are_kept_when_filterparent_is_disabled():
    model = _model(
        """
        <mujoco>
          <option>
            <flag filterparent="disable"/>
          </option>
          <worldbody>
            <body name="parent">
              <joint type="free"/>
              <geom name="gp" type="box" size=".1 .1 .1"/>
              <body name="child" pos=".3 0 0">
                <joint type="hinge" axis="0 0 1"/>
                <geom name="gc" type="box" size=".1 .1 .1"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """
    )

    assert _body_pair_ids(model, "parent", "child") in get_collision_pairs_from_mjmodel(model)


def test_same_weld_body_pairs_are_filtered():
    model = _model(
        """
        <mujoco>
          <worldbody>
            <body name="root">
              <body name="a" pos="-.3 0 0">
                <geom name="ga" type="box" size=".1 .1 .1"/>
              </body>
              <body name="b" pos=".3 0 0">
                <geom name="gb" type="box" size=".1 .1 .1"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """
    )

    assert _body_pair_ids(model, "a", "b") not in get_collision_pairs_from_mjmodel(model)


def test_incompatible_contact_masks_are_filtered():
    model = _model(
        """
        <mujoco>
          <worldbody>
            <body name="a" pos="-1 0 0">
              <joint type="free"/>
              <geom name="ga" type="box" size=".1 .1 .1" contype="1" conaffinity="0"/>
            </body>
            <body name="b" pos="1 0 0">
              <joint type="free"/>
              <geom name="gb" type="box" size=".1 .1 .1" contype="2" conaffinity="0"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )

    assert _body_pair_ids(model, "a", "b") not in get_collision_pairs_from_mjmodel(model)


def test_explicit_geom_pair_bypasses_parent_and_mask_filters():
    model = _model(
        """
        <mujoco>
          <worldbody>
            <body name="parent">
              <joint type="free"/>
              <geom name="gp" type="box" size=".1 .1 .1" contype="1" conaffinity="0"/>
              <body name="child" pos=".3 0 0">
                <joint type="hinge" axis="0 0 1"/>
                <geom name="gc" type="box" size=".1 .1 .1" contype="2" conaffinity="0"/>
              </body>
            </body>
          </worldbody>
          <contact>
            <pair geom1="gp" geom2="gc"/>
          </contact>
        </mujoco>
        """
    )

    assert _body_pair_ids(model, "parent", "child") in get_collision_pairs_from_mjmodel(model)
