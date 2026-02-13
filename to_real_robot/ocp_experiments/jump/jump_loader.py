from pathlib import Path

import numpy as np
import pinocchio as pin
import sobec

try:
    from example_parallel_robots.loader_tools import completeRobotLoader
except ImportError as exc:
    raise ImportError(
        "completeRobotLoader is required. Install example_parallel_robots first."
    ) from exc

try:
    from toolbox_parallel_robots.mounting import closedLoopMountProximal
except ImportError:
    closedLoopMountProximal = None


THIS_DIR = Path(__file__).resolve().parent
TO_REAL_ROBOT_DIR = THIS_DIR.parents[1]
URDF_DIR = TO_REAL_ROBOT_DIR / "model" / "urdf"


def _align_base_height_with_feet(model: pin.Model, q: np.ndarray) -> np.ndarray:
    data = model.createData()
    pin.forwardKinematics(model, data, q)
    pin.framesForwardKinematics(model, data, q)
    l_id = model.getFrameId("foot_left")
    r_id = model.getFrameId("foot_right")
    foot_z = 0.5 * (data.oMf[l_id].translation[2] + data.oMf[r_id].translation[2])
    q[2] -= foot_z
    return q


def _compute_q0_from_contacts(
    model: pin.Model,
    constraint_models,
    q_seed: np.ndarray,
    torso_contact_z_offset: float,
) -> np.ndarray:
    if closedLoopMountProximal is None:
        # Fallback if toolbox is missing: use floating-base shift only.
        q0 = q_seed.copy()
        q0[2] += torso_contact_z_offset
        return q0

    data = model.createData()
    pin.forwardKinematics(model, data, q_seed)
    pin.framesForwardKinematics(model, data, q_seed)

    foot_left_id = model.getFrameId("foot_left")
    foot_right_id = model.getFrameId("foot_right")
    torso_frame_name = (
        "torso_subassembly" if model.existFrame("torso_subassembly") else "torso"
    )
    torso_id = model.getFrameId(torso_frame_name)

    left_target = data.oMf[foot_left_id].copy()
    right_target = data.oMf[foot_right_id].copy()
    torso_target = data.oMf[torso_id].copy()

    # Keep feet on the same support plane.
    avg_foot_z = 0.5 * (left_target.translation[2] + right_target.translation[2])
    left_target.translation[2] = avg_foot_z
    right_target.translation[2] = avg_foot_z

    # This is the requested behavior: move torso contact target, not whole base.
    torso_target.translation[2] += torso_contact_z_offset
    torso_target.translation[1] = 0.03
    extra_constraints = []
    for fid, world_target in (
        (foot_left_id, left_target),
        (foot_right_id, right_target),
        (torso_id, torso_target),
    ):
        frame = model.frames[fid]
        extra_constraints.append(
            pin.RigidConstraintModel(
                pin.ContactType.CONTACT_6D,
                model,
                frame.parentJoint,
                frame.placement,
                0,
                world_target,
                pin.ReferenceFrame.LOCAL,
            )
        )

    all_constraints = list(constraint_models) + extra_constraints
    all_cdata = [cm.createData() for cm in all_constraints]
    return closedLoopMountProximal(model, data, all_constraints, all_cdata)


def load_real_robot(base_height: float = 0.58, torso_contact_z_offset: float = -0.10):
    urdf_file = URDF_DIR / "robot.urdf"
    if not urdf_file.exists():
        raise FileNotFoundError(f"URDF not found at expected path: {urdf_file}")

    model, constraint_models, actuation_model, visual_model, collision_model = completeRobotLoader(
        str(URDF_DIR), freeflyer=True
    )

    q_seed = pin.neutral(model)
    q_seed[2] = base_height
    q_seed = _align_base_height_with_feet(model, q_seed)
    q0 = _compute_q0_from_contacts(model, constraint_models, q_seed, torso_contact_z_offset)

    model.referenceConfigurations["half_sitting"] = q0

    # Bound free-flyer translation around initial configuration.
    # Keep in-place motion and prevent lowering below base configuration.
    model.lowerPositionLimit[0] = q0[0] - 0.03
    model.upperPositionLimit[0] = q0[0] + 0.03
    model.lowerPositionLimit[1] = q0[1] - 0.03
    model.upperPositionLimit[1] = q0[1] + 0.03
    model.lowerPositionLimit[2] = q0[2] - 0.05
    model.upperPositionLimit[2] = q0[2] + 0.30

    left_id = model.getFrameId("foot_left")
    right_id = model.getFrameId("foot_right")
    model.frames[left_id].name = "foot_frame_left"
    model.frames[right_id].name = "foot_frame_right"

    robot = sobec.wwt.RobotWrapper(model, contactKey="foot_frame", closed_loop=False)
    robot.collision_model = collision_model
    robot.visual_model = visual_model
    robot.actuationModel = actuation_model
    robot.loop_constraints_models = constraint_models

    if len(robot.contactIds) != 2:
        raise RuntimeError(
            f"Expected 2 foot contacts, got {len(robot.contactIds)} with key 'foot_frame'."
        )

    armature = np.zeros(model.nv)
    for vid in actuation_model.mot_ids_v:
        armature[vid] = 1e-3
    robot.model.armature = armature

    robot.x0 = np.concatenate([q0, np.zeros(model.nv)])
    return robot
