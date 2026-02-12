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


THIS_DIR = Path(__file__).resolve().parent
TO_REAL_ROBOT_DIR = THIS_DIR.parents[2]
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


def load_real_robot(base_height: float = 0.58):
    model, constraint_models, actuation_model, visual_model, collision_model = completeRobotLoader(
        str(URDF_DIR), freeflyer=True
    )

    q0 = pin.neutral(model)
    q0[2] = base_height
    q0 = _align_base_height_with_feet(model, q0)

    model.referenceConfigurations["half_sitting"] = q0

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
