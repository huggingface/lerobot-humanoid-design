import argparse
from pathlib import Path

import crocoddyl as croc
import numpy as np
import pinocchio as pin
import sobec

from jump_loader import load_real_robot
from jump_params import JumpRealRobotParams


def load_jump_config(path: Path) -> dict:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Jump OCP for to_real_robot/model/urdf/robot.urdf")
    parser.add_argument("--config", type=Path, default=None, help="YAML jump config (fconf style)")
    parser.add_argument("--base-height", type=float, default=0.58)
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--solver", type=str, default="FDDP")
    parser.add_argument("--save", type=Path, default=None, help="Output trajectory .npy")
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    config = {}
    if args.config is not None:
        config = load_jump_config(args.config)

    robot_cfg = config.get("robot", {})
    solver_cfg = config.get("solver", {})
    jump_cfg = config.get("jump", {})

    base_height = float(robot_cfg.get("base_height", args.base_height))
    maxiter = int(solver_cfg.get("maxiter", args.maxiter))
    solver_name = str(solver_cfg.get("name", args.solver))

    robot = load_real_robot(base_height=base_height)
    jump_params = JumpRealRobotParams()
    jump_params.solver_maxiter = maxiter

    contact_pattern = jump_cfg.get("contact_pattern", jump_params.contactPattern)

    ddp = sobec.wwt.buildJumpSolver(robot, contact_pattern, jump_params, solver=solver_name)
    x0s, u0s = sobec.wwt.buildInitialGuess(ddp.problem, jump_params)
    ddp.setCallbacks([croc.CallbackVerbose(), croc.CallbackLogger()])
    ddp.solve(x0s, u0s, maxiter)

    sol = sobec.wwt.Solution(robot, ddp)

    save_path = args.save
    if save_path is None:
        save_cfg = jump_cfg.get("save_file", None)
        if save_cfg:
            save_path = Path(save_cfg)

    if save_path is not None:
        sobec.wwt.save_traj(
            xs=np.array(sol.xs),
            us=np.array(sol.us),
            fs=sol.fs0,
            acs=sol.acs,
            n_iter=ddp.iter,
            filename=str(save_path),
        )
        print(f"Saved trajectory: {save_path}")

    print(f"Solved jump OCP with {len(ddp.xs) - 1} shooting nodes, iter={ddp.iter}")

    if args.visualize:
        try:
            import meshcat
            from pinocchio.visualize import MeshcatVisualizer

            viz = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
            viz.viewer = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
            viz.clean()
            viz.loadViewerModel(rootNodeName="universe")
            viz.play(np.array(ddp.xs)[:, : robot.model.nq], jump_params.DT)
        except Exception as exc:
            print(f"Visualization unavailable: {exc}")

    pin.SE3.__repr__ = pin.SE3.__str__


if __name__ == "__main__":
    main()
