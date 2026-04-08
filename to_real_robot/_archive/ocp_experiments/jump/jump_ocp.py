import argparse
from pathlib import Path
import time

import crocoddyl as croc
import numpy as np
import pinocchio as pin
import sobec

try:
    from .jump_loader import load_real_robot
    from .jump_params import JumpRealRobotParams
except ImportError:
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
    parser.add_argument("--torso-contact-z-offset", type=float, default=None)
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--solver", type=str, default="FDDP")
    parser.add_argument("--save", type=Path, default=None, help="Output trajectory .npy")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--plot-torque", action="store_true")
    parser.add_argument("--save-torque-plot", type=Path, default=Path("/tmp/real_robot_torque.png"))
    parser.add_argument("--record-mp4", type=Path, default=None, help="Record trajectory to mp4")
    parser.add_argument("--record-fps", type=int, default=0, help="0 means fps=round(1/DT)")
    args = parser.parse_args()

    config = {}
    if args.config is not None:
        config = load_jump_config(args.config)

    robot_cfg = config.get("robot", {})
    solver_cfg = config.get("solver", {})
    jump_cfg = config.get("jump", {})

    base_height = float(robot_cfg.get("base_height", args.base_height))
    if args.torso_contact_z_offset is not None:
        torso_contact_z_offset = float(args.torso_contact_z_offset)
    elif "torso_contact_z_offset" in robot_cfg:
        torso_contact_z_offset = float(robot_cfg["torso_contact_z_offset"])
    elif "q0_offset_z" in robot_cfg:
        # Backward-compatible key used in earlier revisions.
        torso_contact_z_offset = float(robot_cfg["q0_offset_z"])
    else:
        torso_contact_z_offset = -0.10
    maxiter = int(solver_cfg.get("maxiter", args.maxiter))
    solver_name = str(solver_cfg.get("name", args.solver))

    robot = load_real_robot(
        base_height=base_height, torso_contact_z_offset=torso_contact_z_offset
    )
    jump_params = JumpRealRobotParams()
    jump_params.solver_maxiter = maxiter

    contact_pattern = jump_cfg.get("contact_pattern", jump_params.contactPattern)
    q0 = robot.x0[: robot.model.nq]
    data0 = robot.model.createData()
    pin.framesForwardKinematics(robot.model, data0, q0)
    lz = data0.oMf[robot.model.getFrameId("foot_frame_left")].translation[2]
    rz = data0.oMf[robot.model.getFrameId("foot_frame_right")].translation[2]
    torso_frame_name = (
        "torso_subassembly"
        if robot.model.existFrame("torso_subassembly")
        else "torso"
    )
    tz = data0.oMf[robot.model.getFrameId(torso_frame_name)].translation[2]
    print(
        "Initial pose: "
        f"base_z={q0[2]:.4f} m, left_foot_z={lz:.4f} m, right_foot_z={rz:.4f} m, "
        f"{torso_frame_name}_z={tz:.4f} m, "
        f"torso_contact_z_offset={torso_contact_z_offset:.4f} m"
    )

    viz = None
    need_viz = args.visualize or (args.record_mp4 is not None)
    if need_viz:
        try:
            import meshcat
            from pinocchio.visualize import MeshcatVisualizer

            viz = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
            viz.viewer = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
            viz.clean()
            viz.loadViewerModel(rootNodeName="universe")
            viz.display(q0)
        except Exception as exc:
            print(f"Visualization unavailable: {exc}")
            viz = None

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

    tau_saved = None
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
        try:
            data_saved = np.load(str(save_path), allow_pickle=True).item()
            if "us" in data_saved:
                tau_saved = np.array(data_saved["us"])
        except Exception as exc:
            print(f"Warning: could not reload saved trajectory for torque plot: {exc}")

    print(f"Solved jump OCP with {len(ddp.xs) - 1} shooting nodes, iter={ddp.iter}")

    if args.plot_torque:
        try:
            import matplotlib
            # Avoid backend crashes on headless machines.
            if not args.visualize:
                matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt

            # Use exactly what is saved on disk when available.
            tau = tau_saved if tau_saved is not None else np.array(sol.us)
            if tau.size == 0:
                print("Torque plot skipped: empty control trajectory.")
            else:
                n_u = tau.shape[1]
                # For 12 motors, overlay left/right torque of each motor family.
                # Order assumed: [L6 motors, R6 motors].
                if n_u == 12:
                    # Motor order (0-based) in this setup:
                    # [hipz, hipy, hipx, knee, ankley, anklex] for left,
                    # then same order for right.
                    motor_names = ["hipz", "hipy", "hipx", "knee", "ankley", "anklex"]
                    fig, axes = plt.subplots(6, 1, figsize=(10, 12), sharex=True)
                    for i, name in enumerate(motor_names):
                        axes[i].plot(tau[:, i], linewidth=1.4, label=f"{name}_left")
                        axes[i].plot(tau[:, i + 6], linewidth=1.4, linestyle="--", label=f"{name}_right")
                        axes[i].set_ylabel(name)
                        axes[i].grid(True, alpha=0.3)
                        axes[i].legend(loc="upper right")
                    axes[-1].set_xlabel("k")
                else:
                    fig, axes = plt.subplots(
                        n_u, 1, figsize=(10, max(4, 1.7 * n_u)), sharex=True
                    )
                    if n_u == 1:
                        axes = [axes]
                    for i in range(n_u):
                        axes[i].plot(tau[:, i], linewidth=1.2)
                        axes[i].set_ylabel(f"u{i}")
                        axes[i].grid(True, alpha=0.3)
                    axes[-1].set_xlabel("k")
                fig.tight_layout()
                fig.savefig(args.save_torque_plot)
                plt.close(fig)
                print(f"Saved torque plot: {args.save_torque_plot}")
                if args.visualize:
                    plt.ion()
                    plt.show()
        except Exception as exc:
            print(f"Torque plot unavailable: {exc}")

    if args.record_mp4 is not None:
        if viz is None:
            print("MP4 recording skipped: viewer unavailable.")
        else:
            try:
                import imageio.v2 as imageio

                fps = args.record_fps if args.record_fps > 0 else max(1, int(round(1.0 / jump_params.DT)))
                q_traj = np.array(ddp.xs)[:, : robot.model.nq]
                frames = []
                for q in q_traj:
                    viz.display(q)
                    img = viz.viewer.get_image()
                    frames.append(np.asarray(img))
                    # Keep temporal coherence for remote meshcat renderer.
                    time.sleep(jump_params.DT)
                imageio.mimsave(str(args.record_mp4), frames, fps=fps)
                print(f"Saved mp4: {args.record_mp4}")
            except Exception as exc:
                print(f"MP4 recording unavailable: {exc}")

    if args.visualize and viz is not None:
        while input("Press q to quit the visualisation") != "q":
            viz.play(np.array(ddp.xs)[:, : robot.model.nq], jump_params.DT)

    pin.SE3.__repr__ = pin.SE3.__str__


if __name__ == "__main__":
    main()
