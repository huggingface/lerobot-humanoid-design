import os

import crocoddyl as croc
import matplotlib.pyplot as plt
import numpy as np
import sobec
import sobec.walk_without_think.plotter

from experiments.V0_sidewalk_param import SideWalkV0Params
from experiments.V0_walk_param import WalkV0Params
from urdf.humanoids_loader import loadTunedHipx

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "lerobot_humanoid_hipxstudy")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    use_sidewalk = True
    walk_params = SideWalkV0Params() if use_sidewalk else WalkV0Params()

    ltorque_hip = []
    lvelocity = []
    lrotate = []
    last_sol = None
    last_ddp = None
    last_velocity_trace = None

    for angle in range(90):
        robot = loadTunedHipx(angle)
        assert len(walk_params.stateImportance) == robot.model.nv * 2
        assert len(walk_params.stateTerminalImportance) == robot.model.nv * 2
        contact_pattern = walk_params.contactPattern

        ddp = sobec.wwt.buildSolver(robot, contact_pattern, walk_params, solver="FDDP")
        x0s, u0s = sobec.wwt.buildInitialGuess(ddp.problem, walk_params)
        ddp.setCallbacks([croc.CallbackVerbose(), croc.CallbackLogger()])
        croc.enable_profiler()
        ddp.solve(init_xs=x0s, init_us=u0s, maxiter=800, init_reg=1e-6, is_feasible=False)

        sol = sobec.wwt.Solution(robot, ddp)
        plotter = sobec.wwt.plotter.WalkPlotter(robot.model, robot.contactIds)
        plotter.setData(contact_pattern, sol.xs, sol.us, sol.fs0)

        rotate = sol.xs[200, 6] <= 0.98
        lrotate.append(rotate)

        vs = sol.xs[:, robot.model.nq :]
        mean_velocity = float(np.mean(vs[:, :2]))
        last_velocity_trace = vs[:, 0]
        torquez = sol.us[20:-20, 0] + sol.us[20:-20, 6]
        torquex = sol.us[20:-20, 1] + sol.us[20:-20, 7]
        torquey = sol.us[20:-20, 2] + sol.us[20:-20, 8]

        ltorque_hip.append([torquez, torquex, torquey])
        lvelocity.append(mean_velocity)
        if angle % 10 == 0:
            print(f"angle={angle} mean_velocity={mean_velocity:.4f}")

        last_sol = sol
        last_ddp = ddp

    lhipz = []
    lhipx = []
    lhipy = []
    lmaxhipz = []
    lmaxhipx = []
    lmaxhipy = []
    for torque in ltorque_hip:
        lhipz.append(np.abs(torque[0]).mean())
        lhipx.append(np.abs(torque[1]).mean())
        lhipy.append(np.abs(torque[2]).mean())
        lmaxhipz.append(np.abs(torque[0]).max())
        lmaxhipx.append(np.abs(torque[1]).max())
        lmaxhipy.append(np.abs(torque[2]).max())

    plt.figure()
    plt.subplot(311)
    plt.plot(lhipz, label="Hip Z", linewidth=2)
    plt.plot(lhipx, label="Hip X", linewidth=2)
    plt.plot(lhipy, label="Hip Y", linewidth=2)
    plt.xlabel("Angle (deg)")
    plt.ylabel("Mean |Torque| (Nm)")
    plt.legend(frameon=False)
    plt.grid(True, which="both", linestyle="--", alpha=0.4)

    plt.subplot(312)
    plt.plot(lmaxhipz, label="Hip Z", linewidth=2)
    plt.plot(lmaxhipx, label="Hip X", linewidth=2)
    plt.plot(lmaxhipy, label="Hip Y", linewidth=2)
    plt.xlabel("Angle (deg)")
    plt.ylabel("Max |Torque| (Nm)")
    plt.legend(frameon=False)
    plt.grid(True, which="both", linestyle="--", alpha=0.4)

    plt.subplot(313)
    plt.xlabel("Angle (deg)")
    plt.ylabel("Mean velocity (ms^-1)")
    plt.plot(lvelocity)
    plt.tight_layout()
    plt.show()

    if last_sol is None or last_ddp is None or last_velocity_trace is None:
        return

    titles = ["Hip Z", "Hip X", "Hip Y", "Knee", "Ankle Y", "Ankle X"]
    torque_limit = [7, 10, 40, 40, 1.5, 1.5]

    plt.figure()
    ltau = []
    for i in range(6):
        plt.subplot(711 + i)
        limit = torque_limit[i]
        plt.plot(last_sol.us[:, i], label="left")
        ltau.append(last_sol.us[:, i])
        plt.plot(last_sol.us[:, 6 + i], label="right")
        plt.ylim(-limit, limit)
        plt.title(titles[i])
        plt.legend()
        plt.ylabel("torque (Nm)")

    plt.subplot(717)
    plt.plot(last_velocity_trace)
    plt.title("torso velocity")
    plt.ylabel("Velocity (ms^-1)")
    plt.show()

    np.save(os.path.join(OUTPUT_DIR, "torque.npy"), np.array(ltau))
    np.save(os.path.join(OUTPUT_DIR, "xs.npy"), np.array(last_ddp.xs))


if __name__ == "__main__":
    main()
