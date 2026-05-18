import os

import crocoddyl as croc
import matplotlib.pyplot as plt
import numpy as np
import sobec
import sobec.walk_without_think.plotter

from experiments.V0_walk_param import WalkV0Params
from urdf.humanoids_loader import loadBipedalPlateform

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "walk_lerobot_humanoid")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    walk_params = WalkV0Params()
    robot = loadBipedalPlateform()

    assert len(walk_params.stateImportance) == robot.model.nv * 2
    assert len(walk_params.stateTerminalImportance) == robot.model.nv * 2
    contact_pattern = walk_params.contactPattern

    import meshcat
    from pinocchio.visualize import MeshcatVisualizer

    viz = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
    viz.viewer = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
    viz.clean()
    viz.loadViewerModel(rootNodeName="universe")
    viz.display(robot.x0[: robot.model.nq])

    ddp = sobec.wwt.buildSolver(robot, contact_pattern, walk_params, solver="FDDP")
    x0s, u0s = sobec.wwt.buildInitialGuess(ddp.problem, walk_params)
    ddp.setCallbacks([croc.CallbackVerbose(), croc.CallbackLogger()])
    croc.enable_profiler()
    ddp.solve(init_xs=x0s, init_us=u0s, maxiter=800, init_reg=1e-6, is_feasible=False)

    sol = sobec.wwt.Solution(robot, ddp)
    plotter = sobec.wwt.plotter.WalkPlotter(robot.model, robot.contactIds)
    plotter.setData(contact_pattern, sol.xs, sol.us, sol.fs0)

    while input("Press q to quit the visualisation") != "q":
        viz.play(np.array(ddp.xs)[:, : robot.model.nq], walk_params.DT)

    vs = sol.xs[:, robot.model.nq :]
    velocity = vs[:, 0]

    titles = ["Hip Z", "Hip X", "Hip Y", "Knee", "Ankle Y", "Ankle X"]
    torque_limit = [7, 10, 40, 40, 1.5, 1.5]

    plt.figure()
    ltau = []
    for i in range(6):
        plt.subplot(711 + i)
        limit = torque_limit[i]
        plt.plot(sol.us[:, i], label="left")
        ltau.append(sol.us[:, i])
        plt.plot(sol.us[:, 6 + i], label="right")
        plt.ylim(-limit, limit)
        plt.title(titles[i])
        plt.legend()
        plt.ylabel("torque (Nm)")

    plt.subplot(717)
    plt.plot(velocity)
    plt.title("torso velocity")
    plt.ylabel("Velocity (ms^-1)")
    plt.show()

    np.save(os.path.join(OUTPUT_DIR, "torque.npy"), np.array(ltau))
    np.save(os.path.join(OUTPUT_DIR, "xs.npy"), np.array(ddp.xs))


if __name__ == "__main__":
    main()
