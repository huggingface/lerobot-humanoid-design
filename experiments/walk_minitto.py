import pinocchio as pin
import crocoddyl as croc
import numpy as np
import matplotlib.pylab as plt  # noqa: F401
from numpy.linalg import norm, pinv, inv, svd, eig  # noqa: F401
from urdf.humanoids_loader import loadbasic,loadTunedHipx,loadBipedalPlateform
from experiments.V0_walk_param import WalkV0Params
from experiments.V0_sidewalk_param import SideWalkV0Params
# Local imports
import sobec
import sobec.walk_without_think.plotter
import sys

import os




### load 6d model
walkParams = WalkV0Params()

cwd=os.getcwd()
robot = loadBipedalPlateform()


assert len(walkParams.stateImportance) == robot.model.nv * 2
assert len(walkParams.stateTerminalImportance) == robot.model.nv * 2

contactPattern = walkParams.contactPattern

import meshcat
from pinocchio.visualize import MeshcatVisualizer
viz = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
viz.viewer = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
viz.clean()
viz.loadViewerModel(rootNodeName="universe")
viz.display(robot.x0[: robot.model.nq])



# #####################################################################################
# ### DDP #############################################################################
# #####################################################################################

ddp = sobec.wwt.buildSolver(robot, contactPattern, walkParams, solver='FDDP')
problem = ddp.problem
x0s, u0s = sobec.wwt.buildInitialGuess(ddp.problem, walkParams)
ddp.setCallbacks([croc.CallbackVerbose(), croc.CallbackLogger()])

Us=np.array(u0s)


croc.enable_profiler()
ddp.solve(init_xs=x0s, init_us=u0s, maxiter=800, init_reg=1e-6, is_feasible=False)

sol = sobec.wwt.Solution(robot, ddp)
plotter = sobec.wwt.plotter.WalkPlotter(robot.model, robot.contactIds)
plotter.setData(contactPattern, sol.xs, sol.us, sol.fs0)
# plotter.plotJointTorques()
# plt.show()
sol = sobec.wwt.Solution(robot, ddp)




while input("Press q to quit the visualisation") != "q":
    viz.play(np.array(ddp.xs)[:, : robot.model.nq], walkParams.DT)



stop



vs = sol.xs[:,robot.model.nq:]
velocity = vs[:,0]

# ### PLOT ######################################################################
# ### PLOT ######################################################################
# ### PLOT ######################################################################
import os
import matplotlib.pyplot as plt

titles = [
    "Hip Z",
    "Hip X",
    "Hip Y",
    "Knee",
    "Ankle Y",
    "Ankle X",
]

plt.figure()
Ltau = []
Ltau_max = []

torque_limit  = [7,10,40,40,1.5,1.5]

for i in range(6):
    plt.subplot(711 + i)
    limit = torque_limit[i]
    plt.plot(sol.us[:, i], label="left")
    Ltau.append(sol.us[:, i])
    plt.plot(sol.us[:, 6 + i], label="right")
    plt.ylim(-limit,limit)
    plt.title(titles[i])
    plt.legend()
    plt.ylabel("torque (Nm)")

plt.subplot(717)
plt.plot(velocity)
plt.title("torso velocity")
plt.ylabel("Velocity (ms^-1)")
cwd = os.getcwd()
plt.show()


np.save(path+"torque.npy",np.array(Ltau))
np.save(path+"xs.npy",np.array(np.array(ddp.xs)))





