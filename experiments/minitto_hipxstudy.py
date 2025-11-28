import pinocchio as pin
import crocoddyl as croc
import numpy as np
import matplotlib.pylab as plt  # noqa: F401
from numpy.linalg import norm, pinv, inv, svd, eig  # noqa: F401
from urdf.humanoids_loader import loadbasic,loadTunedHipx
from experiments.V0_walk_param import WalkV0Params
from experiments.V0_sidewalk_param import SideWalkV0Params
# Local imports
import sobec
import sobec.walk_without_think.plotter
import sys

import os

import matplotlib.pyplot as plt


### load 6d model
if False:
    walkParams = WalkV0Params()
else: 
    walkParams = SideWalkV0Params()
cwd=os.getcwd()


Ltorque_hip = []
Lvelocity = []
Loratate = []
for angle in range(90):
    # print(angle)
    # angle = 34
    robot = loadTunedHipx(angle)


    assert len(walkParams.stateImportance) == robot.model.nv * 2
    assert len(walkParams.stateTerminalImportance) == robot.model.nv * 2

    contactPattern = walkParams.contactPattern

    import meshcat
    from pinocchio.visualize import MeshcatVisualizer
    viz = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
    viz.viewer = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
    viz.clean()
    viz.loadViewerModel(rootNodeName="universe")
    # viz.display(robot.x0[: robot.model.nq])



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

    if sol.xs[200,6]>0.98 :
        rotate = False
    else :
        rotate =True
    Loratate.append(rotate)




    vs = sol.xs[:,robot.model.nq:]
    velocity = np.mean(vs[:,:2])
    # viz.play(np.array(ddp.xs)[:, : robot.model.nq], walkParams.DT)
    
    # ### PLOT ######################################################################
    # ### PLOT ######################################################################
    # ### PLOT ######################################################################

    torquez = sol.us[20:-20, 0] +sol.us[20:-20,6+ 0]
    torquex = sol.us[20:-20, 1]+sol.us[20:-20, 6+1]
    torquey = sol.us[20:-20, 2]+sol.us[20:-20, 6+2]

    Ltorque_hip.append([torquez,torquex,torquey])
    Lvelocity.append(np.mean(velocity))
    print(np.mean(velocity))

Lhipz=[]
Lhipx=[]
Lhipy=[]
Lmaxhipz=[]
Lmaxhipx=[]
Lmaxhipy=[]
for torque in Ltorque_hip:
    Lhipz.append(np.abs(torque[0]).mean())
    Lhipx.append(np.abs(torque[1]).mean())
    Lhipy.append(np.abs(torque[2]).mean())
    Lmaxhipz.append(np.abs(torque[0]).max())
    Lmaxhipx.append(np.abs(torque[1]).max())
    Lmaxhipy.append(np.abs(torque[2]).max())

plt.figure()
plt.subplot(311)

plt.plot(Lhipz, label="Hip Z", linewidth=2)
plt.plot(Lhipx, label="Hip X", linewidth=2)
plt.plot(Lhipy, label="Hip Y", linewidth=2)

plt.xlabel("Angle (deg)")
plt.ylabel("Mean |Torque| (Nm)")

plt.legend(frameon=False)
plt.grid(True, which="both", linestyle="--", alpha=0.4)



plt.subplot(312)
plt.plot(Lmaxhipz, label="Hip Z", linewidth=2)
plt.plot(Lmaxhipx, label="Hip X", linewidth=2)
plt.plot(Lmaxhipy, label="Hip Y", linewidth=2)

plt.xlabel("Angle (deg)")
plt.ylabel("Max |Torque| (Nm)")

plt.legend(frameon=False)
plt.grid(True, which="both", linestyle="--", alpha=0.4)

plt.subplot(313)
plt.xlabel("Angle (deg)")
plt.ylabel("mean velocity (ms^-1)")
plt.plot(Lvelocity)



plt.tight_layout()

plt.show()



strop
import os


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





