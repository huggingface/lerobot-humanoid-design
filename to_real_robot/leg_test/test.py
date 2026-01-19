import os
import numpy as np
import pinocchio as pin
import meshcat
from pinocchio.visualize import MeshcatVisualizer
from example_parallel_robots.loader_tools import completeRobotLoader

from to_real_robot.leg_test.leg_robot import LegRobot
from to_real_robot.leg_test.kinematics import PinToRobotIndexMap

CWD = os.getcwd()
model, constraint_models, actuation_model, visual_model, collision_model = completeRobotLoader(
    CWD + "/urdf/humanoid_v1/urdf",
    freeflyer=False,
)

viz = MeshcatVisualizer(model, visual_model, visual_model)
viz.viewer = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
viz.clean()
viz.loadViewerModel(rootNodeName="universe")
viz.display(pin.neutral(model))

robot = LegRobot(channel="can0")
robot.attach_meshcat(viz)
robot.attach_kinematics(
    model=model,
    constraint_models=constraint_models,
    actuation_model=actuation_model,
    visual_model=visual_model,
    collision_model=collision_model,
    pin_map=PinToRobotIndexMap(hipz=0, hipx=1, hipy=2, knee=3, ankle_pitch=4, ankle_roll=5),
)

# read state (can estop if outside limits)+----------------------
robot.request_state()

# enable (only if not estopped)
if not robot.estop:
    for mid in range(1,7):
        robot.enable(mid)

robot.start_viz(hz=30, auto_request_state=True)

# Send IK target (meters, pin frame convention)



q_sol=np.array([np.pi,  np.pi,  -np.pi/2,   4.50133864,
         5.80062565,  -5.98131086])
target = np.array([0.0, -0.1, -0.5])
q_sol = robot.ik_foot_translation(target,q_prec=q_sol)

s=robot.pin_to_robot_deg(q_sol)

for i in range(1,5):
    pos=s[i]
    if i == 3 or i == 4:
        pos=pos-360
    print(pos)
    robot.move_joint_deg(i,pos,recv=False)
    input("Press Enter to continue...")






q_sol=np.array([np.pi,  np.pi,  -np.pi/2,   4.50133864,
         5.80062565,  -5.98131086])
import time
for i in range(3000):
    ofset=0.15*np.sin(2*np.pi*0.01*i)
    target = np.array([0.0, -0.1, -0.63-ofset])
    q_sol = robot.ik_foot_translation(target,q_prec=q_sol)

    s=robot.pin_to_robot_deg(q_sol)
    for i in range(1,5):
        pos=s[i]
        if i == 3 or i == 4:
            pos=pos-360

        robot.move_joint_deg(i,pos,recv=False)
    time.sleep(0.01)




target_pos=np.array([0.0,-0.1,-0.5])

robot.enable(3)
robot.move_joint_deg(3,-7,recv=False)



robot.enable(4)
robot.move_joint_deg(4,-105,recv=False)



robot.enable(1)
robot.move_joint_deg(1,150,recv=False)



robot.enable(2)
robot.move_joint_deg(2,155,recv=False)


robot.set_joint_gains(3,kp=30,kd=0.5)

robot.set_joint_gains(4,kp=20,kd=0.5)

robot.set_joint_gains(1,kp=30,kd=0.5)

robot.set_joint_gains(2,kp=30,kd=0.5)