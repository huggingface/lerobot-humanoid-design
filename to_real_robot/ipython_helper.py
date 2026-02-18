from pathlib import Path
from bipedal_robot import BipedalRobotController
from ocp_follower import OCPFollower
from ocp_controller_staged import load_ocp_npy, go_to_pose  # helpers already added

# 1) Create robot controller (real CAN)
robot = BipedalRobotController(control_hz=200.0, log_path="bipedal_state_log.csv")
robot.start(mode="control", auto_enable=True)

# 2) Create OCP follower + load trajectory
traj = load_ocp_npy(Path("/tmp/real_robot_jump_mirrored_from_middle_zero.npy"), dt_s=0.01)
follower = OCPFollower(robot)
follower.set_trajectory(traj)

# 3) Put robot in initial configuration
go_to_pose(robot, traj.q_deg[0], duration_s=2.0)

# 4) Launch tracking
follower.run()












from bipedal_robot import BipedalRobotController
from mock_bus import MockBus

robot = BipedalRobotController(bus_can0=MockBus(), bus_can1=MockBus(), control_hz=100.0)
for mid in range(1, 13):
    robot.set_joint_limit(mid, -720.0, 720.0)
robot.set_max_command_delta(1000.0)

robot.start(mode="control", auto_enable=True)
robot.request_state_once()      # ensure valid stamps
robot.attach_default_meshcat()  # or attach_meshcat(viz)

robot._viz_hz = 20.0            # reduce lag a lot
robot.set_action(left={"hipz": -0.06, "hipx": 3.55, "hipy": 0.04, "knee": -16.89, "ankle_pitch": 12.332, "ankle_roll": 0.09},right={"hipz": 0.0021, "hipx": -3.59, "hipy": -0.033, "knee": -17.34, "ankle_pitch": -12.43, "ankle_roll": 0.12},)



# run bipedal_robot.py

# robot.start(mode="state_only", auto_enable=False)


# robot.set_max_command_delta(40.0)

for mid in [1,2,3,4,5, 6,7,8,9,10, 11, 12]:
    robot.set_joint_gains(mid, kp=45.0, kd=0.5) 


from pathlib import Path
from bipedal_robot import BipedalRobotController
from ocp_follower import OCPFollower
from ocp_controller_staged import load_ocp_npy, go_to_pose  # helpers already added

traj = load_ocp_npy(Path("/tmp/real_robot_jump_mirrored_from_middle.npy"), dt_s=0.005)
follower = OCPFollower(robot)
follower.set_trajectory(traj)

go_to_pose(robot, traj.q_deg[0], duration_s=2.0)





traj = load_ocp_npy(Path("/tmp/real_robot_jump.npy"), dt_s=0.04)
follower = OCPFollower(robot)
follower.set_trajectory(traj)



for i in range(20):
    follower.run()



from bipedal_robot import BipedalRobotController
from RL_agent import RLAgent

robot = BipedalRobotController(control_hz=200.0, log_path="bipedal_state_log.csv")
robot.start(mode="state_only", auto_enable=False)

# your manual safety steps...
# robot.set_action(...)
# robot.set_mode("control")
# robot.enable_all()
# robot.set_action(...)  # e.g. go to 0 pose


agent = RLAgent.from_files(
    robot,
    config_path="RL_policy/config.yaml",
    policy_path="RL_policy/2026-02-03_10-07-41.onnx",
)

# optional: load gains from RL mjcf
agent.apply_model_gains_from_mjcf("RL_policy/robot.xml")

# safety scaling (start small)
agent.spec.action_scale = 0.2

# optional command
agent.set_command_twist(0.0, 0.0, 0.0)

agent.start()
