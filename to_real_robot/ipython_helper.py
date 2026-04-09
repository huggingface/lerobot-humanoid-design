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
robot.set_max_command_delta(40.0)

robot.start(mode="control", auto_enable=True)
robot.request_state_once()      # ensure valid stamps
robot.attach_default_meshcat()  # or attach_meshcat(viz)

robot._viz_hz = 20.0            # reduce lag a lot
robot.set_action(left={"hipz": -0.06, "hipx": 3.55, "hipy": 0.04, "knee": -16.89, "ankle_pitch": 12.332, "ankle_roll": 0.09},right={"hipz": 0.0021, "hipx": -3.59, "hipy": -0.033, "knee": -17.34, "ankle_pitch": -12.43, "ankle_roll": 0.12},)

robot.set_action(left={"hipz": -0.0, "hipx": 0, "hipy": 0.0, "knee": 0, "ankle_pitch": 0, "ankle_roll": 0.0},right={"hipz": 0.0, "hipx": 0, "hipy": -0.0, "knee": 0.0, "ankle_pitch": 0.0, "ankle_roll": 0.},)


# run bipedal_robot.py

# robot.start(mode="state_only", auto_enable=False)


# robot.set_max_command_delta(40.0)
gain=[[40,5],[90,5],[90,5],[90,5],[20,3],[20,3]]*2

for mid,gains in zip([1,2,3,4,5, 6,7,8,9,10, 11, 12],gain):
    robot.set_joint_gains(mid, kp=gains[0], kd=gains[1]) 

for mid in [1,7]:
    robot.set_joint_gains(mid, kp=40, kd=1) 

for mid in [5,6,11,12]:
    robot.set_joint_gains(mid, kp=60, kd=2) 


robot.set_joint_gains(2, kp=110, kd=1) 
robot.set_joint_gains(3, kp=110, kd=1) 
robot.set_joint_gains(4, kp=110, kd=1) 
robot.set_joint_gains(8, kp=110, kd=1) 
robot.set_joint_gains(9, kp=110, kd=1) 
robot.set_joint_gains(10, kp=110, kd=1) 


robot.set_joint_gains(8, kp=90, kd=5) 

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
    policy_path="RL_policy/2026-02-22_21-51-53.onnx",
)

# safety scaling (start small)
agent.spec.action_scale = 0.002

# optional command
agent.set_command_twist(0.0, 0.0, 0.0)

agent.start()


### 8BitDo controller quick check + RL integration

from gamepad_controller import GamepadController
import time

# 1) Optional: inspect detected input devices
# for path, name in GamepadController.list_devices():
#     print(path, name)

# 2) Connect and start gamepad reader
pad = GamepadController(
    name_substring="8bitdo",
    deadzone=0.12,
    max_lin_x=0.5,
    max_lin_y=0.3,
    max_yaw_rate=0.5,
)
pad.connect()  # or pad.connect(device_path="/dev/input/eventX")
pad.start()

# 3) Quick manual test (move sticks)
# while True:
#     print("twist:", pad.get_command_twist())
    # time.sleep(0.1)

# 4) Link to RL agent command channel
agent.set_command_source(pad)





### Mock Test thas should pass

from bipedal_robot import BipedalRobotController
from IMU_integration import IMU
from mock_bus import MockBus

imu = IMU(sensor="jy901", mock=False, port="/dev/ttyAMA0", baudrate=9600)
# robot = BipedalRobotController(control_hz=100.0, bus_can0=MockBus(), bus_can1=MockBus(),imu=imu)
robot = BipedalRobotController(control_hz=100.0, imu=imu)

robot.attach_default_meshcat()   # optional

for mid in range(1, 13):
    robot.set_joint_limit(mid, -720.0, 720.0)
robot.set_max_command_delta(1000.0)

robot.start(mode="control", auto_enable=True)
robot.request_state_once()      # ensure valid stamps

robot._viz_hz = 20.0            # reduce lag a lot
robot.set_action(left={"hipz": -0.06, "hipx": 3.55, "hipy": 0.04, "knee": -16.89, "ankle_pitch": 12.332, "ankle_roll": 0.09},right={"hipz": 0.0021, "hipx": -3.59, "hipy": -0.033, "knee": -17.34, "ankle_pitch": -12.43, "ankle_roll": 0.12},)

from pathlib import Path
from bipedal_robot import BipedalRobotController
from ocp_follower import OCPFollower
from ocp_controller_staged import load_ocp_npy, go_to_pose  # helpers already added

traj = load_ocp_npy(Path("/home/lerobot/devel/real_robot_jump.npy"), dt_s=0.005)
follower = OCPFollower(robot)
follower.set_trajectory(traj)

go_to_pose(robot, traj.q_deg[0], duration_s=2.0)


from RL_agent import RLAgent

from gamepad_controller import GamepadController
import time

# 1) Optional: inspect detected input devices
# for path, name in GamepadController.list_devices():
#     print(path, name)

# 2) Connect and start gamepad reader
pad = GamepadController(
    name_substring="8bitdo",
    deadzone=0.12,
    max_lin_x=0.5,
    max_lin_y=0.3,
    max_yaw_rate=0.5,
)
pad.connect()  # or pad.connect(device_path="/dev/input/eventX")
pad.start()

agent = RLAgent.from_files(
    robot,
    config_path="RL_policy/config.yaml",
    policy_path="RL_policy/2026-02-03_10-07-41.onnx",
)


agent = RLAgent.from_files(
    robot,
    config_path="RL_policy/config.yaml",
    policy_path="RL_policy/2026-02-26_06-32-06.onnx",
    log_path="RL_policy/real_robot_debug_ctrl.csv",  # optional
    clamp_ankle_to_true_limits=False,
    log_observation=True,                       # optional
    log_action=True,                            # optional
    log_every_n=1,                              # optional
)


for mid in [1,7]:
    robot.set_joint_gains(mid, kp=40, kd=1) 

for mid in [5,6,11,12]:
    robot.set_joint_gains(mid, kp=60, kd=2) 


robot.set_joint_gains(2, kp=110, kd=1) 
robot.set_joint_gains(3, kp=110, kd=1) 
robot.set_joint_gains(4, kp=110, kd=1) 
robot.set_joint_gains(8, kp=110, kd=1) 
robot.set_joint_gains(9, kp=110, kd=1) 
robot.set_joint_gains(10, kp=110, kd=1) 

# safety scaling (start small)
agent.spec.action_scale = 0.1

# optional command
agent.set_command_twist(0.0, 0.0, 0.0)
agent.spec.joint_vel_source = "finite_difference"  # or "robot_state_estimation"
agent.set_command_source(pad)
agent.start()



### Delay measurement helper

from measure_actuator_delay import measure_actuator_delays

# Assumes `robot` is already started in control mode and has valid state.
# Use `all_joints=True` to sweep the full robot.
results = measure_actuator_delays(
    robot,
    joint="left_hipz",
    all_joints=False,
    fps=100,
    duration_s=2.0,
    freq_hz=1.0,
    amp_deg=10.0,
    pre_roll_s=1.0,
    between_s=0.5,
    save_csv="delay_logs/delay_trace.csv",
    save_png="delay_logs/delay_trace.png",
)
print(results["left_hipz"]["metrics"])


## sim

import importlib, sim_robot
importlib.reload(sim_robot)
from sim_robot import SimBipedalRobotController
from RL_agent import RLAgent

robot = SimBipedalRobotController(control_hz=200.0)   # now uses scene.xml
robot.start(mode="control", auto_enable=True)
robot.start_viewer()  # open MuJoCo window

agent = RLAgent.from_files(
    robot,
    config_path="RL_policy/config.yaml",
    policy_path="RL_policy/2026-02-26_06-32-06.onnx",
)
agent.spec.action_scale = 1   # start lower while debugging
agent.start()



import importlib, sim_robot, RL_agent
importlib.reload(sim_robot)
importlib.reload(RL_agent)

from sim_robot import SimBipedalRobotController
from RL_agent import RLAgent

robot = SimBipedalRobotController(control_hz=200.0)  # now uses sim_scene_safe.xml
robot.enable_debug_trace("RL_policy/sim_robot_trace.csv", every_n=1)

robot.start(mode="control", auto_enable=True)
robot.start_viewer()
robot.set_action(left={"hipz": -0.0, "hipx": 0, "hipy": 0.0, "knee": 0, "ankle_pitch": 0, "ankle_roll": 0.0},right={"hipz": 0.0, "hipx": 0, "hipy": -0.0, "knee": 0.0, "ankle_pitch": 0.0, "ankle_roll": 0.},)

agent = RLAgent.from_files(
    robot,
    config_path="RL_policy/config.yaml",
    policy_path="RL_policy/2026-02-26_20-56-19.onnx",
    log_path="RL_policy/sim_robot_debug_ctrl.csv",  # optional
    clamp_ankle_to_true_limits=False,
    log_observation=True,                       # optional
    log_action=True,                            # optional
    log_every_n=1,                              # optional
)

agent.spec.action_scale = 1
agent.start()


from sim_robot import SimBipedalRobotController

robot = SimBipedalRobotController(control_hz=200.0, fixed_base=False)
robot.start(mode="control", auto_enable=True)
robot.start_viewer()


robot.set_action(
left={
    "hipz": 0.0,
    "hipx": 0.0,
    "hipy": -20.0535,
    "knee": 40.1070,
    "ankle_pitch": -20.0535,  # from ankley_left
    "ankle_roll": 0.0,        # from anklex_left
},
right={
    "hipz": 0.0,
    "hipx": 0.0,
    "hipy": 20.0535,
    "knee": 40.1070,
    "ankle_pitch": 20.0535,   # from ankley_right
    "ankle_roll": 0.0,        # from anklex_right
},
)


### Isolated RL agent (same usage style) + gamepad command source

from RL_agent_isolated import RLAgent
from gamepad_controller import GamepadController
from sim_robot import SimBipedalRobotController
import importlib, sim_robot
importlib.reload(sim_robot)
from sim_robot import SimBipedalRobotController
import warnings, glfw
warnings.filterwarnings("ignore", category=glfw.GLFWError)

robot = SimBipedalRobotController(control_hz=200.0, fixed_base=False)
robot.start(mode="control", auto_enable=True)
robot.start_viewer()
print("debug logs:", robot._debug_action_logs)  # should be False
robot._viz_hz = 20.0  


pad = GamepadController(
    name_substring="8bitdo",
    deadzone=0.12,
    max_lin_x=0.75,
    max_lin_y=0.5,
    max_yaw_rate=0.8,
)
pad.connect()
pad.start()

agent = RLAgent.from_files(
    robot,
    config_path="RL_policy/config.yaml",
    policy_path="RL_policy/2026-03-01_08-33-51.onnx",
    log_path="RL_policy/isolated_debug_ctrl.csv",
    log_observation=True,
    log_action=True,
    log_every_n=1,
)

# manual global scaling remains available
agent.spec.action_scale = 1

# pad provides (lin_x, lin_y, yaw_rate) commands
agent.set_command_source(pad)
agent.start()
import time
for _ in range(50):
    q = robot._read_joint_q_deg()
    robot._warn_joint_state_out_of_bounds(q)
    time.sleep(0.6)














### Mock Test thas should pass

from bipedal_robot import BipedalRobotController
from IMU_integration import IMU
from mock_bus import MockBus

from RL_agent_isolated import RLAgent
from gamepad_controller import GamepadController
imu = IMU(sensor="bno055", i2c_bus=1, address=0x28, rate_hz=100.0, frame_yaw_deg=180.0)

# imu = IMU(sensor="jy901", mock=False, port="/dev/ttyAMA0", baudrate=9600)
# robot = BipedalRobotController(control_hz=100.0, bus_can0=MockBus(), bus_can1=MockBus(),imu=imu)
robot = BipedalRobotController(control_hz=100.0, imu=imu)

robot.attach_default_meshcat()   # optional
robot.set_max_command_delta(60.0)
robot.start(mode="state_only", auto_enable=False)
import time
while True:
    print(imu.read_dict()["gyro_rads"])
    time.sleep(0.1)
for mid in range(1, 13):
    robot.set_joint_limit(mid, -720.0, 720.0)
robot.set_max_command_delta(1000.0)

robot.start(mode="control", auto_enable=True)
robot.request_state_once()      # ensure valid stamps
robot.enforce_command_limits = False

robot._viz_hz = 20.0            # reduce lag a lot

robot.set_action(
left={
    "hipz": 0.0,
    "hipx": 0.0,
    "hipy": 0,
    "knee": 0.,
    "ankle_pitch": 0.0,  # from ankley_left
    "ankle_roll": 0.0,        # from anklex_left
},
right={
    "hipz": 0.0,
    "hipx": 0.0,
    "hipy": 0.0,
    "knee": 00.,
    "ankle_pitch": 0.0,   # from ankley_right
    "ankle_roll": 0.0,        # from anklex_right
},
)
robot.set_mode("control")


for mid in [1,7]:
    robot.set_joint_gains(mid, kp=30, kd=3.0) 

for mid in [5,6,11,12]:
    robot.set_joint_gains(mid, kp=10, kd=0.75) 


robot.set_joint_gains(2, kp=40, kd=3.0) 
robot.set_joint_gains(3, kp=6, kd=4/20) 
robot.set_joint_gains(4, kp=6, kd=4/20) 
robot.set_joint_gains(8, kp=40, kd=3.0) 
robot.set_joint_gains(9, kp=6, kd=4/20)
robot.set_joint_gains(10, kp=6, kd=4/20)


pad = GamepadController(
    name_substring="8bitdo",
    deadzone=0.12,
    max_lin_x=0.75,
    max_lin_y=0.5,
    max_yaw_rate=0.8,
)
pad.connect()
pad.start()

agent = RLAgent.from_files(
    robot,
    config_path="RL_policy/less_noice_high_gain_torque_obs/config.yaml",
    policy_path="RL_policy/less_noice_high_gain_torque_obs/policy.onnx", #2026-03-04_17-28-46.onnx",
    log_path="RL_policy/less_noice_high_gain_torque_obs/debug_ctrl5.csv",
    log_observation=True,
    log_action=True,
    log_every_n=1,
)

agent.spec.joint_vel_source = "auto"  # or "robot_state_estimation"

# manual global scaling remains available
agent.spec.action_scale = 0.0

# pad provides (lin_x, lin_y, yaw_rate) commands
agent.set_command_source(pad)
agent.start()
