"""
Toolbox for motor configuration,
on exectution do th efolowing :
-check current protocole used by the motor and motor id
-swith to private protocole
-reboot motor
-change motor id
-change time delay
-open interface for changing what you want
-change protoocole to MIT protocole
-reboot motor
"""

from time import sleep, time

import can

from robstride_toolkit import *

protocol = None
motor_id = None
ping = False

bus = can.interface.Bus(interface="socketcan", channel="can0")


print("Pinging motor using CANopen protocol...")
for i in range(128):
    motor_id = i
    ping = ping_canopen(bus, motor_id)
    if ping:
        print(f"Motor {motor_id} responded to CANopen ping.")
        protocol = "CANopen"
        break

if not ping:
    print("Pinging motor using private protocol...")
    for i in range(128):
        motor_id = i
        ping = ping_private(bus, motor_id)
        if ping:
            print(f"Motor {motor_id} responded to private ping.")
            protocol = "private"
            break

if not ping:
    print("Pinging motor using MIT protocol...")
    for i in range(128):
        motor_id = i
        ping = ping_mit(bus, motor_id)
        if ping:
            print(f"Motor {motor_id} responded to mit ping.")
            protocol = "MIT"
            break
if not ping:
    print("No motor responded to any protocol ping.")
    raise SystemExit(1)


if protocol != "MIT":
    if protocol == "CANopen":
        print("Switching motor to private protocol...")
        switch_canopen_to_private(bus, motor_id)
        sleep(1.0)
        input("reboot motor and press enter to continue...")
        print("sxitching motor to MIT protocol...")
        switch_private_to_mit(bus, motor_id)
        input("reboot motor and press enter to continue...")
    elif protocol == "private":
        print("Switching motor to MIT protocol...")
        switch_private_to_mit(bus, motor_id)
        input("reboot motor and press enter to continue...")


print("current motor id is :", motor_id)
new_id = int(input("Enter new motor ID (0-127): "))
if 1 <= new_id <= 127:
    print(f"Changing motor ID from {motor_id} to {new_id}...")
    change_motor_id(bus, motor_id, new_id)
else:
    print("Invalid motor ID. Must be between 1 and 127.")





CAN_CMD_ZERO = 0xFE



def set_zero(bus,motor_id):
        data = [0xFF] * 7 + [CAN_CMD_ZERO]
        msg = can.Message(arbitration_id=motor_id, data=data, is_extended_id=False)
        bus.send(msg)
        print(bus.recv(0.5))


def scan_motors_on_two_buses(
    interface: str = "socketcan",
    channel_a: str = "can0",
    channel_b: str = "can1",
    motor_ids=range(1, 13),
) -> dict[str, list[int]]:
    """
    Open two CAN buses, ping motor IDs on each bus, and return present motors.

    Returns:
        {
            "can0": [ids...],
            "can1": [ids...],
        }
    """
    bus_a = can.interface.Bus(interface=interface, channel=channel_a)
    bus_b = can.interface.Bus(interface=interface, channel=channel_b)
    found = {channel_a: [], channel_b: []}
    try:
        for mid in motor_ids:
            if ping_mit(bus_a, int(mid)):
                found[channel_a].append(int(mid))
        for mid in motor_ids:
            if ping_mit(bus_b, int(mid)):
                found[channel_b].append(int(mid))
        return found
    finally:
        try:
            bus_a.shutdown()
        except Exception:
            pass
        try:
            bus_b.shutdown()
        except Exception:
            pass



# stop

### frequency test
import time
enable(bus,1)
mit_control(bus,6,1,0.1,0,0,0,)

msg = can.Message(arbitration_id=6, data=[127, 255, 127, 240, 40, 51, 55, 255], is_extended_id=False)
t_ini = time.time()
n_msg = 0
while time.time() - t_ini < 10.0:
    mit_control(bus,1,1,0.1,0,0,0,)
    n_msg += 1
    # bus.recv(0.01)
t_end = time.time()
dt = t_end - t_ini
print(f"Sent {n_msg} messages in {dt:.2f} seconds ({n_msg / dt:.2f} Hz)")



import can
CAN_CMD_ENABLE = 0xFC
CAN_CMD_DISABLE = 0xFD
CAN_CMD_SET_ZERO = 0xFE
CAN_CMD_CLEAR_FAULT = 0xFB


CAN_CMD_QUERY_PARAM = 0x33
CAN_CMD_WRITE_PARAM = 0x55
CAN_CMD_SAVE_PARAM = 0xAA

bus= can.interface.Bus(interface="socketcan", channel="can1")
motor_id = 1    
data = [0xFF] * 7 + [CAN_CMD_CLEAR_FAULT]
msg = can.Message(arbitration_id=motor_id, data=data, is_extended_id=False)

bus.send(msg)



#### Here write a small code that create 2 bus (can0 and can1) ping evry motor on each bus (1-12) and return a list with the motor present on each bus . 
bus1 = can.interface.Bus(interface="socketcan", channel="can0")
bus2 = can.interface.Bus(interface="socketcan", channel="can1")

#clear bus

msg=True
while msg:
    msg=bus1.recv(0.1)

msg=True
while msg:
    msg=bus2.recv(0.1)


for i in range(1,13):
    if ping_mit(bus1,i):
        print(f"Motor {i} is present on can0")
    if ping_mit(bus2,i):
        print(f"Motor {i} is present on can1")


#### Meshcat display for model/urdf/robit.urdf (fallback to model/urdf/robot.urdf)
import os
from pathlib import Path
import pinocchio as pin
import meshcat
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.visualize import MeshcatVisualizer

cwd = Path(os.getcwd())
urdf_candidates = [cwd / "model" / "urdf" / "robit.urdf", cwd / "model" / "urdf" / "robot.urdf"]
urdf_path = None
for p in urdf_candidates:
    if p.exists():
        urdf_path = p
        break

if urdf_path is None:
    raise FileNotFoundError(
        "Could not find URDF. Expected one of: model/urdf/robit.urdf or model/urdf/robot.urdf"
    )

robot_wrapper = RobotWrapper.BuildFromURDF(str(urdf_path), [str(urdf_path.parent)])
viz = MeshcatVisualizer(
    robot_wrapper.model,
    robot_wrapper.collision_model,
    robot_wrapper.visual_model,
)
viz.viewer = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
viz.clean()
viz.loadViewerModel(rootNodeName="universe")
viz.display(pin.neutral(robot_wrapper.model))


#### Fast state polling (10s) + meshcat update, modular for sign/offset checks
import numpy as np
from leg_test.mit import MotorState, decode_state_frame
from root_constant import MOTORS

# Convention requested:
# - IDs 1..6 on can0, IDs 7..12 on can1
# - hipy is ID 3 (left) / 9 (right)
# - knee is ID 4 (left) / 10 (right)
LEFT_IDS = (1, 2, 3, 4, 5, 6)
RIGHT_IDS = (7, 8, 9, 10, 11, 12)

# Per-axis calibration knobs (edit these in IPython while checking coherence).
MOTOR_SIGN = {mid: 1.0 for mid in range(1, 13)}
MOTOR_OFFSET_DEG = {mid: 0.0 for mid in range(1, 13)}
ANKLE_PITCH_SIGN = -1.0
ANKLE_PITCH_OFFSET_DEG = 0.0
ANKLE_ROLL_SIGN = 1.0
ANKLE_ROLL_OFFSET_DEG = 0.0


# sign still unknown -> keep + first, then flip if needed
OFF_HIPZ = 132.68
OFF_HIPX = 19.394
OFF_HIPY = 88.096
OFF_KNEE = 57.352

# left leg (1..6): [hipz, hipx, hipy, knee, ankle1, ankle2]
MOTOR_OFFSET_DEG[1] = OFF_HIPZ
MOTOR_OFFSET_DEG[2] = OFF_HIPX
MOTOR_OFFSET_DEG[3] = OFF_HIPY
MOTOR_OFFSET_DEG[4] = OFF_KNEE

# right leg (7..12): [hipz, hipx, hipy, knee, ankle1, ankle2]
MOTOR_OFFSET_DEG[7]  = -OFF_HIPZ
MOTOR_OFFSET_DEG[8]  = -OFF_HIPX
MOTOR_OFFSET_DEG[9]  = -OFF_HIPY
MOTOR_OFFSET_DEG[10] = OFF_KNEE


MOTOR_SIGN[4] = -1.0


MOTOR_SIGN[5] = -1.0
MOTOR_SIGN[6] = -1.0



MOTOR_SIGN[11] = -1.0
MOTOR_SIGN[12] = -1.0

def _corrected_motor_deg(state_by_id, motor_id):
    st = state_by_id.get(motor_id, MotorState())
    return MOTOR_SIGN[motor_id] * float(st.position_deg) + MOTOR_OFFSET_DEG[motor_id]


def _fill_leg_q(q, base_idx, ids, state_by_id):
    # Per-leg ID order: [hipz, hipx, hipy, knee, ankle1, ankle2]
    hipz = _corrected_motor_deg(state_by_id, ids[0])
    hipx = _corrected_motor_deg(state_by_id, ids[1])
    hipy = _corrected_motor_deg(state_by_id, ids[2])
    knee = _corrected_motor_deg(state_by_id, ids[3])
    ankle1 = _corrected_motor_deg(state_by_id, ids[4])
    ankle2 = _corrected_motor_deg(state_by_id, ids[5])

    q[base_idx + 0] = np.deg2rad(hipz)
    q[base_idx + 1] = np.deg2rad(hipx)
    q[base_idx + 2] = np.deg2rad(hipy)
    q[base_idx + 3] = np.deg2rad(knee)
    q[base_idx + 4] = np.deg2rad(ANKLE_PITCH_SIGN * ((ankle1 - ankle2) / 2.0) + ANKLE_PITCH_OFFSET_DEG)
    q[base_idx + 5] = np.deg2rad(ANKLE_ROLL_SIGN * ((ankle1 + ankle2) / 2.0) + ANKLE_ROLL_OFFSET_DEG)


def state_to_model_q(state_by_id, nq):
    # Model q layout used here: left leg first, then right leg.
    q = np.zeros(max(12, int(nq)))
    _fill_leg_q(q, 0, LEFT_IDS, state_by_id)
    _fill_leg_q(q, 6, RIGHT_IDS, state_by_id)
    return q[:nq]


def request_state_once_fast(bus_can0, bus_can1, state_by_id, drain_s=0.002):
    # Bus routing (explicit):
    # - IDs 1..6  -> can0
    # - IDs 7..12 -> can1
    for mid in LEFT_IDS:
        bus_can0.send(can.Message(arbitration_id=mid, data=[0xFF] * 7 + [CAN_CMD_CLEAR_FAULT], is_extended_id=False))
    for mid in RIGHT_IDS:
        bus_can1.send(can.Message(arbitration_id=mid, data=[0xFF] * 7 + [CAN_CMD_CLEAR_FAULT], is_extended_id=False))

    t_end = time.perf_counter() + float(drain_s)
    while time.perf_counter() < t_end:
        got = False
        for bus_ in (bus_can0, bus_can1):
            msg = bus_.recv(0.0)
            if msg is None:
                continue
            got = True
            raw = bytes(msg.data)
            if len(raw) < 8:
                continue
            mid = int(raw[0])
            spec = MOTORS.get(mid)
            if spec is None:
                continue
            _, st = decode_state_frame(raw, pmax=spec.pmax_rad, vmax=spec.vmax_rad_s, tmax=spec.tmax_nm)
            state_by_id[mid] = st
        if not got:
            # keep loop hot but avoid burning full CPU when there is no traffic
            time.sleep(0.0001)


def run_fast_state_viz_loop(bus_can0, bus_can1, viz, model, duration_s=10.0):
    state_by_id = {mid: MotorState() for mid in range(1, 13)}
    t0 = time.time()
    n_iter = 0
    while time.time() - t0 < float(duration_s):
        request_state_once_fast(bus_can0, bus_can1, state_by_id, drain_s=0.002)
        q = state_to_model_q(state_by_id, model.nq)
        viz.display(q)
        n_iter += 1
    dt = time.time() - t0
    print(f"Loop done: {n_iter} iterations in {dt:.2f}s ({n_iter / dt:.1f} Hz)")
    return state_by_id


# Copy-paste entrypoint:
# final_state = run_fast_state_viz_loop(bus1, bus2, viz, robot_wrapper.model, duration_s=10.0)
state_by_id = {mid: MotorState() for mid in range(1, 13)}
state_extrema = {
    mid: {"min_pos_deg": float("inf"), "max_pos_deg": float("-inf")}
    for mid in range(1, 13)
}

t0 = time.time()
n_iter = 0
while time.time() - t0 < 100.0:
    request_state_once_fast(bus1, bus2, state_by_id, drain_s=0.01)
    for mid, st in state_by_id.items():
        p = float(st.position_deg)
        if p < state_extrema[mid]["min_pos_deg"]:
            state_extrema[mid]["min_pos_deg"] = p
        if p > state_extrema[mid]["max_pos_deg"]:
            state_extrema[mid]["max_pos_deg"] = p
    q = state_to_model_q(state_by_id, robot_wrapper.model.nq)
    viz.display(q)
    n_iter += 1

dt = time.time() - t0
print(f"Loop done: {n_iter} iterations in {dt:.2f}s ({n_iter/dt:.1f} Hz)")
print("State extrema (raw motor deg):")
for mid in range(1, 13):
    lo = state_extrema[mid]["min_pos_deg"]
    hi = state_extrema[mid]["max_pos_deg"]
    print(f"  m{mid}: min={lo:.3f}, max={hi:.3f}")



# State extrema (raw motor deg):
#   m1: min=150.877, max=312.315  ->-360
#   m2: min=230.113, max=372.934 -> -360°
#   m3: min=191.935, max=361.967 -> -360°
#   m4: min=0.978, max=112.172 -> good
#   m5: min=-89.753, max=62.542  -> good
#   m6: min=-11.638, max=285.830 -> good
#   m7: min=64.894, max=215.255  -> good
#   m8: min=304.689, max=400.409  > -360°
#   m9: min=-0.055, max=162.438  -> good
#   m10: min=261.895, max=359.307 > -360°
#   m11: min=-7.440, max=87.972 -> good
#   m12: min=-78.191, max=352.516 -> good




m01 | req_cal=  +0.000 deg | tgt_cal=  +0.000 deg | send_raw=-132.680 deg
m02 | req_cal=  +0.000 deg | tgt_cal=  +0.000 deg | send_raw= -19.394 deg
m03 | req_cal=  +0.000 deg | tgt_cal=  +0.000 deg | send_raw= -88.096 deg
m04 | req_cal=  +0.000 deg | tgt_cal=  +0.000 deg | send_raw= +57.352 deg
m05 | req_cal=  +0.000 deg | tgt_cal=  +0.000 deg | send_raw=  -0.000 deg
m06 | req_cal=  +0.000 deg | tgt_cal=  +0.000 deg | send_raw=  -0.000 deg
m07 | req_cal=  +0.000 deg | tgt_cal=  +0.000 deg | send_raw=+132.680 deg
m08 | req_cal=  +0.000 deg | tgt_cal=  +0.000 deg | send_raw= +19.394 deg
m09 | req_cal=  +0.000 deg | tgt_cal=  +0.000 deg | send_raw= +88.096 deg
m10 | req_cal=  +0.000 deg | tgt_cal=  +0.000 deg | send_raw= -57.352 deg
m11 | req_cal=  +0.000 deg | tgt_cal=  +0.000 deg | send_raw=  -0.000 deg
m12 | req_cal=  +0.000 deg | tgt_cal=  +0.000 deg | send_raw=  -0.000 deg

m01 | raw=+229.893 deg | stamp=1770893488.877
m02 | raw=+334.207 deg | stamp=1770893488.876
m03 | raw=+256.466 deg | stamp=1770893488.876
m04 | raw= +43.904 deg | stamp=1770893488.876
m05 | raw=+350.713 deg | stamp=1770893488.877
m06 | raw= +18.671 deg | stamp=1770893488.875
m07 | raw=+136.019 deg | stamp=1770893488.875
m08 | raw= +11.550 deg | stamp=1770893488.875
m09 | raw=+108.039 deg | stamp=1770893488.875
m10 | raw=+316.140 deg | stamp=1770893488.877
m11 | raw= +11.550 deg | stamp=1770893488.876
m12 | raw=  -9.198 deg | stamp=1770893488.877
