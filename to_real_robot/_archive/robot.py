import can
import numpy as np
import time
import pinocchio as pin
from example_parallel_robots.loader_tools import completeRobotLoader
import os
import meshcat
from pinocchio.visualize import MeshcatVisualizer
CWD=os.getcwd()
# CWD = os.path.dirname(os.path.abspath(__file__))

pin.SE3.__repr__ = pin.SE3.__str__


CAN_CMD_CLEAR_FAULT = 0xFB
CAN_CMD_ZERO = 0xFE





MOTOR_NAME={1:"hipz",
            2:"hipx",
            4:"hipy",
            3:"knee",
            5:"ankle1",
            6:"ankle2"}


MOTOR_LIMIT_PARAMS={1:(12.57, 33, 14),
                    2:(12.57, 33, 20),
                    3:(12.57, 33, 60),
                    4:(12.57, 33, 60),
                    5:(12.57, 50, 5.5),
                    6:(12.57, 50, 5.5)}

ROBOT_STATE={1:(0, 0, 0),
             2:(0, 0, 0),
             3:(0, 0, 0),
             4:(0, 0, 0),
             5:(0, 0, 0),
             6:(0, 0, 0)}



def _float_to_uint( x: float, x_min: float, x_max: float, bits: int) -> int:
    """Convert float to unsigned integer for CAN transmission."""
    x = max(x_min, min(x_max, x))  # Clamp to range
    span = x_max - x_min
    data_norm = (x - x_min) / span
    return int(data_norm * ((1 << bits) - 1))

def _uint_to_float(x: int, x_min: float, x_max: float, bits: int) -> float:
    """Convert unsigned integer from CAN to float."""
    span = x_max - x_min
    data_norm = float(x) / ((1 << bits) - 1)
    return data_norm * span + x_min




def _decode_motor_state( data: bytes) -> tuple[float, float, float, float]:
    """
    Decode motor state from CAN data.

    Returns:
        Tuple of (position_degrees, velocity_deg_per_sec, torque, temp_mos)
    """
    if len(data) < 8:
        raise ValueError("Invalid motor state data")

    # Extract encoded values
    motor_id = data[0]
    motor_name = MOTOR_NAME[motor_id]
    q_uint = (data[1] << 8) | data[2]
    dq_uint = (data[3] << 4) | (data[4] >> 4)
    tau_uint = ((data[4] & 0x0F) << 8) | data[5]
    t_mos = (data[6] << 8) | data[7]

    # Get motor limits
    pmax, vmax, tmax = MOTOR_LIMIT_PARAMS[motor_id]

    # Decode to physical values (radians)
    position_rad = _uint_to_float(q_uint, -pmax, pmax, 16)
    velocity_rad_per_sec = _uint_to_float(dq_uint, -vmax, vmax, 12)
    torque = _uint_to_float(tau_uint, -tmax, tmax, 12)

    # Convert to degrees
    position_degrees = np.degrees(position_rad)
    velocity_deg_per_sec = np.degrees(velocity_rad_per_sec)

    # Update cached state
    return position_degrees, velocity_deg_per_sec, torque, t_mos / 10





def enable(bus: can.Bus, motor_id: int) -> bool:
    """
    Enable a motor using the MIT protocol.
    """

    arb_id = motor_id

    data = [0xFF] * 8
    data[7] = 0xFC  # Enable command

    msg = can.Message(arbitration_id=arb_id, data=data, is_extended_id=False)

    bus.send(msg)
    msg = bus.recv(0.02)
    return bool(msg)


def disable(bus: can.Bus, motor_id: int) -> bool:
    """
    Disable a motor using the MIT protocol.
    """

    arb_id = motor_id

    data = [0xFF] * 8
    data[7] = 0xFD  # Disable command

    msg = can.Message(arbitration_id=arb_id, data=data, is_extended_id=False)

    bus.send(msg)
    msg = bus.recv(0.02)
    return bool(msg)






def get_robot_state(bus):
    motor_ids=[1,2,3,4,5,6]
    for motor_id in motor_ids :
        data = [0xFF] * 7 + [CAN_CMD_CLEAR_FAULT]
        msg = can.Message(arbitration_id=motor_id, data=data, is_extended_id=False)
        bus.send(msg)
    
    recvs_message=[]
    msg=True
    time.sleep(0.0025)
    while msg:
        msg=bus.recv(0.001)
        if msg:
            recvs_message.append(msg)

    missing_id = []
    if len(recvs_message) < 6 :
        missing_id=[1,2,3,4,5,6]
        for msg in recvs_message:
            recvd_id=msg.data[0]
            try:
                missing_id.remove(recvd_id)
            except:
                pass
        # print(missing_id)
    for msg in recvs_message:
        try:
            pos,speed,torque,temp = _decode_motor_state(msg.data)
            motor_id=msg.data[0]
            ROBOT_STATE[motor_id]=(pos,speed,torque)
        except:
            print("oups")
    return(ROBOT_STATE,missing_id)

def set_zero(bus,motor_id):
        data = [0xFF] * 7 + [CAN_CMD_ZERO]
        msg = can.Message(arbitration_id=motor_id, data=data, is_extended_id=False)
        bus.send(msg)
        print(bus.recv(0.5))

def convert_state_to_meshcat(state):
    q=np.zeros(6)

    position,_,_=ROBOT_STATE[1]
    q[0]=np.deg2rad(position)

    position,_,_=ROBOT_STATE[2]
    q[1]=np.deg2rad(position)

    position,_,_=ROBOT_STATE[3]
    q[3]=np.deg2rad(position)

    position,_,_=ROBOT_STATE[4]
    q[2]=np.deg2rad(position)

    position1,_,_=ROBOT_STATE[5]
    position2,_,_=ROBOT_STATE[6]
    q[4]=-np.deg2rad((position1-position2)/2)
    q[5]=np.deg2rad((position1+position2)/2)
    return(q)

def mit_control(
    bus: can.Bus,
    motor_id: int,
    kp: float,
    kd: float,
    position_degrees: float,
    velocity_deg_per_sec: float,
    torque: float,
    pmax: float = 12.57,
    vmax: float = 50,
    tmax: float = 6,
) :
    """
    Send a MIT-style position/velocity/torque command to a motor.

    Args:
        motor_id: Target motor ID
        kp: Position gain
        kd: Velocity gain
        position_degrees: Target position (degrees)
        velocity_deg_per_sec: Target velocity (degrees/s)
        torque: Target torque (N·m)
    """
    position_rad = np.radians(position_degrees)
    velocity_rad_per_sec = np.radians(velocity_deg_per_sec)

    kp_uint = _float_to_uint(kp, 0, 500, 12)
    kd_uint = _float_to_uint(kd, 0, 5, 12)
    q_uint = _float_to_uint(position_rad, -pmax, pmax, 16)
    dq_uint = _float_to_uint(velocity_rad_per_sec, -vmax, vmax, 12)
    tau_uint = _float_to_uint(torque, -tmax, tmax, 12)

    # Pack data
    data = [0] * 8
    data[0] = (q_uint >> 8) & 0xFF
    data[1] = q_uint & 0xFF
    data[2] = dq_uint >> 4
    data[3] = ((dq_uint & 0xF) << 4) | ((kp_uint >> 8) & 0xF)
    data[4] = kp_uint & 0xFF
    data[5] = kd_uint >> 4
    data[6] = ((kd_uint & 0xF) << 4) | ((tau_uint >> 8) & 0xF)
    data[7] = tau_uint & 0xFF

    msg = can.Message(arbitration_id=motor_id, data=data, is_extended_id=False)
    bus.send(msg)
    msg = bus.recv(0.02)
    return msg




model,constraint_models,actuation_model,visual_model,colision_model=completeRobotLoader(CWD + '/urdf/humanoid_v1/urdf', freeflyer=False)
# rotate=pin.utils.rotate("y",np.deg2rad(180))
# model.jointPlacements[0].rotation=model.jointPlacements[0].rotation@rotate

viz = MeshcatVisualizer(model, visual_model, visual_model)
viz.viewer = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
viz.clean()
viz.loadViewerModel(rootNodeName="universe")
viz.display(pin.neutral(model))

bus = can.interface.Bus(interface="socketcan", channel="can0")


motor_id= 3
data = [0xFF] * 7 + [CAN_CMD_CLEAR_FAULT]
msg = can.Message(arbitration_id=motor_id, data=data, is_extended_id=False)
bus.send(msg)
recved_msg=bus.recv(1)
# print(recved_msg)




import threading, time

latest_q = None
lock = threading.Lock()
stop = False

def viz_loop():
    global latest_q
    period = 1.0 / 30.0
    while not stop:
        with lock:
            q = latest_q
        if q is not None:
            viz.display(q)
        time.sleep(period)

threading.Thread(target=viz_loop, daemon=True).start()


### viz check 

t_ini=time.time()
i=0
old_missing_id=[]
while time.time()-t_ini < 20:
    state,mising_id=get_robot_state(bus)
    q=convert_state_to_meshcat(state)
    i=i+1
    miss=False
    for old in old_missing_id:
        if old in mising_id:
            miss=True
    if miss:
        print(mising_id)
    old_missing_id = mising_id.copy()

    with lock:
        latest_q = q



### ankle limit check

t_ini=time.time()
i=0
old_missing_id=[]
lim_ankle1=[-70, -2]
lim_ankle2=[-1, 80]


while time.time()-t_ini < 10:
    state,mising_id=get_robot_state(bus)
    q_ankle1=state[5][0]
    q_ankle2=state[6][0]
    if q_ankle1 > lim_ankle1[1]:
        lim_ankle1[1]=q_ankle1
    if q_ankle1 < lim_ankle1[0]:
        lim_ankle1[0]=q_ankle1

    if q_ankle2 > lim_ankle2[1]:
        lim_ankle2[1]=q_ankle2
    if q_ankle2 < lim_ankle2[0]:
        lim_ankle2[0]=q_ankle2

    
true_limit_ankle1= [-70, -5]

true_limit_ankle2= [5, 70]

enable(bus,5)
enable(bus,6)


pos_ankle1=6
pos_ankle2=-6
t_ini=time.time()
i=0
descente=False
while t_ini +10>time.time():
    pos_ankle1=-5-65/2-65/2*np.sin(time.time())
    pos_ankle2=+5+65/2+65/2*np.sin(time.time())
    if true_limit_ankle1[0]<pos_ankle1 and pos_ankle1<true_limit_ankle1[1]:
        mit_control(bus,5,10,0.5,pos_ankle1,0,0)
        # print("la")

    if true_limit_ankle2[0]<pos_ankle2 and pos_ankle2<true_limit_ankle2[1]:
        mit_control(bus,6,10,0.5,pos_ankle2,0,0)
    time.sleep(0.01)


disable(bus,5)
disable(bus,6)









#knee limit 


t_ini=time.time()
i=0
old_missing_id=[]
lim_knee=[-112.63311610537617, -19.836352285120917]


while time.time()-t_ini < 10:
    state,mising_id=get_robot_state(bus)
    q_knee=state[3][0]
    if q_knee > lim_knee[1]:
        lim_knee[1]=q_knee
    if q_knee < lim_knee[0]:
        lim_knee[0]=q_knee






true_lim_knee=[-100, -20]

enable(bus,3)


pos_knee=-20
mit_control(bus,3,1,0.1,pos_knee,0,0,12.57,33,60)
time.sleep(0.5)

t_ini=time.time()
while t_ini +10>time.time():
    pos_knee=-20-80/2-80/2*np.sin(time.time())
    if true_lim_knee[0]<pos_knee and pos_knee<true_lim_knee[1]:
        mit_control(bus,3,1,0.1,pos_knee,0,0,12.57,33,60)
    time.sleep(0.01)


disable(bus,3)



### hip y lim



t_ini=time.time()
i=0
old_missing_id=[]
lim_hipy=[-148.1737051857535, -0.032969006568082104]
# lim_hipy=[1000, -10000]



while time.time()-t_ini < 10:
    state,mising_id=get_robot_state(bus)
    q_hipy=state[4][0]
    if q_hipy > lim_hipy[1]:
        lim_hipy[1]=q_hipy
    if q_hipy < lim_hipy[0]:
        lim_hipy[0]=q_hipy




true_limit_hipy = [-130,-15]


enable(bus,4)


pos_hipy=-120
mit_control(bus,4,1,0.1,pos_hipy,0,0,12.57,33,60)
time.sleep(0.5)
t_ini=time.time()
print(np.sin(time.time()-t_ini))
descente=False
while t_ini +10>time.time():
    pos_hipy=-85-90/2*np.sin(time.time()-t_ini)   
    if true_limit_hipy[0]<pos_hipy and pos_hipy<true_limit_hipy[1]:
        mit_control(bus,4,1,0.1,pos_hipy,0,0,12.57,33,60)
        # print("la")
    

disable(bus,4)




### hip x lim

t_ini=time.time()
i=0
old_missing_id=[]
lim_hipx=[20.63311610537617, 180]


while time.time()-t_ini < 10:
    state,mising_id=get_robot_state(bus)
    q_hipx=state[2][0]
    if q_hipx > lim_hipx[1]:
        lim_hipx[1]=q_hipx
    if q_hipx < lim_hipx[0]:
        lim_hipx[0]=q_hipx


true_limit_hipx = [20,160]



enable(bus,2)


pos_hipx=90
mit_control(bus,2,1,0.1,pos_hipx,0,0,12.57,33,20)
time.sleep(0.5)
t_ini=time.time()
print(np.sin(time.time()-t_ini))
descente=False
while t_ini +10>time.time():
    pos_hipx=90-80/2*np.sin(time.time()-t_ini)   
    if true_limit_hipx[0]<pos_hipx and pos_hipx<true_limit_hipx[1]:
        mit_control(bus,2,5,0.5,pos_hipx,0,0,12.57,33,20)
        # print("la")
    

disable(bus,2)

### hip z lim

t_ini=time.time()
i=0
old_missing_id=[]
lim_hipz=[0, 200]


while time.time()-t_ini < 10:
    state,mising_id=get_robot_state(bus)
    q_hipz=state[1][0]
    if q_hipz > lim_hipz[1]:
        lim_hipz[1]=q_hipz
    if q_hipz < lim_hipz[0]:
        lim_hipz[0]=q_hipz


true_limit_hipz = [20,160]
enable(bus,1)


pos_hipz=90
mit_control(bus,1,1,0.1,pos_hipz,0,0,12.57,33,20)
time.sleep(0.5)
t_ini=time.time()
print(np.sin(time.time()-t_ini))
while t_ini +10>time.time():
    pos_hipz=90-80/2*np.sin(time.time()-t_ini)   
    if true_limit_hipz[0]<pos_hipz and pos_hipz<true_limit_hipz[1]:
        mit_control(bus,1,1,0.1,pos_hipz,0,0,12.57,33,14)
        # print("la")
    

for i in range(1,7):
    disable(bus,i)







for i in range(1,7):
    enable(bus,i)


pos_ankle1=-45
pos_ankle2=45
pos_knee=-80
pos_hipy=-90
pos_hipx=90
pos_hipz=90

mit_control(bus,1,5,0.1,pos_hipz,0,0,12.57,33,14)
mit_control(bus,2,5,0.1,pos_hipx,0,0,12.57,33,20)
mit_control(bus,3,5,0.5,pos_knee,0,0,12.57,33,60)
mit_control(bus,4,5,0.5,pos_hipy,0,0,12.57,33,60)
mit_control(bus,5,5,0.5,pos_ankle1,0,0,12.57,50,5.5)
mit_control(bus,6,5,0.5,pos_ankle2,0,0,12.57,50,5.5)


t_ini=time.time()
while t_ini +20>time.time():
    t=time.time()-t_ini
    print(np.sin(2*np.pi*0.2*t))
    pos_ankle1=-45-30*np.sin(2*np.pi*1*t)
    pos_ankle2=45+30*np.sin(2*np.pi*1*t)
    pos_knee=-80-30*np.sin(2*np.pi*0.5*t)
    pos_hipy=-90-20*np.sin(2*np.pi*0.2*t)
    pos_hipx=90+45*np.sin(2*np.pi*0.3*t)
    pos_hipz=90+45*np.sin(2*np.pi*0.2*t)



    mit_control(bus,1,8,0.5,pos_hipz,0,0,12.57,33,14)
    mit_control(bus,2,5,0.1,pos_hipx,0,0,12.57,33,20)
    mit_control(bus,3,5,0.5,pos_knee,0,0,12.57,33,60)
    mit_control(bus,4,5,0.5,pos_hipy,0,0,12.57,33,60)
    mit_control(bus,5,5,0.5,pos_ankle1,0,0,12.57,50,5.5)
    mit_control(bus,6,5,0.5,pos_ankle2,0,0,12.57,50,5.5)