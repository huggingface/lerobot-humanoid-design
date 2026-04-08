# Base actuator gains (no additional scaling).
BASE_KP_HIPZ = 80.0
BASE_KV_HIPZ = 4.0

BASE_KP_HIPX = 80.0
BASE_KV_HIPX = 4.0

BASE_KP_HIP = 100.0
BASE_KV_HIP = 5.0

BASE_KP_KNEE = 100.0
BASE_KV_KNEE = 5.0

BASE_KP_ANKLE = 30.0
BASE_KV_ANKLE = 1.0




for mid in [1,7]:
    robot.set_joint_gains(mid, kp=80, kd=4.0) 

for mid in [5,6,11,12]:
    robot.set_joint_gains(mid, kp=15, kd=0.5) 


robot.set_joint_gains(2, kp=80, kd=4.0) 
robot.set_joint_gains(3, kp=10, kd=0.25) 
robot.set_joint_gains(4, kp=10, kd=0.25) 
robot.set_joint_gains(8, kp=80, kd=4.0) 
robot.set_joint_gains(9, kp=10, kd=0.25)
robot.set_joint_gains(10, kp=10, kd=0.25)
