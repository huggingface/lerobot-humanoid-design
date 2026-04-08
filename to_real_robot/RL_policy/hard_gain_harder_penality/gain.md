# Base actuator gains (no additional scaling).
BASE_KP_HIPZ = 20.0
BASE_KV_HIPZ = 2.0

BASE_KP_HIPX = 40.0
BASE_KV_HIPX = 2.0

BASE_KP_HIP = 200.0
BASE_KV_HIP = 10.0

BASE_KP_KNEE = 200.0
BASE_KV_KNEE = 10.0

BASE_KP_ANKLE = 30.0
BASE_KV_ANKLE = 1.0




for mid in [1,7]:
    robot.set_joint_gains(mid, kp=10, kd=0.5) 

for mid in [5,6,11,12]:
    robot.set_joint_gains(mid, kp=15, kd=0.5) 


robot.set_joint_gains(2, kp=20, kd=0.5) 
robot.set_joint_gains(3, kp=20, kd=0.5) 
robot.set_joint_gains(4, kp=20, kd=0.5) 
robot.set_joint_gains(8, kp=20, kd=0.5) 
robot.set_joint_gains(9, kp=20, kd=0.5)
robot.set_joint_gains(10, kp=20, kd=0.5)
