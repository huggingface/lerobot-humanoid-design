for mid in [1,7]:
    robot.set_joint_gains(mid, kp=40, kd=4.0) 

for mid in [5,6,11,12]:
    robot.set_joint_gains(mid, kp=40, kd=1.25) 


robot.set_joint_gains(2, kp=40, kd=4.0) 
robot.set_joint_gains(3, kp=120/10, kd=4.0/10) 
robot.set_joint_gains(4, kp=120/10, kd=4.0/10) 
robot.set_joint_gains(8, kp=40, kd=4.0) 
robot.set_joint_gains(9, kp=120/10, kd=4.0/10)
robot.set_joint_gains(10, kp=120/10, kd=4.0/10)