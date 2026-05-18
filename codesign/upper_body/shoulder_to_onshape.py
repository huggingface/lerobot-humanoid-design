
from codesign.upper_body.arm_evaluation import ArmEvaluator
import numpy as np
import pinocchio as pin

def rotation_matrix_to_euler_xyz(R):
    """
    Convert a 3x3 rotation matrix to Euler angles (XYZ order).

    Convention:
        - Right-handed coordinate system
        - Active rotations on column vectors
        - Overall rotation: R = Rz(z) @ Ry(y) @ Rx(x)
        - Returns angles (x, y, z) in radians

    Parameters
    ----------
    R : array_like, shape (3, 3)
        Rotation matrix.

    Returns
    -------
    x, y, z : float
        Euler angles around X, Y, Z axes (in radians).
    """
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        raise ValueError("R must be a 3x3 matrix")

    # Protect against numerical issues:
    # For this convention, sin(y) = -R[2,0]
    sy = -R[2, 0]
    sy = np.clip(sy, -1.0, 1.0)

    # Compute angles
    y = np.arcsin(sy)                  # rotation about Y
    x = np.arctan2(R[2, 1], R[2, 2])   # rotation about X
    z = np.arctan2(R[1, 0], R[0, 0])   # rotation about Z

    return x, y, z

evaluate=ArmEvaluator()
dx=np.array([ 45.9095, 137.2909, -81.4492, -30.54  ,  -9.0298,  37.7497])
evaluate.evaluate(dx) #modify the robot
model=evaluate.model
data=model.createData()
q0=pin.neutral(model)
pin.framesForwardKinematics(model,data,q0)


torso_id=model.getFrameId("torso")
oMtorso=data.oMf[torso_id]
oMshoulder1=data.oMi[2]
oMshoulder2=data.oMi[3]
oMshoulder3=data.oMi[4]
oMelbow=data.oMi[5]

torsoMshoulder1=oMtorso.inverse()*oMshoulder1
shoulder1Mshoulder2=oMshoulder1.inverse()*oMshoulder2
shoulder2Mshoulder3=oMshoulder2.inverse()*oMshoulder3
shoulder3Melbow=oMshoulder3.inverse()*oMelbow

T1=torsoMshoulder1.translation
euler1= np.rad2deg(rotation_matrix_to_euler_xyz(torsoMshoulder1.rotation))

T2 = shoulder1Mshoulder2.translation
euler2= np.rad2deg(rotation_matrix_to_euler_xyz(shoulder1Mshoulder2.rotation))

T3 = shoulder2Mshoulder3.translation
euler3= np.rad2deg(rotation_matrix_to_euler_xyz(shoulder2Mshoulder3.rotation))




T4=shoulder3Melbow.translation
euler4 = np.rad2deg(rotation_matrix_to_euler_xyz(shoulder3Melbow.rotation))



RT4 = shoulder3Melbow.inverse().translation
Reuler4 =np.rad2deg(rotation_matrix_to_euler_xyz(shoulder3Melbow.inverse().rotation))
