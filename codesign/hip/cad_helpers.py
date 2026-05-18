import numpy as np


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
    y = np.arcsin(sy)                # rotation about Y
    x = np.arctan2(R[2, 1], R[2, 2]) # rotation about X
    z = np.arctan2(R[1, 0], R[0, 0]) # rotation about Z

    return x, y, z
