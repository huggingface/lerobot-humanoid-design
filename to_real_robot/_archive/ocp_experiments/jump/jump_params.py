import numpy as np

try:
    from .params_base import ParamsBase
except ImportError:
    from params_base import ParamsBase


class JumpRealRobotParams(ParamsBase):
    mainJointsIds = [
        "hipz_right",
        "hipy_right",
        "hipx_right",
        "knee_right",
        "hipz_left",
        "hipy_left",
        "hipx_left",
        "knee_left",
    ]

    # Squat-first profile (no real flight), used to tune in-place motion.
    DT = 0.005
    TStand = 1
    TPush = 20
    Tstart = TStand + TPush

    # Keep a tiny "fly" duration to stay compatible with jump pipeline.
    TFlyUp = 20
    TFlyDown = TFlyUp
    TFly = TFlyUp + TFlyDown

    TLand = TPush
    Tend = 1

    contactPattern = (
        [[1, 1]] * (TStand + TPush)
        + [[0, 0]] * (TFlyUp + TFlyDown)
        + [[1, 1]] * (TLand + Tend)
    )
    Ttotal = len(contactPattern)
    TMid = Tstart + TFlyUp

    v0 = 0.0
    href = 0.0

    comRefTrajWeight = 3e5
    comRefTraj = [np.array([1.0, 1.0, 1.0]) for _ in range(Ttotal)]
    comRefTrajImportance = [np.array([0.0, 0.0, 0.0]) for _ in range(Ttotal)]
    # Upward-only CoM motion (z-axis only), never below nominal 1.0 ratio.
    rise_z = 1.15
    for t in range(TStand, Tstart):
        alpha = (t - TStand) / max(1, (TPush - 1))
        z_ref = (1.0 - alpha) * 1.0 + alpha * rise_z
        comRefTraj[t] = np.array([1.0, 1.0, z_ref])
        comRefTrajImportance[t] = np.array([0.0, 0.0, 1.0])
    for t in range(Tstart, Tstart + TFly):
        comRefTraj[t] = np.array([1.0, 1.0, rise_z])
        comRefTrajImportance[t] = np.array([0.0, 0.0, 1.0])
    for t in range(Tstart + TFly, Tstart + TFly + TLand):
        alpha = (t - (Tstart + TFly)) / max(1, (TLand - 1))
        z_ref = (1.0 - alpha) * rise_z + alpha * 1.0
        comRefTraj[t] = np.array([1.0, 1.0, z_ref])
        comRefTrajImportance[t] = np.array([0.0, 0.0, 1.0])

    footSize = 0.05
    feetCollisionWeight = 1000

    impactAltitudeWeight = 0
    impactRotationWeight = 0
    impactVelocityWeight = 0
    refMainJointsAtImpactWeight = 0

    refStateWeight = 0.7
    refTorqueWeight = 0.05
    stateTerminalWeight = 200
    refForceWeight = 200
    copWeight = 1000

    transitionDuration = 4
    solver_th_stop = 1e-4
    solver_maxiter = 200
    solver_reg_min = 1e-6

    saveFile = "/tmp/real_robot_jump.npy"
    guessFile = None
    preview = True
    save = False

    def __init__(self) -> None:
        basis_q_weights = [0, 0, 0, 50, 50, 0]
        leg_q_weights = [10, 10, 1, 1, 1, 1]
        basis_v_weights = [0, 0, 0, 3, 3, 1]
        leg_v_weights = [5, 5, 2, 2, 1, 1]

        self.stateImportance = np.array(
            basis_q_weights + leg_q_weights * 2 + basis_v_weights + leg_v_weights * 2
        )
        nv = len(basis_v_weights) + 2 * len(leg_v_weights)
        self.stateTerminalImportance = np.array(
            [10, 10, 10, 10, 10, 10] + [10] * (nv - 6) + [1] * nv
        )
        # Stronger torque minimization on ankle joints.
        # Order follows robot.yaml motors:
        # [hipz_l, hipy_l, hipx_l, knee_l, anklex_l, ankley_l,
        #  hipz_r, hipy_r, hipx_r, knee_r, anklex_r, ankley_r]
        self.controlImportance = np.array([1, 1, 1, 1, 4, 4, 1, 1, 1, 1, 4, 4])

    def getReferenceForces(self, grav: float, com0) -> list[np.ndarray]:
        # In squat mode, keep nominal support forces on both feet the whole time.
        reference_forces = [np.array([0.5, 0.5]) * grav for _ in range(self.Ttotal)]

        self.referenceForces = reference_forces
        return reference_forces
