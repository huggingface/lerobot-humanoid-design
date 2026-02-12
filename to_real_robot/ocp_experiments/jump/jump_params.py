import numpy as np

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

    DT = 0.015
    TStand = 10
    TPush = 10
    Tstart = TStand + TPush

    TFlyUp = 13
    TFlyDown = TFlyUp
    TFly = TFlyUp + TFlyDown

    TLand = 10
    Tend = 10

    contactPattern = (
        [[1, 1]] * (TStand + TPush)
        + [[0, 0]] * (TFlyUp + TFlyDown)
        + [[1, 1]] * (TLand + Tend)
    )
    Ttotal = len(contactPattern)
    TMid = Tstart + TFlyUp

    v0 = 9.81 * TFlyUp * DT
    href = v0 * DT * TFlyUp / 2.0

    comRefTrajWeight = 1e5
    comRefTraj = [np.array([1.0, 1.0, 1.0]) for _ in range(Ttotal)]
    comRefTrajImportance = [np.array([0.0, 0.0, 0.0]) for _ in range(Ttotal)]
    comRefTraj[TMid] = np.array([1.0, 1.0, href])
    comRefTrajImportance[TMid] = np.array([0.0, 0.0, 1.0])

    footSize = 0.05
    feetCollisionWeight = 0

    impactAltitudeWeight = 1e5
    impactRotationWeight = 1e3
    impactVelocityWeight = 1e3
    refMainJointsAtImpactWeight = 0

    refStateWeight = 0.7
    refTorqueWeight = 0.05
    stateTerminalWeight = 1e3
    refForceWeight = 500
    copWeight = 5

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
        leg_q_weights = [1, 1, 1, 1, 1, 1]
        basis_v_weights = [0, 0, 0, 3, 3, 1]
        leg_v_weights = [1, 1, 1, 1, 1, 1]

        self.stateImportance = np.array(
            basis_q_weights + leg_q_weights * 2 + basis_v_weights + leg_v_weights * 2
        )
        nv = len(basis_v_weights) + 2 * len(leg_v_weights)
        self.stateTerminalImportance = np.array(
            [0, 0, 10, 0, 0, 50] + [1] * (nv - 6) + [1] * nv
        )
        self.controlImportance = np.array([1] * 12)

    def getReferenceForces(self, grav: float, com0) -> list[np.ndarray]:
        alpha_push = 1 + self.v0 / (9.81 * self.TPush * self.DT)
        fpush = alpha_push / 2.0 * grav

        alpha_land = 1 + self.v0 / (9.81 * self.TLand * self.DT)
        fland = alpha_land / 2.0 * grav

        reference_forces = [np.array([0.5, 0.5]) * grav for _ in range(self.TStand)]
        reference_forces += [np.array([fpush, fpush]) for _ in range(self.TPush)]
        reference_forces += [np.array([0.0, 0.0]) for _ in range(self.TFly)]
        reference_forces += [np.array([fland, fland]) for _ in range(self.TLand)]
        reference_forces += [np.array([0.5, 0.5]) * grav for _ in range(self.Tend)]

        self.referenceForces = reference_forces
        return reference_forces
