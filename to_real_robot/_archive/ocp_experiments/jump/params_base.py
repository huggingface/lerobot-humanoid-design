import numpy as np


def roundToOdd(x):
    """
    Round a number to the nearest odd number.
    For double support phase, taking an odd number of steps is better.
    """
    return 2 * int(np.round(x / 2 - 0.5001)) + 1


class ParamsBase:
    """
    Base Param class aligned with SOBEC walk_without_think params.
    """
    mainJointsIds = []

    DT = 0.015
    Tstart = int(0.2 / DT)
    Tsingle = int(0.4 / DT)
    Tdouble = roundToOdd(0.01 / DT)
    Tfly = int(0.6 / DT)
    Tend = int(0.2 / DT)
    Tmpc = int(1.4 / DT)
    Tsimu = int(10 / DT)
    transitionDuration = (Tdouble - 1) // 2

    cycle = []
    contactPattern = []

    vcomWeight = 0
    vcomRef = np.r_[0.0, 0.0, 0.0]
    vcomImportance = np.array([0.0, 0.0, 0.0])

    comWeight = 0
    comRef = np.r_[0.0, 0.0, 0.0]
    comImportance = np.array([0.0, 0.0, 0.0])

    comHeightWeight = 0
    comHeightTargets = [np.r_[0.0, 0.0, 0.8]]
    comHeightTimes = []

    impactAltitudeWeight = 0
    impactRotationWeight = 0
    impactVelocityWeight = 0
    refMainJointsAtImpactWeight = 0

    refStateWeight = 0.0
    refTorqueWeight = 0.0
    stateTerminalWeight = 0
    refForceWeight = 0

    centerOfFrictionWeight = 0
    coneAxisWeight = 0.0
    conePenaltyWeight = 0
    copWeight = 0
    feetCollisionWeight = 0
    groundColWeight = 0
    footSize = 0.0
    verticalFootVelWeight = 0
    jointLimitWeight = 0
    refJointAcceleration = 0.0

    flyHighWeight = 0
    flyHighSlope = 0
    slope = 0.0
    minimalNormalForce = 0.0
    withNormalForceBoundOnly = False
    footMinimalDistance = 0.0

    kktDamping = 0
    baumgartGains = np.array([0, 100])
    solver_th_stop = 1e-3
    solver_maxiter = 200
    solver_reg_min = 1e-6

    saveFile = None
    guessFile = None
    preview = False
    save = False
    stateImportance = np.array([])
    stateTerminalImportance = np.array([])
    controlImportance = np.array([])


def round_to_odd(x: float) -> int:
    # Backward-compatible alias used by local scripts.
    return roundToOdd(x)
