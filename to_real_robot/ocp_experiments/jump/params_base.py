import numpy as np


def round_to_odd(x: float) -> int:
    return 2 * int(np.round(x / 2 - 0.5001)) + 1


class ParamsBase:
    DT = 0.015
    Tstart = int(0.2 / DT)
    Tsingle = int(0.4 / DT)
    Tdouble = round_to_odd(0.01 / DT)
    Tfly = int(0.6 / DT)
    Tend = int(0.2 / DT)

    vcomWeight = 0
    vcomRef = np.r_[0.0, 0.0, 0.0]
    vcomImportance = np.array([0.0, 0.0, 0.0])

    comWeight = 0
    comRef = np.r_[0.0, 0.0, 0.0]
    comImportance = np.array([0.0, 0.0, 0.0])

    impactAltitudeWeight = 0
    impactRotationWeight = 0
    impactVelocityWeight = 0
    refMainJointsAtImpactWeight = 0

    refStateWeight = 0.0
    refTorqueWeight = 0.0
    stateTerminalWeight = 0
    refForceWeight = 0
    copWeight = 0

    footSize = 0.0
    footMinimalDistance = 0.0

    kktDamping = 0
    baumgartGains = np.array([0, 100])
    solver_th_stop = 1e-4
    solver_maxiter = 200
    solver_reg_min = 1e-6

    saveFile = None
    guessFile = None
    preview = False
    save = False
