import numpy as np
import pinocchio as pin
def roundToOdd(x):
    '''
    Round a number to the nearest odd number.
    For double support phase, taking an odd number of steps is better.
    '''
    return 2 * int(np.round(x / 2 - 0.5001)) + 1


class WalkV0Params:
    '''
    Parameters for the walk of the Battobot robot.
    '''
    mainJointsIds = [
        'hipz_right',
        'hipy_right',
        'knee_right',
        'hipz_left',
        'hipy_left',
        'knee_left',
    ]
    ankleLimitsWeight = 0
    # Define time steps
    DT = 0.015
    Tstart = int(0.2 / DT)
    Tsingle = int(0.3 / DT)  
    Tdouble = roundToOdd(0.003 / DT)
    Tend = int(0.2 / DT)
    Tmpc = int(1.4 / DT)
    Tsimu = int(10 / DT)
    transitionDuration = (Tdouble - 1) // 2

    cycle = ( [[0, 1]] * Tsingle
              + [[1, 1]] * Tdouble
              + [[1, 0]] * Tsingle
              + [[1, 1]] * Tdouble
            )
    contactPattern = (
        []
        + [[1, 1]] * Tstart
        + (cycle * 6)
        + [[1, 1]] * Tend
        + [[1, 1]]
    )

    ## Define costs
    # * Task specific cost
    vcomWeight = 1000
    vcomRef = np.r_[ 1., 0., 0 ]
    vcomImportance = np.array([1, 1, 0])

    comWeight = 0
    comRef = np.r_[ 0, 0, 0.65]
    comImportance = np.array([0, 0, 1])

    comHeightWeight = 0
    comHeightTargets = [np.r_[0.0, 0.0, 0.65]]
    comHeightTimes = []
    
    # * Impact Time costs
    impactAltitudeWeight = 1e4  # /
    impactRotationWeight = 1e4  # /
    impactVelocityWeight = 1e4  # /
    refMainJointsAtImpactWeight = 0

    # * Regularisation costs
    refStateWeight = 0.2       # /
    refTorqueWeight = 0.02     # /
    stateTerminalWeight = 1e2
    refForceWeight = 1000    # /

    # * Realism costs
    centerOfFrictionWeight = 0.2
    coneAxisWeight =  0.000
    conePenaltyWeight = 0
    copWeight = 100
    feetCollisionWeight = 100 # 1000
    groundColWeight = 0
    footSize = 0.05
    verticalFootVelWeight = 0 # 20
    jointLimitWeight = 0
    refJointAcceleration = 0.0

    flyHighWeight =  100
    flyHighSlope = 3/2e-2
    slope = 0.0000
    minimalNormalForce = 1.0
    withNormalForceBoundOnly = False
    footMinimalDistance = 0.2

    # Solver parameters
    kktDamping = 0
    baumgartGains = np.array([0, 100])
    transitionDuration = 3
    solver_th_stop = 1e-3
    solver_maxiter = 200
    solver_reg_min = 1e-6

    # Save parameters
    saveFile = "/tmp/stairs_virgile_closed.npy"
    guessFile = None
    preview = True
    save = False

    def __init__(self):
        self.ref_rots = [
                    [pin.SE3.Identity(), pin.SE3.Identity()] for _ in self.contactPattern
                ]
        
        basisQWeights = [0,0,0,50,50,0]
        legQWeights = [
            20, 5, 1, # hip z, x, y
            1, # knee (passive)
            1, 1, # ankle x, y
        ]
        basisVWeights = [0,0,0,3,3,0]
        legVWeights = [
            20, 5, 1, # hip z, x, y
            1, # knee (passive)
            1, 1, # ankle x, y
        ]
        self.stateImportance = np.array(
            basisQWeights + legQWeights * 2 + basisVWeights + legVWeights * 2
        )
        nv = len(basisVWeights) + 2* len(legVWeights)
        self.stateTerminalImportance = np.array([0, 0, 0, 0, 0, 0] + [0] * (nv - 6) + [1] * nv)
        self.controlImportance = np.array([1] * 12)
