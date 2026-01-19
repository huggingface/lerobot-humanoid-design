import pinocchio as pin
import numpy as np
from toolbox_parallel_robots.mounting import closedLoopMountProximal,closedLoopMountScipy,closedLoopMountCasadi

def moveHandAB(robot,hand,A,B,q_init=None):
    """
    Move the hand from A( SE3) to B (SE3).
    robot: pinocchio model of the robot
    hand: name of the hand frame
    A: target position (3,)
    B: target orientation (3,3)
    """
    model=robot.model
    data=model.createData()
    Lplacement=[A]*1000
    for i in range(1000):
        alpha=i/999
        translation=(1-alpha)*A.translation+alpha*B.translation
        orientation=pin.log(A.rotation)*(1-alpha)+pin.log(B.rotation)*alpha
        oriention=pin.exp(orientation)
        placement=pin.SE3.Identity()
        placement.rotation=oriention
        placement.translation=translation
        Lplacement[i]=placement



    # import meshcat
    # from pinocchio.visualize import MeshcatVisualizer
    # viz = MeshcatVisualizer(model, robot.visual_model, robot.visual_model)
    # viz.viewer = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
    # viz.clean()
    # viz.loadViewerModel(rootNodeName="universe")


    Lcontact_frame=[]
    torso_name="torso"
    torso_placement=pin.SE3.Identity()
    torso_placement.translation[2]=161*1e-3
    torso_placement.translation[0]=0.0
    id_torso=model.getFrameId(torso_name)
    Lcontact_frame.append([model.frames[id_torso],torso_placement])

    nconstraint_model=[]
    for f1,placement in Lcontact_frame[:]:
        nconstraint_model.append(pin.RigidConstraintModel(pin.ContactType.CONTACT_6D,model,f1.parentJoint,f1.placement,0,placement,pin.ReferenceFrame.LOCAL))

    ncdata=[c.createData() for c in nconstraint_model]

    hand_id = robot.model.getFrameId(hand)
    frame=robot.model.frames[hand_id]
    if not q_init is None:
        q_sol=q_init
    else:   
        q_sol=pin.neutral(model)
    qsols=[]
    for placement in Lplacement:
    # Define the target placement
        
        # Create a constraint for the hand frame to reach the target placement

        constraint_model = pin.RigidConstraintModel(pin.ContactType.CONTACT_3D,model,frame.parentJoint,frame.placement,0,placement,pin.ReferenceFrame.LOCAL)
        constraint_data = constraint_model.createData()
        
        # Solve for q that satisfies the constraint
        q_sol = closedLoopMountProximal(model, data, nconstraint_model+[constraint_model], ncdata+[constraint_data],q_prec=q_sol)
        qsols.append(q_sol)
        # viz.display(q_sol)
        print(placement)



    return qsols