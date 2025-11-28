
import pinocchio as pin
import numpy as np
import sobec
import os
import yaml
from yaml.loader import SafeLoader
from example_parallel_robots.loader_tools import completeRobotLoader
from toolbox_parallel_robots.mounting import closedLoopMountProximal,closedLoopMountScipy,closedLoopMountCasadi
pin.SE3.__repr__ = pin.SE3.__str__

CWD = os.path.dirname(os.path.abspath(__file__))





def loadbasic():
    model,constraint_models,actuation_model,visual_model,colision_model = completeRobotLoader(CWD + '/humanoid_v0/urdf', freeflyer=True)

    for c in constraint_models:
        c.corrector.Kp[:]=np.ones(6)*10
        c.corrector.Kd[:]=np.ones(6)*2
    model.armature[actuation_model.mot_ids_v]=[3400*8*1e-7,1477*18*1e-7,1477*18*1e-7,1477*6*1e-7,1477*6*1e-7,1477*18*1e-7]*2

    data=model.createData()
    cdata=[c.createData() for c in constraint_models]

    
    entraxe=-0.105
    foot_id=[model.getFrameId(f) for f in ["foot_right","foot_left"]]
    Lcontact_frame =[]
    for fid in foot_id:
        f=model.frames[fid]
        if "right" in f.name:
            placement=pin.SE3.Identity()
            placement.translation[1]=entraxe
            placement.rotation=pin.utils.rotate('z',np.deg2rad(0))
            Lcontact_frame.append([f,placement.copy()])
        else:
            placement=pin.SE3.Identity()
            placement.translation[1]=-entraxe
            placement.rotation= pin.utils.rotate('z',np.deg2rad(0))
            Lcontact_frame.append([f,placement.copy()])


    base_height= 0.65
    Lcontact_frame =[]
    for fid in foot_id:
        f=model.frames[fid]
        if "left" in f.name:
            placement=pin.SE3.Identity()
            placement.translation[1]=-entraxe
            placement.rotation=pin.utils.rotate('z',np.deg2rad(0))
            Lcontact_frame.append([f,placement.copy()])
        else:
            placement=pin.SE3.Identity()
            placement.translation[1]=entraxe
            placement.rotation= pin.utils.rotate('z',np.deg2rad(0))
            Lcontact_frame.append([f,placement.copy()])

    
    torso_name="torso"
    torso_placement=pin.SE3.Identity()
    torso_placement.translation[2]=base_height
    torso_placement.translation[0]=0.0
    id_torso=model.getFrameId(torso_name)
    Lcontact_frame.append([model.frames[id_torso],torso_placement])

    nconstraint_model=[]
    for f1,placement in Lcontact_frame[:]:
        nconstraint_model.append(pin.RigidConstraintModel(pin.ContactType.CONTACT_6D,model,f1.parentJoint,f1.placement,0,placement,pin.ReferenceFrame.LOCAL))

    ncdata=[c.createData() for c in nconstraint_model]
    

    q0 = closedLoopMountProximal(model,data,constraint_models+nconstraint_model[:],cdata+ncdata[:])

    # import meshcat
    # from pinocchio.visualize import MeshcatVisualizer


    # viz = MeshcatVisualizer(model, visual_model, visual_model)
    # viz.viewer = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
    # viz.clean()
    # viz.loadViewerModel(rootNodeName="universe")
    # q0 = closedLoopMountProximal(model,data,constraint_models+nconstraint_model[:],cdata+ncdata[:])
    # viz.display(q0)




    model.referenceConfigurations["half_sitting"] = q0

    print([f.name for f in model.frames])
    idfoot=[model.getFrameId(n) for n in ["foot_left","foot_right"]]
    for idf in idfoot:
        model.frames[idf].name += "48646"



    robot = sobec.wwt.RobotWrapper(model, contactKey="48646", closed_loop=True)
    robot.collision_model = visual_model
    robot.visual_model = visual_model
    robot.actuationModel = actuation_model
    robot.loop_constraints_models = constraint_models
    assert len(robot.contactIds) == 2


    return(robot)





    
def loadTunedHipx(angle = 0):


    model,constraint_models,actuation_model,visual_model,colision_model = completeRobotLoader(CWD + '/humanoid_v0/urdf', freeflyer=True)


    displacement1 = pin.SE3.Identity()
    displacement1.translation[2] = 101.312e-3

    displacement2 = pin.SE3.Identity()
    displacement2.rotation = pin.utils.rotate("y",np.deg2rad(angle))


    displacement3 = pin.SE3.Identity()
    displacement3.translation[2] = -110e-3

    displacement4 = pin.SE3.Identity()
    displacement4.rotation = pin.utils.rotate("x",np.deg2rad(-180))

    model.jointPlacements[3] = displacement1 * displacement2 *displacement3 * displacement4
    model.jointPlacements[9] = displacement1 *displacement2 * displacement3 * displacement4




    for c in constraint_models:
        c.corrector.Kp[:]=np.ones(6)*10
        c.corrector.Kd[:]=np.ones(6)*2
    model.armature[actuation_model.mot_ids_v]=[3400*8*1e-7,1477*18*1e-7,1477*18*1e-7,1477*6*1e-7,1477*6*1e-7,1477*18*1e-7]*2

    data=model.createData()
    cdata=[c.createData() for c in constraint_models]

    
    entraxe=-0.105
    foot_id=[model.getFrameId(f) for f in ["foot_right","foot_left"]]
    Lcontact_frame =[]
    for fid in foot_id:
        f=model.frames[fid]
        if "right" in f.name:
            placement=pin.SE3.Identity()
            placement.translation[1]=entraxe
            placement.rotation=pin.utils.rotate('z',np.deg2rad(0))
            Lcontact_frame.append([f,placement.copy()])
        else:
            placement=pin.SE3.Identity()
            placement.translation[1]=-entraxe
            placement.rotation= pin.utils.rotate('z',np.deg2rad(0))
            Lcontact_frame.append([f,placement.copy()])


    base_height= 0.65
    Lcontact_frame =[]
    for fid in foot_id:
        f=model.frames[fid]
        if "left" in f.name:
            placement=pin.SE3.Identity()
            placement.translation[1]=-entraxe
            placement.rotation=pin.utils.rotate('z',np.deg2rad(0))
            Lcontact_frame.append([f,placement.copy()])
        else:
            placement=pin.SE3.Identity()
            placement.translation[1]=entraxe
            placement.rotation= pin.utils.rotate('z',np.deg2rad(0))
            Lcontact_frame.append([f,placement.copy()])

    
    torso_name="torso"
    torso_placement=pin.SE3.Identity()
    torso_placement.translation[2]=base_height
    torso_placement.translation[0]=0.0
    id_torso=model.getFrameId(torso_name)
    Lcontact_frame.append([model.frames[id_torso],torso_placement])

    nconstraint_model=[]
    for f1,placement in Lcontact_frame[:]:
        nconstraint_model.append(pin.RigidConstraintModel(pin.ContactType.CONTACT_6D,model,f1.parentJoint,f1.placement,0,placement,pin.ReferenceFrame.LOCAL))

    ncdata=[c.createData() for c in nconstraint_model]
    





    q0 = closedLoopMountProximal(model,data,constraint_models+nconstraint_model[:],cdata+ncdata[:])

    # import meshcat
    # from pinocchio.visualize import MeshcatVisualizer


    # viz = MeshcatVisualizer(model, visual_model, visual_model)
    # viz.viewer = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
    # viz.clean()
    # viz.loadViewerModel(rootNodeName="universe")
    # q0 = closedLoopMountProximal(model,data,constraint_models+nconstraint_model[:],cdata+ncdata[:])
    # viz.display(q0)




    model.referenceConfigurations["half_sitting"] = q0

    print([f.name for f in model.frames])
    idfoot=[model.getFrameId(n) for n in ["foot_left","foot_right"]]
    for idf in idfoot:
        model.frames[idf].name += "48646"



    robot = sobec.wwt.RobotWrapper(model, contactKey="48646", closed_loop=True)
    robot.collision_model = visual_model
    robot.visual_model = visual_model
    robot.actuationModel = actuation_model
    robot.loop_constraints_models = constraint_models
    assert len(robot.contactIds) == 2


    return(robot)









    
def loadRobot():



    model,constraint_models,actuation_model,visual_model,colision_model = completeRobotLoader(CWD + '/humanoid_v0/urdf', freeflyer=True)


    model.armature[actuation_model.mot_ids_v]=[3400*8*1e-7,1477*18*1e-7,1477*18*1e-7,1477*6*1e-7,1477*6*1e-7,1477*18*1e-7]*2

    return(model,constraint_models,actuation_model,visual_model,colision_model)





def tuneModel(model,dx):


    oMz_left = pin.SE3.Identity()
    oMz_left.translation[:] = [56e-5, 0.105,0]
    oMz_left.rotation = np.array([[1,0,0],[0,-1,0],[0,0,-1]])

    oMz_right = pin.SE3.Identity()
    oMz_right.translation[:] = [56e-5, -0.105,0]
    oMz_right.rotation = np.array([[1,0,0],[0,-1,0],[0,0,-1]])    


    ### Hipe z rotation 
    zMc = pin.SE3.Identity()
    zMc.translation[2] = 101.312e-3

    oMc_left = oMz_left * zMc
    oMc_right =  oMz_right * zMc


    n_cMz_1_left  = pin.SE3.Identity()
    n_cMz_1_left.rotation = pin.utils.rotate("x",np.deg2rad(dx[0])) 

    n_cMz_1_right  = pin.SE3.Identity()
    n_cMz_1_right.rotation = pin.utils.rotate("x",np.deg2rad(-dx[0])) 

    n_cMz_2  = pin.SE3.Identity()
    n_cMz_2.rotation = pin.utils.rotate("y",np.deg2rad(dx[1])) 


    n_cMz_3 = pin.SE3.Identity()
    n_cMz_3.translation[2] = -101.312e-3

    n_cMz_left = n_cMz_1_left * n_cMz_2 * n_cMz_3
    n_cMz_right = n_cMz_1_right * n_cMz_2 * n_cMz_3


    n_oMz_left = oMc_left*n_cMz_left
    n_oMz_right = oMc_right*n_cMz_right

    model.jointPlacements[2] = n_oMz_left
    model.jointPlacements[8] = n_oMz_right

    ### Hip x rotation



    cMx_0_left = pin.SE3.Identity()
    cMx_0_left.rotation = pin.utils.rotate("z",np.deg2rad(dx[2])) 


    cMx_0_right = pin.SE3.Identity()
    cMx_0_right.rotation = pin.utils.rotate("z",np.deg2rad(-dx[2])) 

    cMx_1 = pin.SE3.Identity()
    cMx_1.rotation = pin.utils.rotate("y",np.deg2rad(dx[3]))  


    cMx_2 = pin.SE3.Identity()
    cMx_2.translation[2] = -110e-3

    cMx_3 = pin.SE3.Identity()
    cMx_3.rotation = pin.utils.rotate("x",np.deg2rad(-180))

    cMx_left = cMx_0_left * cMx_1 * cMx_2 * cMx_3

    cMx_right = cMx_0_right * cMx_1 * cMx_2 * cMx_3




    model.jointPlacements[3] = zMc * cMx_left 
    model.jointPlacements[9] = zMc * cMx_right


    ### Hipy rotation 


    xMy_left = pin.SE3.Identity()
    xMy_left.translation[:] =[0,-0.03, -0.11]
    xMy_left.rotation = np.array([[np.cos(np.deg2rad(45)) ,-np.cos(np.deg2rad(45)),0],
                                  [0                      ,0                      ,1],
                                  [-np.cos(np.deg2rad(45)),-np.cos(np.deg2rad(45)),0]])    



    xMy_right = pin.SE3.Identity()
    xMy_right.translation[:] =[0,0.03, -0.11]
    xMy_right.rotation = np.array([[np.cos(np.deg2rad(45)),np.cos(np.deg2rad(45)),0],
                                  [0                      ,0                     ,-1],
                                  [-np.cos(np.deg2rad(45)),np.cos(np.deg2rad(45)),0]])    


    xMy_right = pin.SE3.Identity()
    xMy_right.translation[:] =[0,0.03, -0.11]



    cMy_left =  pin.SE3.Identity()
    cMy_left.translation[:]  = [0,-30e-3, 0.]
    cMy_left.rotation =  np.array([[1,0,0],[0,0,-1],[0,1,0]])

    cMy_right =  pin.SE3.Identity()
    cMy_right.translation[:]  = [0,30e-3, 0.]
    cMy_right.rotation =  np.array([[1,0,0],[0,0,1],[0,-1,0]])


    n_cMy_1_left = pin.SE3.Identity()
    n_cMy_1_left.rotation = pin.utils.rotate("x",np.deg2rad(dx[4])) 

    n_cMy_2_left = pin.SE3.Identity()
    n_cMy_2_left.rotation = pin.utils.rotate("z",np.deg2rad(dx[5])) 

    n_cMy_1_right = pin.SE3.Identity()
    n_cMy_1_right.rotation = pin.utils.rotate("x",-np.deg2rad(dx[4])) 

    n_cMy_2_right = pin.SE3.Identity()
    n_cMy_2_right.rotation = pin.utils.rotate("z",-np.deg2rad(dx[5])) 

    n_cMy_left = n_cMy_1_left * n_cMy_2_left * cMy_left
    n_cMy_right = n_cMy_1_right * n_cMy_2_right * cMy_right


    n_xMy_left = cMx_left.inverse() * n_cMy_left
    n_xMy_right = cMx_right.inverse() * n_cMy_right

    model.jointPlacements[4] = n_xMy_left 
    model.jointPlacements[10] = n_xMy_right

    return(model)

def create_robot(dx):
    model,constraint_models,actuation_model,visual_model,collision_model = loadRobot()
    model=tuneModel(model,dx)


    data=model.createData()
    
    entraxe=-0.105
    foot_id=[model.getFrameId(f) for f in ["foot_right","foot_left"]]
    Lcontact_frame =[]
    for fid in foot_id:
        f=model.frames[fid]
        if "right" in f.name:
            placement=pin.SE3.Identity()
            placement.translation[1]=entraxe
            placement.rotation=pin.utils.rotate('z',np.deg2rad(0))
            Lcontact_frame.append([f,placement.copy()])
        else:
            placement=pin.SE3.Identity()
            placement.translation[1]=-entraxe
            placement.rotation= pin.utils.rotate('z',np.deg2rad(0))
            Lcontact_frame.append([f,placement.copy()])


    base_height= 0.65
    Lcontact_frame =[]
    for fid in foot_id:
        f=model.frames[fid]
        if "left" in f.name:
            placement=pin.SE3.Identity()
            placement.translation[1]=-entraxe
            placement.rotation=pin.utils.rotate('z',np.deg2rad(0))
            Lcontact_frame.append([f,placement.copy()])
        else:
            placement=pin.SE3.Identity()
            placement.translation[1]=entraxe
            placement.rotation= pin.utils.rotate('z',np.deg2rad(0))
            Lcontact_frame.append([f,placement.copy()])

    
    torso_name="torso"
    torso_placement=pin.SE3.Identity()
    torso_placement.translation[2]=base_height
    torso_placement.translation[0]=0.0
    id_torso=model.getFrameId(torso_name)
    Lcontact_frame.append([model.frames[id_torso],torso_placement])

    nconstraint_model=[]
    for f1,placement in Lcontact_frame[:]:
        nconstraint_model.append(pin.RigidConstraintModel(pin.ContactType.CONTACT_6D,model,f1.parentJoint,f1.placement,0,placement,pin.ReferenceFrame.LOCAL))

    ncdata=[c.createData() for c in nconstraint_model]




    q0 = closedLoopMountProximal(model,data,nconstraint_model[:],ncdata[:])

    import meshcat
    from pinocchio.visualize import MeshcatVisualizer


    viz = MeshcatVisualizer(model, visual_model, visual_model)
    viz.viewer = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
    viz.clean()
    viz.loadViewerModel(rootNodeName="universe")
    q0 = closedLoopMountProximal(model,data,nconstraint_model[:],ncdata[:])
    viz.display(q0)




    model.referenceConfigurations["half_sitting"] = q0

    print([f.name for f in model.frames])
    idfoot=[model.getFrameId(n) for n in ["foot_left","foot_right"]]
    for idf in idfoot:
        model.frames[idf].name += "48646"



    robot = sobec.wwt.RobotWrapper(model, contactKey="48646", closed_loop=True)
    robot.collision_model = visual_model
    robot.visual_model = visual_model
    robot.actuationModel = actuation_model
    robot.loop_constraints_models = constraint_models
    assert len(robot.contactIds) == 2


    return(robot)