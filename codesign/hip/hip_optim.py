
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
import re
import yaml
from yaml.loader import SafeLoader
from warnings import warn
from os.path import dirname, exists, join
import sys
import numpy as np
from toolbox_parallel_robots import freezeJoints, ActuationModel
from example_parallel_robots.robot_options import ROBOTS
from example_parallel_robots.path import EXAMPLE_PARALLEL_ROBOTS_MODEL_DIR, EXAMPLE_PARALLEL_ROBOTS_SOURCE_DIR
from toolbox_parallel_robots.mounting import closedLoopMountProximal,closedLoopMountScipy,closedLoopMountCasadi

from urdf.humanoids_loader import loadRobot,tuneModel,create_robot


from codesign.hip.V0_walk_param import WalkV0Params
from codesign.hip.V0_sidewalk_param import SideWalkV0Params
import sobec
import crocoddyl as croc

import copy


import matplotlib.pyplot as plt
class V0Loader():
    def __init__(self):
        self.model,self.constraint_models,self.actuation_model,self.visual_model,self.collision_model = loadRobot()






    def tune(self,dx):
        model = tuneModel(copy.copy(self.model),dx)
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


        model.referenceConfigurations["half_sitting"] = q0

        print([f.name for f in model.frames])
        idfoot=[model.getFrameId(n) for n in ["foot_left","foot_right"]]
        for idf in idfoot:
            model.frames[idf].name += "48646"



        robot = sobec.wwt.RobotWrapper(model, contactKey="48646", closed_loop=True)
        robot.collision_model = self.visual_model
        robot.visual_model =  self.visual_model
        robot.actuationModel =  self.actuation_model
        robot.loop_constraints_models =  self.constraint_models
        assert len(robot.contactIds) == 2
        return(robot)


class EvaluateRobot():
    def __init__(self):
        self.loader = V0Loader()
        self.walkparams = WalkV0Params()
        self.sidewalkparams = SideWalkV0Params()
        self.TauZlim=5.5
        self.TauXlim=10.
        self.TauYlim=30.


    def evaluate_walk(self,dx,vizual = False):
        robot = self.loader.tune(dx)
        walkParams = self.walkparams
        contactPattern = walkParams.contactPattern

        ddp = sobec.wwt.buildSolver(robot, contactPattern, walkParams, solver='FDDP')
        problem = ddp.problem
        x0s, u0s = sobec.wwt.buildInitialGuess(ddp.problem, walkParams)
        # ddp.setCallbacks([croc.CallbackVerbose(), croc.CallbackLogger()])
        ddp.solve(init_xs=x0s, init_us=u0s, maxiter=800, init_reg=1e-6, is_feasible=False)

        sol = sobec.wwt.Solution(robot, ddp)
        vs = sol.xs[:,robot.model.nq:]
        velocity = 1-np.mean(vs[:,:2])


        # torquez = np.sum(np.exp(np.abs(sol.us[20:-20, 0] ) + np.abs(sol.us[20:-20,6+ 0]) -  2*self.TauZlim))
        # torquex = np.sum(np.exp(np.abs(sol.us[20:-20, 1]) + np.abs(sol.us[20:-20, 6+1])  -  2*self.TauXlim))
        # torquey = np.sum(np.exp(np.abs(sol.us[20:-20, 2]) + np.abs(sol.us[20:-20, 6+2])  -  2*self.TauYlim))



        torquez = np.sum(np.exp(np.abs(sol.us[20:-20, 0]) -  1*self.TauZlim)) + np.sum(np.exp(np.abs(sol.us[20:-20, 6]) -  1*self.TauZlim))
        torquex = np.sum(np.exp(np.abs(sol.us[20:-20, 1])   -  1*self.TauXlim)) +np.sum(np.exp(np.abs(sol.us[20:-20, 7]) -  1*self.TauXlim))
        torquey = np.sum(np.exp(np.abs(sol.us[20:-20, 2])   -  1*self.TauYlim)) + np.sum(np.exp(np.abs(sol.us[20:-20, 8]) -  1*self.TauYlim))



        if vizual:
            import meshcat
            from pinocchio.visualize import MeshcatVisualizer
            viz = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
            viz.viewer = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
            viz.clean()
            viz.loadViewerModel(rootNodeName="universe")
            while input("Press q to quit the visualisation") != "q":
                viz.play(np.array(ddp.xs)[:, : robot.model.nq], walkParams.DT)
        
        self.velocity = velocity
        
        return(torquez+torquex+torquey+10*velocity)

    def evaluate_sidewalk(self,dx,vizual = False):
        robot = self.loader.tune(dx)
        walkParams = self.sidewalkparams
        contactPattern = walkParams.contactPattern

        ddp = sobec.wwt.buildSolver(robot, contactPattern, walkParams, solver='FDDP')
        problem = ddp.problem
        x0s, u0s = sobec.wwt.buildInitialGuess(ddp.problem, walkParams)
        # ddp.setCallbacks([croc.CallbackVerbose(), croc.CallbackLogger()])
        ddp.solve(init_xs=x0s, init_us=u0s, maxiter=800, init_reg=1e-6, is_feasible=False)

        sol = sobec.wwt.Solution(robot, ddp)
        vs = sol.xs[:,robot.model.nq:]
        velocity = 1-np.mean(vs[:,:2])


        # torquez = np.sum(np.exp(np.abs(sol.us[20:-20, 0] ) + np.abs(sol.us[20:-20,6+ 0]) -  2*self.TauZlim))
        # torquex = np.sum(np.exp(np.abs(sol.us[20:-20, 1]) + np.abs(sol.us[20:-20, 6+1])  -  2*self.TauXlim))
        # torquey = np.sum(np.exp(np.abs(sol.us[20:-20, 2]) + np.abs(sol.us[20:-20, 6+2])  -  2*self.TauYlim))



        torquez = np.sum(np.exp(np.abs(sol.us[20:-20, 0]) -  1*self.TauZlim)) + np.sum(np.exp(np.abs(sol.us[20:-20, 6]) -  1*self.TauZlim))
        torquex = np.sum(np.exp(np.abs(sol.us[20:-20, 1])   -  1*self.TauXlim)) +np.sum(np.exp(np.abs(sol.us[20:-20, 7]) -  1*self.TauXlim))
        torquey = np.sum(np.exp(np.abs(sol.us[20:-20, 2])   -  1*self.TauYlim)) + np.sum(np.exp(np.abs(sol.us[20:-20, 8]) -  1*self.TauYlim))



        if vizual:
            import meshcat
            from pinocchio.visualize import MeshcatVisualizer
            viz = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
            viz.viewer = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
            viz.clean()
            viz.loadViewerModel(rootNodeName="universe")
            while input("Press q to quit the visualisation") != "q":
                viz.play(np.array(ddp.xs)[:, : robot.model.nq], walkParams.DT)
        
        self.velocity = velocity
        return(torquez+torquex+torquey+10*velocity)

    def evaluate(self,dx):
        r1 = self.evaluate_walk(dx)
        r2 = self.evaluate_sidewalk(dx)
        return(0.66*r1 + 0.33*r2)

    def vizalize(self,dx):
        robot = self.loader.tune(dx)
        walkParams = self.sidewalkparams
        contactPattern = walkParams.contactPattern

        ddp = sobec.wwt.buildSolver(robot, contactPattern, walkParams, solver='FDDP')
        problem = ddp.problem
        x0s, u0s = sobec.wwt.buildInitialGuess(ddp.problem, walkParams)
        # ddp.setCallbacks([croc.CallbackVerbose(), croc.CallbackLogger()])
        ddp.solve(init_xs=x0s, init_us=u0s, maxiter=800, init_reg=1e-6, is_feasible=False)

        sol = sobec.wwt.Solution(robot, ddp)
        vs = sol.xs[:,robot.model.nq:]
        velocity = 1-np.mean(vs[:,:2])
        plt.figure()
        plt.subplot(511)
        plt.plot(sol.us[20:-20, 0])
        plt.subplot(512)
        plt.plot(sol.us[20:-20, 1])
        plt.subplot(513)
        plt.plot(sol.us[20:-20, 2])
        plt.subplot(514)
        plt.plot(vs[:,1])
        import meshcat
        from pinocchio.visualize import MeshcatVisualizer
        viz = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
        viz.viewer = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
        viz.clean()
        viz.loadViewerModel(rootNodeName="universe")
        while input("Press q to quit the visualisation") != "q":
            viz.play(np.array(ddp.xs)[:, : robot.model.nq], walkParams.DT)


        walkParams = self.walkparams
        contactPattern = walkParams.contactPattern

        ddp = sobec.wwt.buildSolver(robot, contactPattern, walkParams, solver='FDDP')
        problem = ddp.problem
        x0s, u0s = sobec.wwt.buildInitialGuess(ddp.problem, walkParams)
        # ddp.setCallbacks([croc.CallbackVerbose(), croc.CallbackLogger()])
        ddp.solve(init_xs=x0s, init_us=u0s, maxiter=800, init_reg=1e-6, is_feasible=False)

        sol = sobec.wwt.Solution(robot, ddp)
        vs = sol.xs[:,robot.model.nq:]
        velocity = 1-np.mean(vs[:,:2])

        plt.subplot(511)
        plt.plot(sol.us[20:-20, 0])
        plt.subplot(512)
        plt.plot(sol.us[20:-20, 1])
        plt.subplot(513)
        plt.plot(sol.us[20:-20, 2])
        plt.subplot(515)
        plt.plot(vs[:,0])


        while input("Press q to quit the visualisation") != "q":
            viz.play(np.array(ddp.xs)[:, : robot.model.nq], walkParams.DT)
        
        plt.show()

if __name__ == "__main__":
    dx1 = np.array([0.,0.,0.,90.,0.,0.])
    dx2 = np.array([0.,0.,0.,45.,0.,0.])
    dx3=np.array([ -6.30157784,  -2.9773718 , -67.92956906,  37.73483886,
       -23.44075896, -79.37300617])
    dx4=np.array([-6.65684339,  -1.73665785, -68.81551501,  37.43446151,
         -23.24196683, -78.50777904])
    dx5=np.array([ -5.76495799,  -2.60621304, -62.80618123,  36.23841476,
         -22.81023506, -78.14087952])

    evaluate = EvaluateRobot()
    stop
    Lvalue=[]
    for dx in [dx1,dx2,dx3,dx4,dx5]:
        value =evaluate.evaluate(dx)
        Lvalue.append(copy.copy(value))




    stop
    from cmaes import CMA

    optimizer = CMA(mean=np.float64(dx), sigma=1.3)
    for generation in range(50):
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            value =evaluate.evaluate(x)
            solutions.append((x, np.float64(value)))
            print(f"#{generation} {value} (x1={x[0]}, x2 = {x[1]})")
        optimizer.tell(solutions)