from urdf.humanoids_loader import *
from codesign.upper_body.move_arms_utils import *
from toolbox_parallel_robots.jacobian   import *
from toolbox_parallel_robots.actuation_data import *
from sympy import Matrix
from toolbox_parallel_robots.mounting import closedLoopMountProximal,closedLoopMountScipy,closedLoopMountCasadi

np.set_printoptions(precision=4, suppress=True)


import copy


class ArmEvaluator:
    def __init__(self):
        self.robot=loadUpperBody()
        self.model=self.robot.model
        self.original_model=copy.deepcopy(self.model)
        model=self.model
        self.q_init = self.robot.model.referenceConfigurations["half_sitting"]
        self.actuation_model = self.robot.actuationModel
        self.actuation_data = ActuationData(self.model, [], self.actuation_model)
        self.data=self.model.createData()
        self.constraint_models=self.robot.loop_constraints_models
        self.constraint_datas=[c.createData() for c in self.constraint_models]
        self.curve_approx=[0,0,0]











        Lcontact_frame =[]
        
        torso_name="torso"
        torso_placement=pin.SE3.Identity()
        torso_placement.translation[2]=161*1e-3
        torso_placement.translation[0]=0.0
        id_torso=model.getFrameId(torso_name)
        Lcontact_frame.append([model.frames[id_torso],torso_placement])

        torso_constraint=[]
        for f1,placement in Lcontact_frame[:]:
            torso_constraint.append(pin.RigidConstraintModel(pin.ContactType.CONTACT_6D,model,f1.parentJoint,f1.placement,0,placement,pin.ReferenceFrame.LOCAL))

        self.torso_constraint_datas=[c.createData() for c in torso_constraint]
        self.torso_constraint = torso_constraint

        import meshcat
        from pinocchio.visualize import MeshcatVisualizer
        self.viz = MeshcatVisualizer(self.robot.model, self.robot.visual_model, self.robot.visual_model)
        self.viz.viewer = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
        self.viz.clean()
        self.viz.loadViewerModel(rootNodeName="universe")
        
    def tunemodel(self,dx,display=False):
        new_model=tuneArmModel(self.original_model,dx)
        new_data=new_model.createData()

        self.model=self.robot.model=copy.deepcopy(new_model)
        self.data=new_data


        Lcontact_frame =[]
        
        torso_name="torso"
        torso_placement=pin.SE3.Identity()
        torso_placement.translation[2]=161*1e-3
        torso_placement.translation[0]=0.0
        id_torso=self.model.getFrameId(torso_name)
        Lcontact_frame.append([self.model.frames[id_torso],torso_placement])

        torso_constraint=[]
        for f1,placement in Lcontact_frame[:]:
            torso_constraint.append(pin.RigidConstraintModel(pin.ContactType.CONTACT_6D,self.model,f1.parentJoint,f1.placement,0,placement,pin.ReferenceFrame.LOCAL))

        self.torso_constraint_datas=[c.createData() for c in torso_constraint]
        self.torso_constraint = torso_constraint


        pin.forwardKinematics(self.model,self.data,pin.neutral(self.model))
        if display:

            import meshcat
            from pinocchio.visualize import MeshcatVisualizer
            self.viz = MeshcatVisualizer(self.robot.model, self.robot.visual_model, self.robot.visual_model)
            self.viz.viewer = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
            self.viz.clean()
            self.viz.loadViewerModel(rootNodeName="universe")
            # self.viz.display(pin.neutral(new_model))



    def evaluate_configuration(self,q,F):
        pin.forwardKinematics(self.model, self.data, q, np.zeros(self.model.nv))
        pin.updateFramePlacements(self.model,self.data) 
        oMf=self.data.oMf[14]

        inverseConstraintKinematicsSpeed(self.model,self.data,[],[],self.actuation_model,self.actuation_data,q,14,np.zeros(6))
        J= self.actuation_data.Jf_closed
        cost=J.T @ (oMf.actionInverse.T @ F)

        elbow_placement= self.data.oMi[5]
        cost+=(1000*max([+elbow_placement.translation[1]+0.13,0]))**2
        

        return np.linalg.norm(cost)**2
    
    def compute_gradient(self,original_q,F):
        grad=np.zeros(self.model.nv)
        base_cost=self.evaluate_configuration(original_q,F)
        self.curve_approx[1]=base_cost
        J= self.actuation_data.Jf_closed
        A=Matrix(J[:3,:4])
        vq=np.array(A.nullspace()[0])
        nvq=np.zeros(self.model.nv)
        nvq[6:]=vq.T
        q=pin.integrate(self.model,original_q,nvq*1e-5)    
        new_cost=self.evaluate_configuration(q,F)
        self.curve_approx[2]=new_cost
        q=pin.integrate(self.model,original_q,-nvq*1e-5)    
        new_cost=self.evaluate_configuration(q,F)
        self.curve_approx[0]=new_cost

        a=(self.curve_approx[2]+self.curve_approx[0]-2*self.curve_approx[1])/((1e-5)**2)
        b=(self.curve_approx[2]-self.curve_approx[0])/(2*1e-5)
        c=self.curve_approx[1]

        alpha = -b/(a)
        grad=(self.curve_approx[2]-self.curve_approx[0])/2e-5
        nvq=nvq / (np.linalg.norm(nvq) + 1e-12)
        if abs(grad) < 1e-1:
            direction=alpha * nvq
        else:
            direction=-grad * nvq /np.linalg.norm(grad)*0.1
        


        return alpha,direction


    def find_better_configuration(self,q,F,disp=False):
        if disp:
            self.viz.display(q)
        best_q=q
        best_cost=self.evaluate_configuration(q,F)
        alpha,direction=self.compute_gradient(q,F)
        iter=0
        step=1
        if disp:
            print("starting optimization")
            print("iter:",0," cost:",best_cost,"alpha:",alpha)
        while np.abs(alpha) > 1e-5 and iter<500:
            nq = pin.integrate(self.model,best_q,direction*step*1e-5)
            cost=self.evaluate_configuration(nq,F)
            if cost < best_cost:
                best_cost=cost
                best_q=nq
                step=step*1.2
            else:
                
                # print("no improvement")
                step=step*0.5
            alpha,direction=self.compute_gradient(best_q,F)
            iter+=1
            if disp:
                self.viz.display(best_q)
                print("iter:",iter," cost:",best_cost,"alpha:",alpha)
            
        if disp:
            print("optimization finished")
        return best_q,best_cost






    def evaluate_on_trajAB(self,A,B,qs=None,Feval=np.array([0,0,10,0,0,0]),disp=False):
        if qs is None:
            qs=self.q_init

        arme_frme_id = self.robot.model.getFrameId("hand")
        frame=self.robot.model.frames[arme_frme_id]
        constraint = pin.RigidConstraintModel(pin.ContactType.CONTACT_3D,self.model,frame.parentJoint,frame.placement,0,A,pin.ReferenceFrame.LOCAL)
        constraint_data = constraint.createData()
        q=closedLoopMountProximal(self.model,self.data,self.torso_constraint+self.constraint_models+[constraint],self.torso_constraint_datas+self.constraint_datas+[constraint_data],q_prec=qs)
        cost = self.evaluate_configuration(q,Feval)
        total_cost=0

        q,ncost=self.find_better_configuration(q,Feval,disp=disp)
        oq=q.copy()
        # print("initial cost:",cost," final cost:",ncost)
        for i in range(100):
            alpha=i/99
            translation=(1-alpha)*A.translation+alpha*B.translation
            orientation=pin.log(A.rotation)*(1-alpha)+pin.log(B.rotation)*alpha
            oriention=pin.exp(orientation)
            placement=pin.SE3.Identity()
            placement.rotation=oriention
            placement.translation=translation

            constraint = pin.RigidConstraintModel(pin.ContactType.CONTACT_3D,self.model,frame.parentJoint,frame.placement,0,placement,pin.ReferenceFrame.LOCAL)
            constraint_data = constraint.createData()
            q=closedLoopMountProximal(self.model,self.data,self.torso_constraint+self.constraint_models+[constraint],self.torso_constraint_datas+self.constraint_datas+[constraint_data],q_prec=q)
            cost = self.evaluate_configuration(q,Feval)
            total_cost+=cost
            nq,ncost=self.find_better_configuration(q,Feval,disp=False)

            if np.linalg.norm(nq - oq) < 400e-1:
                q=nq
            oq=q.copy()
            if disp and np.linalg.norm(nq - q) < 400e-1:
                print("better configuration found, cost improved from ",cost," to ",ncost, "amelioration:",cost - ncost)
            elif disp:
                print("large deviation detected, skipping gradient step, keeping cost at ",cost, " instead of ",ncost, "amelioration:",cost - ncost)
            
            if disp:
                self.viz.display(q)
        return(total_cost)

    def evaluateForce(self,dx,disp=True):

        self.tunemodel(dx,disp)
        self.data=self.model.createData()
        q=np.zeros(self.model.nq)

        A = pin.SE3.Identity()
        A.translation = np.array([0.23, -0.07, 0.2])
        B = pin.SE3.Identity()
        B.translation = np.array([0.23, -0.07, 0.5])
        

        
        Feval=np.array([0,0,10,0,0,0])
        cout1=self.evaluate_on_trajAB(A,B,q,Feval=Feval,disp=disp)

        A = pin.SE3.Identity()
        A.translation = np.array([0.15, -0.07, 0.35])
        B = pin.SE3.Identity()
        B.translation = np.array([0.32, -0.07, 0.35])
        Feval=np.array([10,0,0,0,0,0])
        cout2=self.evaluate_on_trajAB(A,B,q,Feval=Feval,disp=disp)
        return(cout1+cout2)



    def evaluateManipuliability(self,dx,disp=True):



        A = pin.SE3.Identity()
        A.translation = np.array([0.25, -0.1, 0.35])
        B = pin.SE3.Identity()
        B.translation = np.array([0.15, -0.25, 0.35])

        self.tunemodel(dx,False)
        self.data=self.model.createData()

        q=np.zeros(self.model.nq)
        cout1=self.evaluate_on_trajAB_manipulability(A,B,q,disp=disp)

        q=np.zeros(self.model.nq)
        A = pin.SE3.Identity()
        A.translation = np.array([0.15, -0.25, 0.3])
        B = pin.SE3.Identity()
        B.translation = np.array([0.15, -0.25, 0.55])
        cout2=self.evaluate_on_trajAB_manipulability(A,B,q,disp=disp)
        return(cout1+cout2)



    def findBetterConfigurationManipualability(self,q,disp=True):


        if disp:
            self.viz.display(q)
        best_q=q
        best_cost=self.evaluate_configuration_manipulability(q)
        alpha,direction=self.compute_gradient_manipulability(q)
        iter=0
        step=1
        if disp:
            print("starting optimization")
            print("iter:",0," cost:",best_cost,"alpha:",alpha)
        while np.abs(alpha) > 1e-5 and iter<500:
            nq = pin.integrate(self.model,best_q,direction*step*1e-5)
            cost=self.evaluate_configuration_manipulability(nq)
            if cost < best_cost:
                best_cost=cost
                best_q=nq
                step=step*1.2
            else:
                
                # print("no improvement")
                step=step*0.5
            alpha,direction=self.compute_gradient_manipulability(best_q)
            iter+=1
            if disp:
                self.viz.display(best_q)
                print("iter:",iter," cost:",best_cost,"alpha:",alpha)
            
        if disp:
            print("optimization finished")
        return best_q,best_cost



    def evaluate_configuration_manipulability(self,q):
        pin.forwardKinematics(self.model, self.data, q, np.zeros(self.model.nv))
        pin.updateFramePlacements(self.model,self.data) 
        oMf=self.data.oMf[14]

        inverseConstraintKinematicsSpeed(self.model,self.data,[],[],self.actuation_model,self.actuation_data,q,14,np.zeros(6))
        J= self.actuation_data.Jf_closed
        cost=((1/np.linalg.det(J.T @ J))**2)/100
        # print(J.T @ J)
        elbow_placement= self.data.oMi[5]
        cost+=(1000*max([+elbow_placement.translation[1]+0.13,0]))**2
        

        return np.linalg.norm(cost)**2
    
    def compute_gradient_manipulability(self,original_q):
        grad=np.zeros(self.model.nv)
        base_cost=self.evaluate_configuration_manipulability(original_q)
        self.curve_approx[1]=base_cost
        J= self.actuation_data.Jf_closed
        A=Matrix(J[:3,:4])
        vq=np.array(A.nullspace()[0])
        nvq=np.zeros(self.model.nv)
        nvq[6:]=vq.T
        q=pin.integrate(self.model,original_q,nvq*1e-5)    
        new_cost=self.evaluate_configuration_manipulability(q)
        self.curve_approx[2]=new_cost
        q=pin.integrate(self.model,original_q,-nvq*1e-5)    
        new_cost=self.evaluate_configuration_manipulability(q)
        self.curve_approx[0]=new_cost

        a=(self.curve_approx[2]+self.curve_approx[0]-2*self.curve_approx[1])/((1e-5)**2)
        b=(self.curve_approx[2]-self.curve_approx[0])/(2*1e-5)
        c=self.curve_approx[1]

        alpha = -b/(a)
        grad=(self.curve_approx[2]-self.curve_approx[0])/2e-5
        nvq=nvq / (np.linalg.norm(nvq) + 1e-12)
        if abs(grad) < 1e-1:
            direction=alpha * nvq
        else:
            direction=-grad * nvq /np.linalg.norm(grad)*0.2
        


        return alpha,direction





    def evaluate_on_trajAB_manipulability(self,A,B,qs=None,disp=False):
        if qs is None:
            qs=self.q_init

        arme_frme_id = self.robot.model.getFrameId("hand")
        frame=self.robot.model.frames[arme_frme_id]
        constraint = pin.RigidConstraintModel(pin.ContactType.CONTACT_3D,self.model,frame.parentJoint,frame.placement,0,A,pin.ReferenceFrame.LOCAL)
        constraint_data = constraint.createData()
        q=closedLoopMountProximal(self.model,self.data,self.torso_constraint+self.constraint_models+[constraint],self.torso_constraint_datas+self.constraint_datas+[constraint_data],q_prec=qs)
        cost = self.evaluate_configuration_manipulability(q)


        q,ncost=self.findBetterConfigurationManipualability(q,disp=disp)
        #print("initial cost:",cost," final cost:",ncost)
        total_cost=0
        for i in range(100):
            alpha=i/99
            translation=(1-alpha)*A.translation+alpha*B.translation
            orientation=pin.log(A.rotation)*(1-alpha)+pin.log(B.rotation)*alpha
            oriention=pin.exp(orientation)
            placement=pin.SE3.Identity()
            placement.rotation=oriention
            placement.translation=translation

            constraint = pin.RigidConstraintModel(pin.ContactType.CONTACT_3D,self.model,frame.parentJoint,frame.placement,0,placement,pin.ReferenceFrame.LOCAL)
            constraint_data = constraint.createData()
            q=closedLoopMountProximal(self.model,self.data,self.torso_constraint+self.constraint_models+[constraint],self.torso_constraint_datas+self.constraint_datas+[constraint_data],q_prec=q)
            cost = self.evaluate_configuration_manipulability(q)

            nq,ncost=self.findBetterConfigurationManipualability(q,disp=False)

            pin.framesForwardKinematics(self.model,self.data,nq)
            oMf=self.data.oMf[arme_frme_id]
            dist=np.linalg.norm(oMf.translation-translation)
            total_cost+=dist+cost/100000
            q=nq
            if disp and np.linalg.norm(nq - q) < 35e-1:
                q=nq
                print("better configuration found, cost improved from ",cost," to ",ncost, "amelioration:",cost - ncost)
            elif disp:
                print("large deviation detected, skipping gradient step, keeping cost at ",cost, " instead of ",ncost, "amelioration:",cost - ncost)
            
            if disp:
                self.viz.display(q)
        return(total_cost)

    def evaluate(self,dx,disp=False):

        cout1=self.evaluateForce(dx,disp)
        # cout2=self.evaluateManipuliability(dx,disp)
        return(cout1)#+cout2/10)
    

if __name__=="__main__":


    evaluate = ArmEvaluator()
    
    dx=np.array([-1.2386,  0.7196,  2.974 , -1.0484, -4.4469, -5.9738])
    dx=np.array([ 45.9095, 137.2909, -81.4492, -30.54  ,  -9.0298,  37.7497])
    from cmaes import CMA
    optimizer = CMA(mean=np.float64(dx), sigma=15)
    for generation in range(60):
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            value =evaluate.evaluate(x)
            solutions.append((x, np.float64(value)))
            print(f"#{generation} {value} (x1={x[0]}, x2 = {x[1]})")
        optimizer.tell(solutions)



    dx1=np.array([-1.2386,  0.7196,  2.974 , -1.0484, -4.4469, -5.9738, -2.6952,
         -4.374 ])
    

    dx2=np.array([-2.3684, -0.8973,  3.9837, -3.3336, -5.7216, -3.4934, -3.1598,
         -2.4684])
    
    dx3=np.array([-0.1227,  0.4567,  5.4507, -2.4195, -6.3413, -6.6535, -5.2577,
         -5.1175])
    
    dx4=np.array([ 13.6416,  38.854 ,  64.0544,  58.2893,  18.4691,  15.987 ,
          13.7576, -21.0037])
    
    dx5=np.array([ 45.9095, 137.2909, -81.4492, -30.54  ,  -9.0298,  37.7497]) #2116

    dx6 = np.array([ 54.262 , 153.5354, -97.7809, -53.5955, -12.3195,  29.3566]) #1875
    dx6 = np.array([ 54. , 153.5, -97., -53.5, -12.,  29.]) #2163
# A = pin.SE3.Identity()
# A.translation = np.array([0.23, -0.1, 0.2])
# B = pin.SE3.Identity()
# B.translation = np.array([0.23, -0.1, 0.5])

# eval = ArmEvaluator()

# Feval=np.array([0,0,10,0,0,0])




# dx=np.array([-90,0,-30,0,0,0])
# # eval.evaluateForce(dx)
# eval.evaluateManipuliability(dx)

# # self=eval


# A = pin.SE3.Identity()
# A.translation = np.array([0.23, -0.1, 0.2])
# B = pin.SE3.Identity()
# B.translation = np.array([0.23, -0.1, 0.5])
# Feval=np.array([0,0,10,0,0,0])

# self.tunemodel(dx,False)
# q=np.zeros(self.model.nq)
# cout1=self.evaluate_on_trajAB(A,B,q,Feval=Feval,disp=True)


# ## resultat different :


# self.tunemodel(dx,False)

# A = pin.SE3.Identity()
# A.translation = np.array([0.23, -0.1, 0.2])
# B = pin.SE3.Identity()
# B.translation = np.array([0.23, -0.1, 0.5])
# Feval=np.array([0,0,10,0,0,0])


# q=np.zeros(self.model.nq)
# cout1=self.evaluate_on_trajAB(A,B,q,Feval=Feval,disp=True)


# A = pin.SE3.Identity()
# A.translation = np.array([0.15, -0.1, 0.35])
# B = pin.SE3.Identity()
# B.translation = np.array([0.32, -0.1, 0.35])
# Feval=np.array([10,0,0,0,0,0])
# cout2=self.evaluate_on_trajAB(A,B,q,Feval=Feval,disp=True)


# eval.tunemodel(dx,True)
# q=np.zeros(eval.model.nq)
# A = pin.SE3.Identity()
# A.translation = np.array([0.2, -0.25, 0.25])
# B = pin.SE3.Identity()
# B.translation = np.array([0.2, -0.25, 0.55])
# t=eval.evaluate_on_trajAB_manipulability(A,B,q,disp=True)
# print(t)
# eval.evaluate_on_trajAB(A,B,q,Feval=Feval,disp=True)

# robot=loadUpperBody()
# q_init = robot.model.referenceConfigurations["half_sitting"]
# qsols= moveHandAB(robot,"hand",A,B,q_init=q_init)
# model=robot.model
# data=model.createData()
# actuation_model = robot.actuationModel
# actuation_data = ActuationData(model, [], actuation_model)






# vq=inverseConstraintKinematicsSpeed(model,data,[],[],actuation_model,actuation_data,qsols[0],14,np.array([0,0,0,0,0,0.1]))


# q=pin.integrate(model,qsols[0],vq*1e-2)
# qsols=[]
# for i in range(100):
#     velocity=[0,0,1,0,0,0]
#     pin.forwardKinematics(model, data, q, np.zeros(model.nv))
#     oMf=pin.updateFramePlacement(model,data,14)
    
#     vq=inverseConstraintKinematicsSpeed(model,data,[],[],actuation_model,actuation_data,q,14,oMf.inverse().action@velocity)
#     q=pin.integrate(model,q,vq*1e-3)
#     qsols.append(q)

# import meshcat
# from pinocchio.visualize import MeshcatVisualizer
# viz = MeshcatVisualizer(robot.model, robot.visual_model, robot.visual_model)
# viz.viewer = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
# viz.clean()
# viz.loadViewerModel(rootNodeName="universe")
# for q in qsols:
#     viz.display(q)


# from sympy import Matrix

# import time
# t_init=time.time()
# alphas=np.linspace(-1000,1000,1000)
# for i in range(100):
#     #find jacobian dor last configuration
#     vq=inverseConstraintKinematicsSpeed(model,data,[],[],actuation_model,actuation_data,q,14,oMf.inverse().action@velocity)
#     J= actuation_data.Jf_closed
    
#     #find nullspace
#     A=Matrix(J[:3,:4])
#     vq=np.array(A.nullspace()[0])
#     nvq=np.zeros(model.nv)
#     nvq[6:]=vq.T

#     #deduct new configuration
#     q=pin.integrate(model,q,nvq*1e-5)
#     # viz.display(q)
# t_end=time.time()   
# print("time sympy:",t_end-t_init)


# def gradient(q,forces):
#     inverseConstraintKinematicsSpeed(model,data,[],[],actuation_model,actuation_data,q,14,np.zeros(model.nv))
#     J= actuation_data.Jf_closed
#     cost1=J.T @ forces
#     A=Matrix(J[:3,:4])
#     vq=np.array(A.nullspace()[0])
#     nvq=np.zeros(model.nv)
#     nvq[6:]=vq.T

#     #deduct new configuration
#     q=pin.integrate(model,q,nvq*1e-5)    
#     inverseConstraintKinematicsSpeed(model,data,[],[],actuation_model,actuation_data,q,14,np.zeros(model.nv))
#     J= actuation_data.Jf_closed
#     cost2=J.T @ forces
#     grad=(cost2-cost1)/1e-5
#     return grad



# import time
# alphas=np.linspace(-1000,1000,1000)
# vq=np.zeros(model.nv)
# for alpha in alphas:
#     J= actuation_data.Jf_closed
#     v=oMf.inverse().action@velocity
#     A=J[:3,:4]

#     # vq=np.linalg.pinv(J)@v + (np.eye(4)-np.linalg.pinv(J[:3,:4])@J[:3,:4])*np.array([1,1,1,1])*alpha
#     mul=(np.eye(4)-np.linalg.pinv(A) @A)@np.array([1,1,1,1])
#     vq[6:]= np.linalg.pinv(J)@v + mul*alpha
#     cq=pin.integrate(model,q,vq*1e-3)
#     viz.display(cq)




# q = np.array([0.0, 0.0, 0.0, 0.0])

