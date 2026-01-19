### inverse kinematics 


import can
import numpy as np
import time
import pinocchio as pin
from example_parallel_robots.loader_tools import completeRobotLoader
from toolbox_parallel_robots.mounting import closedLoopMountProximal,closedLoopMountScipy,closedLoopMountCasadi
from toolbox_parallel_robots.inverse_kinematics import closedLoopInverseKinematicsProximal,closedLoopInverseKinematicsScipy
import meshcat
from pinocchio.visualize import MeshcatVisualizer

import os
import time
import threading
import tkinter as tk


CWD=os.getcwd()
# CWD = os.path.dirname(os.path.abspath(__file__))

pin.SE3.__repr__ = pin.SE3.__str__


model,constraint_models,actuation_model,visual_model,colision_model=completeRobotLoader(CWD + '/urdf/humanoid_v1/urdf', freeflyer=False)

viz = MeshcatVisualizer(model, visual_model, visual_model)
viz.viewer = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
viz.clean()
viz.loadViewerModel(rootNodeName="universe")
viz.display(pin.neutral(model))

def ikine_leg(target_pos,q_prec=None):
    if q_prec is None:
        q_prec=pin.neutral(model)
    data=model.createData()
    pin.framesForwardKinematics(model,data,q_prec)
    foot_frame_id=model.getFrameId("foot")
    foot_frame=model.frames[foot_frame_id]

    target_M=pin.SE3(np.eye(3),target_pos)
    # constraint=pin.RigidConstraintModel(pin.ContactType.CONTACT_6D,model,foot_frame.parentJoint,foot_frame.placement,0,target_M,pin.ReferenceFrame.LOCAL)    
    # cmodels=[constraint]
    # cdata=[c.createData() for c in cmodels]
    q=closedLoopInverseKinematicsProximal(model,data,[],[],target_M,"foot",q_prec=q_prec.tolist())
    return q


def pin2robot(q_pin): 
    q_robot=np.zeros(6) 
    q_robot[0]=q_pin[0]%2*np.pi 
    q_robot[1]=q_pin[0]%2*np.pi 
    q_robot[2]=q_pin[0]%2*np.pi 
    q_robot[3]=q_pin[0]%2*np.pi 
    q_robot[4]=q_pin[5]%2*np.pi-q_pin[4]%2*np.pi 
    q_robot[5]=q_pin[5]%2*np.pi+q_pin[4]%2*np.pi 
    return q_robot



target_pos=np.array([0.0,-0.1,-0.5])
q_sol=np.array([np.pi,  np.pi,  -np.pi/2,   4.50133864,
         5.80062565,  -5.98131086])


q_sol=ikine_leg(target_pos,q_sol)
viz.display(q_sol)

# ----------------------------
# Tkinter UI state (thread-safe-ish)
# ----------------------------
state = {
    "x": 0.0,
    "y": -0.15,
    "z": -0.50,
    "dirty": True,
    "running": True,
}
lock = threading.Lock()


def start_tk():
    root = tk.Tk()
    root.title("IK target sliders")






    def mk_slider(label, key, from_, to, resolution, init,orient="horizontal"):
        frame = tk.Frame(root)
        frame.pack(fill="x", padx=10, pady=6)

        tk.Label(frame, text=label, width=8, anchor="w").pack(side="left")

        var = tk.DoubleVar(value=init)

        def on_change(_val):
            with lock:
                state[key] = float(var.get())
                state["dirty"] = True

        s = tk.Scale(
            frame,
            variable=var,
            from_=from_,
            to=to,
            resolution=resolution,
            length=420,
            command=on_change,
            orient=orient,
        )
        s.pack(side="left", padx=8)

        val_label = tk.Label(frame, text=f"{init:.3f}", width=10, anchor="e")
        val_label.pack(side="left")

        def update_label():
            val_label.config(text=f"{var.get():.3f}")
            root.after(50, update_label)

        update_label()
        return s

    mk_slider("z (m)", "z", -0.80, -0.20, 0.001, state["z"],orient="vertical")
    mk_slider("x (m)", "x", -0.30, 0.30, 0.001, state["x"])
    mk_slider("y (m)", "y", -0.35, 0.05, 0.001, state["y"])
    

    # Button row
    btns = tk.Frame(root)
    btns.pack(fill="x", padx=10, pady=10)

    def reset():
        # remet les valeurs; on déclenche dirty
        with lock:
            state["x"] = 0.0
            state["y"] = -0.15
            state["z"] = -0.50
            state["dirty"] = True
        # Note: on ne force pas la position des sliders ici (possible, mais plus long).
        # Si tu veux le "vrai reset visuel", dis-moi et je te le fais.

    tk.Button(btns, text="Reset (logic)", command=reset).pack(side="left")

    def on_close():
        with lock:
            state["running"] = False
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


# ----------------------------
# Run UI in separate thread
# ----------------------------
ui_thread = threading.Thread(target=start_tk, daemon=True)
ui_thread.start()

# ----------------------------
# Main loop: recompute IK + display when dirty
# ----------------------------
print("Tkinter sliders opened. Close the window to stop.")

last_tuple = None
try:
    while True:
        with lock:
            running = state["running"]
            dirty = state["dirty"]
            x, y, z= state["x"], state["y"], state["z"]

            if dirty:
                state["dirty"] = False

        if not running:
            break

        cur = (x, y, z)
        if cur != last_tuple:
            target = np.array([x, y, z])
            q_sol = ikine_leg(target,q_sol)
            viz.display(q_sol)
            last_tuple = cur

        time.sleep(1.0 / max(1.0, float(25)))

except KeyboardInterrupt:
    pass

print("Stopped.")
