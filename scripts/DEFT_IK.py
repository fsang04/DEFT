import torch
import numpy as np
from pathlib import Path
from numpy.linalg import norm, solve
import matplotlib.pyplot as plt
import pinocchio as pin
import sys 
import os
import zipfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# load URDF into model and create data
model_dir = Path(__file__).parent.parent / "urdf"
filename = str(model_dir / "kinova.urdf")
kinova = pin.buildModelFromUrdf(filename)
data = kinova.createData()

# set parameters for iterative IK
eps = 1e-4 # tol
IT_MAX = 5000
DT = 1e-2
damp = 1e-2 # damping for iter step

def IK(ee_pos_des, q_init):
    # for visualization
    q_history = []
    ee_pos_history = []
    error_history = []

    q_curr = np.array(q_init)
    q_history.append(q_curr.copy())
    ee_frame_id = kinova.getFrameId("end_effector") # fixed joints represented as frames
    success = False

    for i in range(IT_MAX):
        pin.forwardKinematics(kinova, data, q_curr)
        pin.updateFramePlacement(kinova, data, ee_frame_id) # or use framesForwardsKinematics
        ee_pos = data.oMf[ee_frame_id].translation # retrieve current ee world position
        
        pos_err = ee_pos - ee_pos_des # compute position error 
        # if i % 100 == 0:
        #     print(f"Iteration {i}: error = {norm(pos_err):.6f} m")
        if norm(pos_err) < eps:
            success = True
            break

        # compute jacobian for frame 
        jac = pin.computeFrameJacobian(
            kinova, data, q_curr, ee_frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        ) # 6x7

        # set orientation error = 0 to keep fixed ee orientation
        err = np.hstack([pos_err, np.zeros(3)])  # 6x1

        # use damped pseudoinverse
        v = -jac.T.dot(solve(jac.dot(jac.T) + damp * np.eye(6), err)) # 7x6 * 6x6 * 6x1 -> 7x1
        
        # update joint angles with step
        q_curr = pin.integrate(kinova, q_curr, v * DT) 

        q_history.append(q_curr.copy())
        ee_pos_history.append(ee_pos.copy())
        error_history.append(norm(pos_err))

    if success:
        print(f"\nConvergence achieved with final joint configuration {q_curr.flatten().tolist()}")
        print(f"\nFinal error: {err}")
    else:
        print("\nConvergence was not reached.")

    return q_curr, q_history, ee_pos_history, error_history

def animate(q_history, ee_pos_history, error_history, ee_pos_des):
    # visualize convergence 
    fig = plt.figure(figsize=(16, 6))
    
    # Left subplot: 3D trajectory
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Plot end effector trajectory
    ee_traj = np.array(ee_pos_history)
    ax1.plot(ee_traj[:,0], ee_traj[:,1], ee_traj[:,2], 'b-', linewidth=2, label='EE Trajectory')
    
    # Plot starting position
    ax1.scatter(*ee_traj[0], c='green', marker='o', s=200, label='Start Position')
    
    # Plot target position
    ax1.scatter(*ee_pos_des, c='red', marker='*', s=300, label='Target Position')
    
    # Labels and legend
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('IK Solver: End Effector Trajectory')
    ax1.legend()
    
    # Right subplot: Error convergence
    ax2 = fig.add_subplot(122)
    ax2.plot(error_history, 'r-', linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Position Error (m)')
    ax2.set_title('Convergence: Error vs Iteration')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')  # log scale shows convergence better
    
    plt.tight_layout()
    plt.savefig('ik_plot.png', dpi=300, bbox_inches='tight')
    print("Figure saved to ik_plot.png")

if __name__ == "__main__":

    # load prerecorded waypoints 
    zip_path = Path(__file__).parent.parent / 'Sep16waypoint.zip'
    intermediate_waypoint = np.deg2rad(np.array([325.56,16.28,178.43,291.64,15.29,284.82,52.85]))
    intermediate_waypoint = intermediate_waypoint.astype(np.float32)

    case_number = 1 # for testing
    file_name = f'Sep16waypoint/projected_kinova_case{case_number}_waypoints.npy'
    
    print(f"loading waypoints from case {case_number}...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        with zf.open(file_name) as file:
            config_list = np.load(file)
    
    print(f"loaded {len(config_list)} waypoints")
    # Print all waypoints
    print("\nAll waypoint configurations:")
    np.set_printoptions(precision=3, suppress=True)
    print(config_list)
    
    # extract initial and final joint configs from waypoints
    q_init = config_list[0].astype(np.float64)  # inital config
    # q_init = intermediate_waypoint.astype(np.float64)  # inital config at intermediate waypoint
    q_final = config_list[-1].astype(np.float64)  # final config
    print(f"\ninitial joint configuration: {q_init}")
    print(f"final joint configuration: {q_final}")

    # use FK to get target EE position in final config
    ee_frame_id = kinova.getFrameId("end_effector")
    pin.framesForwardKinematics(kinova, data, q_final)
    ee_pos_des = data.oMf[ee_frame_id].translation.copy()
    print(f"\ndesired EE position: {ee_pos_des}")
   
    # use FK to get initial EE positionn
    pin.framesForwardKinematics(kinova, data, q_init)
    ee_pos_init = data.oMf[ee_frame_id].translation.copy()
    
    print(f"initial EE position: {ee_pos_init}")
    # print(f"EE displacement needed: {norm(ee_pos_des - ee_pos_init):.4f} m")
    
    # solve IK to recover desired final joint angles from target EE position
    print("\n" + "="*60)
    print("TESTING IK SOLVER")
    print("="*60)
    print(f"Target EE pos: {ee_pos_des}")
    print(f"Starting from: {ee_pos_init}")
    
    q_solution, q_hist, ee_hist, err_hist = IK(ee_pos_des, q_init)
    
    # verify solution by comparing with original final config
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)
    print(f"Original final config: {q_final}")
    print(f"IK solution config:    {q_solution}")
    print(f"Joint angle error:     {norm(q_solution - q_final):.6f} rad")
    
    # check final EE position
    pin.framesForwardKinematics(kinova, data, q_solution)
    ee_pos_achieved = data.oMf[ee_frame_id].translation
    print(f"\nTarget EE position:   {ee_pos_des}")
    print(f"Achieved EE position: {ee_pos_achieved}")
    print(f"EE position error:    {norm(ee_pos_achieved - ee_pos_des):.6e} m")

    animate(q_hist, ee_hist, err_hist, ee_pos_des)


# --------------------------

# def IK(ee_pos_des, q_curr):
# Input: desired ee position
# Output: desired joint angles
# iterative jacobian solver: 
# 1. Compute current ee pos with FK(q_curr)
# 2. Loop:
    # a. Compute error: pos_err = ee_pos_des - ee_pos_curr
    # b. If within tol of target, return q_curr
    # b. compute Jacboian at current config 
    # c. compute joint angle update
    # d. update current joint angles
# (did not converge): return q_curr

# def FK(q):
# input: current joint angles 
# output: current ee position

# def compute_jacobian:
# numerical or analytical?
# use library? need robot model?