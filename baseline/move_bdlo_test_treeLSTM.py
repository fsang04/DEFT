#!/usr/bin/env python3
"""
Load BDLO1 data and run shape-matching tests on the Tree-LSTM baseline model.
Mirrors the structure of move_bdlo_test_fixed_end.py but replaces DEFT_sim with
BDLOTreeLSTM and iterative_predict() with a manual autoregressive rollout.
"""

import torch
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
try:
    import cyipopt
    IPOPT_AVAILABLE = True
except ImportError:
    IPOPT_AVAILABLE = False
    print("Warning: cyipopt not available. Install with: pip install cyipopt")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deft.utils.util import Eval_DEFTData, Test_DEFTData
from tree_lstm import BDLOTreeLSTM
from torch.utils.data import DataLoader
from itertools import islice
from datetime import datetime
from pathlib import Path

# Global simulation time configuration
SIM_TIME_HORIZON = 100  # Adjust this to change simulation duration

# BDLO1 / "ends" clamp configuration (matches train_tree_lstm.py BDLO_CONFIGS)
N_PARENT = 13
CS_N_VERT = (5, 4)
COUPLING = [4, 8]
HIDDEN_SIZE = 128
PARENT_CLAMP = [0, 1, -2, -1]   # same as torch.tensor((0, 1, -2, -1))


def setup_tree_lstm(load_checkpoint=None):
    """Create a BDLOTreeLSTM model and optionally load a saved checkpoint."""
    torch.set_default_dtype(torch.float64)
    device = "cpu"

    model = BDLOTreeLSTM(
        hidden_size=HIDDEN_SIZE,
        n_parent_vertices=N_PARENT,
        cs_n_vert=CS_N_VERT,
        rigid_body_coupling_index=COUPLING,
        input_size=9,   # position(3) + velocity(3) + clamped_target_hint(3)
    )
    model = model.double()
    model = model.to(device)

    if load_checkpoint:
        state_dict = torch.load(load_checkpoint, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint from {load_checkpoint}")

    return model


def _build_hints(clamped_positions_at_t, parent_clamp_indices, batch, n_branch, n_vert, device, dtype):
    """
    Build a clamped-target-hint tensor from a slice of the planned trajectory.

    Args:
        clamped_positions_at_t: [batch, n_branch, n_vert, 3]  - full-size slice at time t
        parent_clamp_indices: list/tensor of clamped parent vertex indices
    Returns:
        hints: [batch, n_branch, n_vert, 3]  (zeros everywhere except clamped parent verts)
    """
    hints = torch.zeros(batch, n_branch, n_vert, 3, device=device, dtype=dtype)
    for idx in parent_clamp_indices:
        hints[:, 0, idx] = clamped_positions_at_t[:, 0, idx]
    return hints


def treelstm_rollout(model, b_DLOs_vertices_traj, previous_b_DLOs_vertices_traj,
                     clamped_full, parent_clamp_indices, time_horizon):
    """
    Autoregressive rollout of the Tree-LSTM model.

    At each step t the model receives:
      - current vertex positions
      - previous vertex positions (finite-difference velocity proxy)
      - clamped-target hints taken from clamped_full[:, t]

    After each step the clamped parent vertices are hard-overwritten with their
    planned positions (same as enforce_clamps in train_tree_lstm.py).

    Args:
        model: BDLOTreeLSTM (eval mode expected)
        b_DLOs_vertices_traj:          [batch, T_data, n_branch, n_vert, 3]
        previous_b_DLOs_vertices_traj: [batch, T_data, n_branch, n_vert, 3]
        clamped_full: [batch, time_horizon, n_branch, n_vert, 3]
            Planned clamped-vertex positions for each rollout step.
        parent_clamp_indices: list/tensor of clamped parent vertex indices
        time_horizon: int

    Returns:
        predicted: [batch, time_horizon, n_branch, n_vert, 3]
    """
    batch  = b_DLOs_vertices_traj.shape[0]
    n_branch = b_DLOs_vertices_traj.shape[2]
    n_vert   = b_DLOs_vertices_traj.shape[3]
    device = b_DLOs_vertices_traj.device
    dtype  = b_DLOs_vertices_traj.dtype

    prev = previous_b_DLOs_vertices_traj[:, 0]   # [batch, n_branch, n_vert, 3]
    curr = b_DLOs_vertices_traj[:, 0]             # [batch, n_branch, n_vert, 3]

    steps = []
    for t in range(time_horizon):
        hints = _build_hints(clamped_full[:, t], parent_clamp_indices,
                             batch, n_branch, n_vert, device, dtype)
        pred = model(curr, prev, clamped_target_hints=hints)

        # Hard-enforce clamped positions
        for idx in parent_clamp_indices:
            pred[:, 0, idx] = clamped_full[:, t, 0, idx]

        steps.append(pred.unsqueeze(1))
        prev = curr
        curr = pred

    return torch.cat(steps, dim=1)   # [batch, time_horizon, n_branch, n_vert, 3]


def create_custom_trajectory(initial_state, time_horizon, parent_clamped_selection, pulled_end):
    """Create custom control trajectory for clamped vertices only"""
    # Extract only clamped vertices: [batch, time, n_clamped, 3]
    batch = initial_state.shape[0]
    n_clamped = len(parent_clamped_selection)
    custom_traj = torch.zeros(batch, time_horizon, n_clamped, 3, dtype=torch.float64)
    
    # Initialize with initial positions of clamped vertices
    initial_clamped = initial_state[0, 0, 0, parent_clamped_selection]
    
    # Simple linear motion: first two stay fixed, last two move slowly in +x direction
    for t in range(time_horizon):
        custom_traj[0, t] = initial_clamped.clone()
        
        # Move last two vertices slowly in x direction
        progress = t / time_horizon
        if pulled_end == 1:
            custom_traj[0, t, -2, 0] += 0.3 * progress
            custom_traj[0, t, -1, 0] += 0.3 * progress
        if pulled_end == 0:
            custom_traj[0, t, 0, 0] -= 0.3 * progress
            custom_traj[0, t, 1, 0] -= 0.3 * progress
    return custom_traj

def expand_move_disp_to_4(move_disp_3, pulled_end: int):
    """Expand 3 DOF to 12 DOF. Moving ends has non-zero movement while the fixing end has zero movement."""
    # move_disp_3: shape (3,)
    move_disp_3 = move_disp_3.reshape(3)
    z = torch.zeros_like(move_disp_3)
    if pulled_end == 0:
        return torch.stack([move_disp_3, move_disp_3, z, z], dim=0)  # [4,3]
    elif pulled_end == 1:
        return torch.stack([z, z, move_disp_3, move_disp_3], dim=0)
    else:
        raise ValueError(f"pulled_end must be 0 or 1, got {pulled_end}")

def compute_gradient_wrt_control(model, b_DLOs_vertices_traj, previous_b_DLOs_vertices_traj,
                                  target_vertices, custom_contr ol, pulled_end):
    """Compute gradient of final position cost w.r.t. reduced 3 DOF displacement"""
    batch = 1
    parent_clamped_selection = torch.tensor((0, 1, 11, 12))

    # Get initial and final positions from custom_control
    initial_clamped = custom_control[0, 0]  # [n_clamped, 3]
    final_clamped = custom_control[0, -1]   # [n_clamped, 3]

    # only count the moving side
    if pulled_end == 0:
        move_disp0 = ((final_clamped[0] - initial_clamped[0]) + (final_clamped[1] - initial_clamped[1])) / 2
    elif pulled_end == 1:
        move_disp0 = ((final_clamped[2] - initial_clamped[2]) + (final_clamped[3] - initial_clamped[3])) / 2
    else:
        raise ValueError(f"pulled_end must be 0 or 1, got {pulled_end}")
    move_disp = move_disp0.clone().requires_grad_(True)   # shape [3]
    final_displacements = expand_move_disp_to_4(move_disp, pulled_end)  # [4,3]

    # Interpolate trajectory from initial to final
    clamped_full = torch.zeros(batch, SIM_TIME_HORIZON, 3, 13, 3, dtype=torch.float64)
    for t in range(SIM_TIME_HORIZON):
        alpha = t / (SIM_TIME_HORIZON - 1)
        interpolated = initial_clamped + alpha * final_displacements
        for i, idx in enumerate(parent_clamped_selection):
            clamped_full[0, t, 0, idx] = interpolated[i]

    predicted_vertices = treelstm_rollout(
        model, b_DLOs_vertices_traj, previous_b_DLOs_vertices_traj,
        clamped_full, PARENT_CLAMP, SIM_TIME_HORIZON
    )

    final_pred = predicted_vertices[0, -1]
    final_target = target_vertices[0, -1]
    cost = torch.sum((final_pred - final_target) ** 2)

    # Gradient w.r.t. reduced 3 DOF
    grad = torch.autograd.grad(cost, move_disp)[0]

    return grad.detach(), cost.item()

def finite_difference_gradient(model, b_DLOs_vertices_traj, previous_b_DLOs_vertices_traj,
                               target_vertices, custom_control, pulled_end, eps=1e-6):
    """Compute gradient using finite differences for verification (reduced 3 DOF)"""
    parent_clamped_selection = torch.tensor((0, 1, -2, -1))

    # Get initial and final positions
    initial_clamped = custom_control[0, 0]
    final_clamped = custom_control[0, -1]

    # only count the moving side
    if pulled_end == 0:
        move_disp0 = ((final_clamped[0] - initial_clamped[0]) + (final_clamped[1] - initial_clamped[1])) / 2
    elif pulled_end == 1:
        move_disp0 = ((final_clamped[2] - initial_clamped[2]) + (final_clamped[3] - initial_clamped[3])) / 2
    else:
        raise ValueError(f"pulled_end must be 0 or 1, got {pulled_end}")

    def compute_cost(move_disp_3):
        expanded = expand_move_disp_to_4(move_disp_3, pulled_end)  # [4,3]

        with torch.no_grad():
            clamped_full = torch.zeros(1, SIM_TIME_HORIZON, 3, 13, 3, dtype=torch.float64)
            for t in range(SIM_TIME_HORIZON):
                alpha = t / (SIM_TIME_HORIZON - 1)
                interpolated = initial_clamped + alpha * expanded
                for i, idx in enumerate(parent_clamped_selection):
                    clamped_full[0, t, 0, idx] = interpolated[i]

            pred = treelstm_rollout(
                model, b_DLOs_vertices_traj, previous_b_DLOs_vertices_traj,
                clamped_full, PARENT_CLAMP, SIM_TIME_HORIZON
            )
            return torch.sum((pred[0, -1] - target_vertices[0, -1]) ** 2).item()

    fd_grad = torch.zeros_like(move_disp0)  # shape [3]

    test_indices = [0, 1, 2]
    for idx in test_indices:
        disp_plus = move_disp0.clone()
        disp_plus[idx] += eps
        cost_plus = compute_cost(disp_plus)

        disp_minus = move_disp0.clone()
        disp_minus[idx] -= eps
        cost_minus = compute_cost(disp_minus)

        fd_grad[idx] = (cost_plus - cost_minus) / (2 * eps)

    return fd_grad, test_indices, eps

class DEFTOptimizationProblem:
    """IPOPT problem wrapper for Tree-LSTM trajectory optimization"""
    def __init__(self, model, b_DLOs_vertices_traj, previous_b_DLOs_vertices_traj,
                 target_vertices, pulled_end):
        self.model = model
        # Disable gradients for model parameters during optimization
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.b_DLOs_vertices_traj = b_DLOs_vertices_traj.detach()
        self.previous_b_DLOs_vertices_traj = previous_b_DLOs_vertices_traj.detach()
        self.target_vertices = target_vertices.detach()
        self.pulled_end = pulled_end    # 0 = head end moves, 1 = tail end moves
        self.parent_clamped_selection = torch.tensor((0, 1, 11, 12))
        self.initial_clamped = b_DLOs_vertices_traj[0, 0, 0, self.parent_clamped_selection].detach().clone()
        self.n = 3  # only allow one end to move; 3 DOF
        self.iteration = 0
        self.converged = False
        self.best_solution = None  # Store solution when converged
        
    def _expand_displacements(self, x):
        """Expand 3 DOF to 12 DOF by coupling vertex pairs and give zeros to unmoved vertex"""
        disp_pair = x[0]
        fixed_pair = torch.zeros_like(disp_pair)

        if self.pulled_end == 0:
            return torch.stack([disp_pair, disp_pair, fixed_pair, fixed_pair])
        elif self.pulled_end == 1:
            return torch.stack([fixed_pair, fixed_pair, disp_pair, disp_pair])
        else:
            raise ValueError("pulled_end only allows 0 for pulling head of the parent branche, 1 for pulling the end.")
        
    def objective(self, x):
        """Compute objective: final configuration error"""
        final_displacements = self._expand_displacements(torch.from_numpy(x.reshape(1, 3)))

        # Build trajectory
        clamped_full = torch.zeros(1, SIM_TIME_HORIZON, 3, 13, 3, dtype=torch.float64)
        for t in range(SIM_TIME_HORIZON):
            alpha = t / (SIM_TIME_HORIZON - 1)
            interpolated = self.initial_clamped + alpha * final_displacements
            for i, idx in enumerate(self.parent_clamped_selection):
                clamped_full[0, t, 0, idx] = interpolated[i]

        with torch.no_grad():
            predicted_vertices = treelstm_rollout(
                self.model, self.b_DLOs_vertices_traj, self.previous_b_DLOs_vertices_traj,
                clamped_full, PARENT_CLAMP, SIM_TIME_HORIZON
            )
        
        cost = torch.sum((predicted_vertices[0, -1] - self.target_vertices[0, -1]) ** 2)
        
        # Compute endpoint distances
        pred_ends = predicted_vertices[0, -1, 0, self.parent_clamped_selection]
        target_ends = self.target_vertices[0, -1, 0, self.parent_clamped_selection]
        distances = torch.norm(pred_ends - target_ends, dim=1)
        
        # print(f"  [Iter {self.iteration}] Endpoint distances: {distances.numpy()}")
        
        # Early stopping: all endpoints within 0.02
        if torch.all(distances < 0.025) and not self.converged:
            print(f"  *** Converged! All endpoints within 0.025 ***")
            self.converged = True
            self.best_solution = x.copy()  # Save the converged solution

        self.iteration += 1
        return cost.item()
    
    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                    d_norm, regularization_size, alpha_du, alpha_pr, ls_trials):
        """Intermediate callback for early stopping"""
        if self.converged:
            return False  # Stop optimization
        return True  # Continue optimization
    
    def gradient(self, x):
        """Compute gradient using PyTorch autograd"""
        # Re-enable grad for model params temporarily for autograd to flow through
        for param in self.model.parameters():
            param.requires_grad = False

        reduced_displacements = torch.from_numpy(x.reshape(1, 3)).double().requires_grad_(True)

        final_displacements = self._expand_displacements(reduced_displacements)

        clamped_full = torch.zeros(1, SIM_TIME_HORIZON, 3, 13, 3, dtype=torch.float64)
        for t in range(SIM_TIME_HORIZON):
            alpha = t / (SIM_TIME_HORIZON - 1)
            interpolated = self.initial_clamped + alpha * final_displacements
            for i, idx in enumerate(self.parent_clamped_selection):
                clamped_full[0, t, 0, idx] = interpolated[i]

        predicted_vertices = treelstm_rollout(
            self.model, self.b_DLOs_vertices_traj, self.previous_b_DLOs_vertices_traj,
            clamped_full, PARENT_CLAMP, SIM_TIME_HORIZON
        )

        cost = torch.sum((predicted_vertices[0, -1] - self.target_vertices[0, -1]) ** 2)

        # Gradient w.r.t. reduced 3 DOF
        grad = torch.autograd.grad(cost, reduced_displacements)[0]
        return grad.detach().numpy().flatten()

def trajectory_optimization_ipopt(model, b_DLOs_vertices_traj, previous_b_DLOs_vertices_traj,
                                  target_vertices, pulled_end):
    """Optimize trajectory using IPOPT"""
    if not IPOPT_AVAILABLE:
        raise ImportError("cyipopt not available. Install with: pip install cyipopt")

    # Create problem
    problem_obj = DEFTOptimizationProblem(
        model, b_DLOs_vertices_traj, previous_b_DLOs_vertices_traj,
        target_vertices, pulled_end
    )
    
    n = 3  # 1 independent displacements * 3 coords
    lb = [-1e20] * n
    ub = [1e20] * n
    
    # Initial guess: zero displacements (stay at initial positions)
    x0 = np.zeros(n)
    
    # Create IPOPT problem
    problem = cyipopt.Problem(
        n=n,
        m=0,  # No constraints
        problem_obj=problem_obj,
        lb=lb,
        ub=ub
    )
    
    # Set options
    problem.add_option("print_level", 5)
    problem.add_option("tol", 1e-3)
    problem.add_option("max_iter", 10)
    
    print("\nStarting IPOPT optimization...")
    solution, info = problem.solve(x0)
    
    # Use saved solution if early stopping occurred
    if problem_obj.best_solution is not None:
        print("Using early-stopped solution (distances < 0.025)")
        solution = problem_obj.best_solution
    
    print(f"\nIPOPT Status: {info['status_msg']}")
    print(f"Final objective: {info['obj_val']:.6f}")
    
    # Generate optimized trajectory for visualization
    # changed to only 3 DOF
    final_displacements = problem_obj._expand_displacements(torch.from_numpy(solution.reshape(1, 3)))
    clamped_full = torch.zeros(1, SIM_TIME_HORIZON, 3, 13, 3, dtype=torch.float64)
    for t in range(SIM_TIME_HORIZON):
        alpha = t / (SIM_TIME_HORIZON - 1)
        interpolated = problem_obj.initial_clamped + alpha * final_displacements
        for i, idx in enumerate(problem_obj.parent_clamped_selection):
            clamped_full[0, t, 0, idx] = interpolated[i]
    
    with torch.no_grad():
        optimized_traj = treelstm_rollout(
            problem_obj.model, b_DLOs_vertices_traj, previous_b_DLOs_vertices_traj,
            clamped_full, PARENT_CLAMP, SIM_TIME_HORIZON
        )

    return solution.reshape(1, 3), info, optimized_traj

def move_bdlo_with_data(checkpoint_path=None):
    """Load real data from dataset and run a preview prediction"""
    torch.set_default_dtype(torch.float64)
    device = "cpu"

    # Setup Tree-LSTM model with checkpoint
    model = setup_tree_lstm(load_checkpoint=checkpoint_path)
    model.eval()

    # Load evaluation dataset
    dt = 0.01
    data_dt = 0.01
    total_time = 500
    frame_stride = int(round(dt / data_dt))
    eval_time_horizon = total_time // frame_stride - 2
    eval_dataset = Eval_DEFTData(
        BDLO_type=1,
        n_parent_vertices=13,
        n_children_vertices=(5, 4),
        n_branch=3,
        rigid_body_coupling_index=[4, 8],
        eval_set_number=1,
        total_time=total_time,
        eval_time_horizon=eval_time_horizon,
        device=device
    )

    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    # Get one sample
    previous_b_DLOs_vertices_traj, b_DLOs_vertices_traj, target_b_DLOs_vertices_traj = next(islice(eval_loader, 20, None))

    # Determine which end is being pulled
    parent = b_DLOs_vertices_traj[0, :, 0]          # [T, 13, 3]
    head = parent[:, 0]                              # [T, 3]
    end  = parent[:, -1]                             # [T, 3]
    head_motion = torch.norm(head[1:] - head[:-1], dim=1).mean()
    end_motion  = torch.norm(end[1:]  - end[:-1],  dim=1).mean()
    pulled_end = 0 if head_motion > end_motion else 1

    parent_clamped_selection = torch.tensor((0, 1, -2, -1))
    custom_control = create_custom_trajectory(b_DLOs_vertices_traj, SIM_TIME_HORIZON, parent_clamped_selection, pulled_end)

    print(f"Processing 1 sample: {b_DLOs_vertices_traj.shape}")

    with torch.no_grad():
        batch = 1
        clamped_full = torch.zeros(batch, SIM_TIME_HORIZON, 3, 13, 3, dtype=torch.float64)
        clamped_full[:, :, 0, parent_clamped_selection] = custom_control

        predicted_vertices = treelstm_rollout(
            model, b_DLOs_vertices_traj, previous_b_DLOs_vertices_traj,
            clamped_full, PARENT_CLAMP, SIM_TIME_HORIZON
        )

    print(f"Prediction completed: {predicted_vertices.shape}")
    return predicted_vertices, None, target_b_DLOs_vertices_traj

def animate_prediction(predicted_vertices, target_vertices, skip_frames=5, title_prefix="Prediction"):
    """Create animation showing optimization process moving toward static target"""
    pred = predicted_vertices[0]  # [time, branch, vert, 3]
    target = target_vertices[0, -1]  # [branch, vert, 3] - static final configuration
    
    print(f"\n[Animation] Target: target_vertices[0, -1]")
    print(f"  Shape: {target.shape}")
    print(f"  Branch 0 endpoints: {target[0, [0,1,11,12]]}")
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['red', 'green', 'blue']
    
    def update(frame):
        ax.clear()
        
        # Plot static target (faded)
        for i in range(3):
            verts = target[i]
            mask = torch.any(verts != 0, dim=-1)
            valid = verts[mask]
            if len(valid) > 0:
                ax.plot(valid[:, 0], valid[:, 1], valid[:, 2], 'o-', color=colors[i], 
                    linewidth=2, alpha=0.3, label=f'Target {i}')
        
        # Overlay current prediction (solid)
        for i in range(3):
            verts = pred[frame, i]
            mask = torch.any(verts != 0, dim=-1)
            valid = verts[mask]
            if len(valid) > 0:
                ax.plot(valid[:, 0], valid[:, 1], valid[:, 2], 'o-', color=colors[i], 
                    linewidth=2, markersize=6, label=f'{title_prefix} {i}')
        
        ax.set_title(f'{title_prefix} (solid) → Target (faded) - Frame {frame}')
        ax.set_xlim(-0.8, 0.4)
        ax.set_ylim(-0.3, 0.3)
        ax.set_zlim(-0.1, 0.3)
        ax.legend()
    
    frames = range(0, pred.shape[0], skip_frames)
    anim = FuncAnimation(fig, update, frames=frames, interval=50)
    plt.tight_layout()
    return anim

if __name__ == "__main__":
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checkpoint_path = os.path.join(repo_root, "save_model", "TreeLSTM_ends_1_100_1.pth")

    print("Loading and previewing with trained Tree-LSTM model...")
    pred_verts, _, target = move_bdlo_with_data(checkpoint_path=checkpoint_path)
    print(f"Done!")

    # Load model for test loop
    print("\nRunning shape-matching tests...")
    torch.set_default_dtype(torch.float64)
    model = setup_tree_lstm(load_checkpoint=checkpoint_path)
    model.eval()

    # Data loading must match dt=0.02 sampling
    dt, data_dt = 0.02, 0.01
    frame_stride = int(round(dt / data_dt))  # 2
    total_time = 500
    eval_time_horizon = total_time // frame_stride - 2

    test_dataset = Test_DEFTData(
        BDLO_type=1, n_parent_vertices=13, n_children_vertices=(5, 4),
        n_branch=3, rigid_body_coupling_index=[4, 8],
        eval_set_number=1, total_time=total_time, eval_time_horizon=eval_time_horizon,
        device="cpu", frame_stride=frame_stride
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    number_test = len(test_loader)
    success_count = 0
    fail_list = []

    anim_dir = Path(__file__).resolve().parent / "testing_animations_TreeLSTM_ends_1_100_1"
    anim_dir.mkdir(parents=True, exist_ok=True)

    for i, (previous_b_DLOs_vertices_traj, b_DLOs_vertices_traj, target_b_DLOs_vertices_traj, pulled_end) in enumerate(test_loader):

        parent_clamped_selection = torch.tensor((0, 1, -2, -1))
        custom_control = create_custom_trajectory(b_DLOs_vertices_traj, SIM_TIME_HORIZON, parent_clamped_selection, pulled_end)

        grad, cost = compute_gradient_wrt_control(
            model, b_DLOs_vertices_traj, previous_b_DLOs_vertices_traj,
            target_b_DLOs_vertices_traj, custom_control, pulled_end
        )
        print(f"Cost: {cost:.6f}")
        print(f"Gradient shape: {grad.shape}  # [3,] = 3 DOF (1 independent displacement)")
        print(f"Gradient norm: {torch.norm(grad).item():.6f}")

        # Verify with finite differences
        fd_grad, test_indices, eps = finite_difference_gradient(
            model, b_DLOs_vertices_traj, previous_b_DLOs_vertices_traj,
            target_b_DLOs_vertices_traj, custom_control, pulled_end
        )

        print(f"\nGradient comparison (eps={eps}):")
        for idx in test_indices:
            ag = grad[idx].item()
            fd = fd_grad[idx].item()
            print(f"  Coord {idx}: Autograd={ag:.6f}, FiniteDiff={fd:.6f}, Diff={abs(ag-fd):.6f}")

        # Run IPOPT optimization
        if IPOPT_AVAILABLE:
            print("\n" + "="*60)
            print("IPOPT TRAJECTORY OPTIMIZATION (Tree-LSTM)")
            print("="*60)
            import time
            start_time = time.time()
            optimized_displacements, info, optimized_traj = trajectory_optimization_ipopt(
                model, b_DLOs_vertices_traj, previous_b_DLOs_vertices_traj,
                target_b_DLOs_vertices_traj, pulled_end
            )
            elapsed_time = time.time() - start_time

            print(f"\nOptimized displacements shape: {optimized_displacements.shape}")
            print(f"\n*** OPTIMIZATION TIME: {elapsed_time:.3f} seconds ***")

            pred_ends = optimized_traj[0, -1, 0, parent_clamped_selection]   # shape [4,3]
            target_ends = target_b_DLOs_vertices_traj[0, -1, 0, parent_clamped_selection]
            distances = torch.norm(pred_ends - target_ends, dim=1)

            if torch.all(distances < 0.025):
                status = "success"
                success_count += 1
                print(f"[{i+1}/{number_test}] success, distances={distances.cpu().numpy()}")
            else:
                status = "fail"
                fail_list.append(i + 1)
                print(f"[{i+1}/{number_test}] fail, distances={distances.cpu().numpy()}")

            anim_name = f"iter_{i+1:04d}_{status}.mp4"
            anim_path = anim_dir / anim_name
            print(f"\nCreating animation for iteration {i+1} ...")
            anim = animate_prediction(optimized_traj, target_b_DLOs_vertices_traj, title_prefix=f"Testing sample {i+1}")
            anim.save(str(anim_path), writer="ffmpeg", fps=20)
            print(f"Animation saved to: {anim_path}")
            plt.close()

        else:
            print("\nIPOPT not available. Skipping IPOPT optimization.")

    summary1 = f"Success count: {success_count} out of {number_test} tests."
    summary2 = f"Fail cases are (one-based): {fail_list}"

    print(summary1)
    print(summary2)

    log_path = "move_bdlo_summary.txt"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n===== {timestamp} =====\n")
        f.write(summary1 + "\n")
        f.write(summary2 + "\n")