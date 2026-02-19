# Importing necessary libraries and modules
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
# Importing custom utility functions and classes
# DEFT_initialization: Initializes mass, MOI, rod orientation, etc. for the BDLO
# construct_b_DLOs: Constructs undeformed states for the BDLO
# clamp_index: Builds the necessary clamp indices for boundary condition enforcement
# index_init: Initializes certain indexing variables for the model
# save_pickle: Utility function to save data (e.g., losses) to a pickle file
# Train_DEFTData / Eval_DEFTData: Custom dataset classes to load training/evaluation data
# DEFT_sim: Simulation model class
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deft.utils.util import DEFT_initialization, construct_b_DLOs, clamp_index, index_init, save_pickle, Train_DEFTData, Eval_DEFTData
from deft.core.DEFT_sim import DEFT_sim
from tqdm import tqdm
import os
import argparse
import matplotlib.pyplot as plt

# changed by felicia: 
import itertools
from collections import deque # import for grid search

class EarlyStoppingTracker:
    """Track eval losses and detect consistent increases"""
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience  # How many eval periods to wait
        self.min_delta = min_delta  # Minimum change to qualify as improvement
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
        self.loss_history = deque(maxlen=patience)
    
    def __call__(self, eval_loss):
        self.loss_history.append(eval_loss)
        
        if self.best_loss is None:
            self.best_loss = eval_loss
            return False
        
        # Check if loss improved
        if eval_loss < (self.best_loss - self.min_delta):
            self.best_loss = eval_loss
            self.counter = 0
            return False
        
        # Loss didn't improve
        self.counter += 1
        
        # Check if consistently increasing
        if len(self.loss_history) >= self.patience:
            recent_losses = list(self.loss_history)
            is_increasing = all(recent_losses[i] >= recent_losses[i-1] 
                              for i in range(1, len(recent_losses)))
            if is_increasing or self.counter >= self.patience:
                self.should_stop = True
                return True
        
        return False
        
def train(
    train_batch, 
    BDLO_type, 
    total_time, 
    train_time_horizon, 
    undeform_vis, 
    inference_vis, 
    inference_1_batch,
    residual_learning, 
    clamp_type, 
    load_model,
    dt: float = 0.02,
    data_dt: float = 0.01,
    frame_stride: int | None = None,
    integration_substeps: int = 2,
    # NEW PARAMETERS for grid search:
    bend_stiffness_parent_init=None,
    bend_stiffness_child1_init=None,
    bend_stiffness_child2_init=None,
    damping_parent_init=None,
    damping_child1_init=None,
    damping_child2_init=None,
    early_stopping_patience=5,  # changed: Number of eval periods before stopping
    training_case=5,  # changed: currently set to 5 for testing.  pass as parameter t incremenet during grid search
    grid_search=False,  # changed: flag to enable early stopping only during grid search
):  
    # total_time is the number of *effective* timesteps returned by the dataloader / simulated by the model.
    #
    # If the dataset pickles are recorded at `data_dt` (e.g. 0.01s) but we want to simulate at a larger
    # timestep `dt` (e.g. 0.05s), we downsample dataset frames by:
    #   frame_stride = dt / data_dt   (must be an integer)
    #
    # The dataset reader will then load `total_time * frame_stride` raw frames from disk and keep every
    # `frame_stride`-th frame, returning sequences of length `total_time` that match the simulation dt.
    if frame_stride is None:
        frame_stride = int(round(dt / data_dt))
    if frame_stride < 1:
        raise ValueError(f"frame_stride must be >= 1, got {frame_stride}")
    if not np.isclose(frame_stride * data_dt, dt):
        raise ValueError(
            f"dt ({dt}) must be an integer multiple of data_dt ({data_dt}). "
            f"Got frame_stride={frame_stride} -> {frame_stride * data_dt}."
        )
    if total_time < 3:
        raise ValueError(f"total_time must be >= 3, got {total_time}")
    
    # The total_time parameter is the maximum timesteps of the loaded data
    # The train_time_horizon is how many timesteps to unroll the simulation during training
    # The function trains or partially fine-tunes a DEFT model for a specific branched BDLO type

    eval_time_horizon = (total_time // frame_stride) - 2  # effective timesteps after downsampling  # Number of timesteps for evaluation

    # Explanation of notation in the code:
    # - undeformed_BDLO: A tensor containing the initial (undeformed) vertex positions of the branched BDLO
    # - n_parent_vertices / n_child1_vertices / n_child2_vertices: The number of vertices for the main branch, child1, and child2, respectively

    # Prepare BDLO-specific data depending on BDLO_type
    if BDLO_type == 1:
        # Set the undeformed shape of BDLO1 as a tensor of shape [1, 20, 3], then permute to [n_parent_vertices+..., 3]
        undeformed_BDLO = torch.tensor([[[-0.6790, -0.6355, -0.5595, -0.4539, -0.3688, -0.2776, -0.1857,
                                          -0.0991, 0.0102, 0.0808, 0.1357, 0.2081, 0.2404, -0.4279,
                                          -0.4880, -0.5394, -0.5559, 0.0698, 0.0991, 0.1125]],
                                        [[0.0035, -0.0066, -0.0285, -0.0349, -0.0704, -0.0663, -0.0744,
                                          -0.0957, -0.0702, -0.0592, -0.0452, -0.0236, -0.0134, -0.0813,
                                          -0.1233, -0.1875, -0.2178, -0.1044, -0.1858, -0.2165]],
                                        [[0.0108, 0.0104, 0.0083, 0.0104, 0.0083, 0.0145, 0.0133,
                                          0.0198, 0.0155, 0.0231, 0.0199, 0.0154, 0.0169, 0.0160,
                                          0.0153, 0.0090, 0.0121, 0.0205, 0.0155, 0.0148]]]).permute(1, 2, 0)

        # Number of vertices along the parent branch and the two child branches
        n_parent_vertices = 13
        n_child1_vertices = 5
        n_child2_vertices = 4

        # Depending on clamp_type, we set the train/eval dataset sizes and the selection of clamped vertices
        if clamp_type == "ends":
            train_set_number = 77
            eval_set_number = 24
            parent_clamped_selection = torch.tensor((0, 1, -2, -1))
            child1_clamped_selection = torch.tensor((2))
            child2_clamped_selection = torch.tensor((2))
        else:
            train_set_number = 71
            eval_set_number = 18
            parent_clamped_selection = torch.tensor((2, -2, -1))
            child1_clamped_selection = torch.tensor((2))
            child2_clamped_selection = torch.tensor((2))

        # cs_n_vert holds the number of child1 and child2 vertices
        cs_n_vert = (n_child1_vertices, n_child2_vertices)
        # n_vert is the parent branch vertex count
        n_vert = n_parent_vertices
        # Number of edges in the parent branch is n_vert - 1
        n_edge = n_vert - 1

        # Sanity check: parent branch should have more vertices than any child branch
        if n_parent_vertices <= max(cs_n_vert):
            raise Exception("warning: number of parent's vertices is larger than children's!")

        # # Define the stiffness parameters as nn.Parameters for optimization or subsequent usage
        # bend_stiffness_parent = nn.Parameter(4e-3 * torch.ones((1, 1, n_edge), device=device))
        # bend_stiffness_child1 = nn.Parameter(4e-3 * torch.ones((1, 1, n_edge), device=device))
        # bend_stiffness_child2 = nn.Parameter(4e-3 * torch.ones((1, 1, n_edge), device=device))
        # twist_stiffness = nn.Parameter(1e-4 * torch.ones((1, n_branch, n_edge), device=device))

        # # Damping parameters for each branch
        # damping = nn.Parameter(torch.tensor((2.5, 3, 3), device=device)) # changed (2.5, 2.5, 2.5)
        
        # changed by felicia: use grid search values if available 
        bs_parent_val = bend_stiffness_parent_init if bend_stiffness_parent_init else 4e-3
        bs_child1_val = bend_stiffness_child1_init if bend_stiffness_child1_init else 4e-3
        bs_child2_val = bend_stiffness_child2_init if bend_stiffness_child2_init else 4e-3
        damp_parent_val = damping_parent_init if damping_parent_init else 2.5
        damp_child1_val = damping_child1_init if damping_child1_init else 3.0
        damp_child2_val = damping_child2_init if damping_child2_init else 3.0

        bend_stiffness_parent = nn.Parameter(bs_parent_val * torch.ones((1, 1, n_edge), device=device))
        bend_stiffness_child1 = nn.Parameter(bs_child1_val * torch.ones((1, 1, n_edge), device=device))
        bend_stiffness_child2 = nn.Parameter(bs_child2_val * torch.ones((1, 1, n_edge), device=device))
        twist_stiffness = nn.Parameter(1e-4 * torch.ones((1, n_branch, n_edge), device=device)) # keep same
        damping = nn.Parameter(torch.tensor((damp_parent_val, damp_child1_val, damp_child2_val), device=device))

        # If we use residual learning, learning_weight is used to scale the residual from the GNN
        if residual_learning:
            learning_weight = nn.Parameter(torch.tensor(0.02, device=device))
        else:
            learning_weight = nn.Parameter(torch.tensor(0.00, device=device))

        # Indices that define which vertices couple the child branches to the parent
        rigid_body_coupling_index = [4, 8]

        # Mass and moment-of-inertia scaling factors
        parent_mass_scale = 1.
        parent_moment_scale = 10.
        moment_ratio = 0.1
        children_moment_scale = (0.5, 0.5)
        children_mass_scale = (1, 1)

    if BDLO_type == 2:
        # Similar initialization for BDLO2
        undeformed_BDLO = torch.tensor([
            [[0.0150, 0.0157, 0.0125, 0.0109, 0.0164, 0.0131, 0.0104,
              0.0081, 0.0083, 0.0079, 0.0093, 0.0108, 0.0150, 0.0109,
              0.0116, 0.0110, 0.0111, 0.0084, 0.0103, 0.0097]],
            [[0.1521, 0.1426, 0.1021, 0.0928, 0.0882, 0.0711, 0.0678,
              0.0894, 0.1109, 0.1374, 0.1708, 0.1855, 0.0339, -0.0410,
              -0.1058, -0.1266, 0.0465, -0.0155, -0.0733, -0.0954]],
            [[-0.1706, -0.1409, -0.0875, -0.0018, 0.0702, 0.1685, 0.2583,
              0.3363, 0.3894, 0.4529, 0.5106, 0.5355, 0.0704, 0.0615,
              0.0093, -0.0080, 0.3405, 0.3217, 0.2929, 0.2834]]
        ]).permute(1, 2, 0)

        n_parent_vertices = 12
        n_child1_vertices = 5
        n_child2_vertices = 5
        train_set_number = 110
        eval_set_number = 37

        cs_n_vert = (n_child1_vertices, n_child2_vertices)
        n_vert = n_parent_vertices
        n_edge = n_vert - 1
        if n_parent_vertices <= max(cs_n_vert):
            raise Exception("warning: number of parent's vertices is larger than children's!")

        # Stiffness parameters
        bend_stiffness_parent = nn.Parameter(2e-3 * torch.ones((1, 1, n_edge), device=device))
        bend_stiffness_child1 = nn.Parameter(1.5e-3 * torch.ones((1, 1, n_edge), device=device))
        bend_stiffness_child2 = nn.Parameter(1.5e-3 * torch.ones((1, 1, n_edge), device=device))
        twist_stiffness = nn.Parameter(1e-4 * torch.ones((1, n_branch, n_edge), device=device))

        # Damping parameters
        damping = nn.Parameter(torch.tensor((2.5, 2., 2.), device=device))

        if residual_learning:
            learning_weight = nn.Parameter(torch.tensor(0.02, device=device))
        else:
            learning_weight = nn.Parameter(torch.tensor(0.00, device=device))

        # Rigid body coupling index: the parent-children connection points
        rigid_body_coupling_index = [4, 7]

        # Mass, MOI scaling, etc.
        parent_mass_scale = 1.
        parent_moment_scale = 10.
        moment_ratio = 0.1
        children_moment_scale = (0.5, 0.5)
        children_mass_scale = (1, 1)

        # Which vertices are clamped in the dataset
        parent_clamped_selection = torch.tensor((0, 1, -2, -1))
        child1_clamped_selection = torch.tensor((2))
        child2_clamped_selection = torch.tensor((2))

    if BDLO_type == 3:
        # Initialization for BDLO3
        undeformed_BDLO = torch.tensor([
            [[0.0099, 0.0114, 0.0109, 0.0084, 0.0130, 0.0143, 0.0119,
              0.0133, 0.0135, 0.0136, 0.0136, 0.0151, 0.0124, 0.0093,
              0.0120, 0.0132, 0.0121]],
            [[-0.0444, -0.0684, -0.1235, -0.1722, -0.1973, -0.2265, -0.2232,
              -0.1956, -0.1675, -0.1150, -0.0544, -0.0249, -0.2632, -0.3340,
              -0.2580, -0.3370, -0.3594]],
            [[-0.5656, -0.5434, -0.4977, -0.4399, -0.3552, -0.2506, -0.1563,
              -0.0530, 0.0092, 0.0709, 0.1222, 0.1390, -0.3781, -0.4082,
              -0.0169, -0.0347, -0.0420]]
        ]).permute(1, 2, 0)

        n_parent_vertices = 12
        n_child1_vertices = 3
        n_child2_vertices = 4

        if clamp_type == "ends":
            train_set_number = 103
            eval_set_number = 26
            parent_clamped_selection = torch.tensor((0, 1, -2, -1))
            child1_clamped_selection = torch.tensor((2))
            child2_clamped_selection = torch.tensor((2))
        else:
            train_set_number = 39
            eval_set_number = 12
            parent_clamped_selection = torch.tensor((3, -2, -1))
            child1_clamped_selection = torch.tensor((2))
            child2_clamped_selection = torch.tensor((2))

        cs_n_vert = (n_child1_vertices, n_child2_vertices)
        n_vert = n_parent_vertices
        n_edge = n_vert - 1
        if n_parent_vertices <= max(cs_n_vert):
            raise Exception("warning: number of parent's vertices is larger than children's!")

        # Stiffness parameters
        bend_stiffness_parent = nn.Parameter(2.5e-3 * torch.ones((1, 1, n_edge), device=device, dtype=torch.float64))
        bend_stiffness_child1 = nn.Parameter(2.5e-3 * torch.ones((1, 1, n_edge), device=device, dtype=torch.float64))
        bend_stiffness_child2 = nn.Parameter(2.5e-3 * torch.ones((1, 1, n_edge), device=device, dtype=torch.float64))
        twist_stiffness = nn.Parameter(1e-4 * torch.ones((1, n_branch, n_edge), device=device, dtype=torch.float64))
        damping = nn.Parameter(torch.tensor((2., 2., 2.), device=device, dtype=torch.float64))

        if residual_learning:
            learning_weight = nn.Parameter(torch.tensor(0.02, device=device, dtype=torch.float64))
        else:
            learning_weight = nn.Parameter(torch.tensor(0.00, device=device, dtype=torch.float64))

        rigid_body_coupling_index = [4, 7]
        parent_mass_scale = 1.
        parent_moment_scale = 10.
        moment_ratio = 0.1
        children_moment_scale = (0.5, 0.5)
        children_mass_scale = (1, 1)

    if BDLO_type == 4:
        # Initialization for BDLO4
        undeformed_BDLO = torch.tensor([
            [[0.0108, 0.0122, 0.0112, 0.0116, 0.0116, 0.0169, 0.0122,
              0.0198, 0.0173, 0.0140, 0.0152, 0.0156, 0.0120, 0.0107,
              0.0163, 0.0154]],
            [[-0.1680, -0.1938, -0.2439, -0.2991, -0.3230, -0.3345, -0.3376,
              -0.3248, -0.3100, -0.2727, -0.2182, -0.1878, -0.3922, -0.4643,
              -0.3866, -0.4500]],
            [[-0.5774, -0.5491, -0.4909, -0.4085, -0.3219, -0.2371, -0.1568,
              -0.0645, 0.0231, 0.0828, 0.1411, 0.1664, -0.3430, -0.3652,
              0.0434, 0.0658]]
        ]).permute(1, 2, 0)

        n_parent_vertices = 12
        n_child1_vertices = 3
        n_child2_vertices = 3
        train_set_number = 74
        eval_set_number = 25

        cs_n_vert = (n_child1_vertices, n_child2_vertices)
        n_vert = n_parent_vertices
        n_edge = n_vert - 1
        if n_parent_vertices <= max(cs_n_vert):
            raise Exception("warning: number of parent's vertices is larger than children's!")

        # Stiffness parameters
        bend_stiffness_parent = nn.Parameter(3e-3 * torch.ones((1, 1, n_edge), device=device, dtype=torch.float64))
        bend_stiffness_child1 = nn.Parameter(4e-3 * torch.ones((1, 1, n_edge), device=device, dtype=torch.float64))
        bend_stiffness_child2 = nn.Parameter(4e-3 * torch.ones((1, 1, n_edge), device=device, dtype=torch.float64))
        twist_stiffness = nn.Parameter(1e-4 * torch.ones((1, n_branch, n_edge), device=device, dtype=torch.float64))

        # Damping for each of the 3 branches
        damping = nn.Parameter(torch.tensor((3., 4, 4.), device=device, dtype=torch.float64))

        if residual_learning:
            learning_weight = nn.Parameter(torch.tensor(0.02, device=device, dtype=torch.float64))
        else:
            learning_weight = nn.Parameter(torch.tensor(0.00, device=device, dtype=torch.float64))

        # Connection indices for the child branches
        rigid_body_coupling_index = [4, 8]

        parent_mass_scale = 1.
        parent_moment_scale = 10.
        moment_ratio = 0.1
        children_moment_scale = (0.5, 0.5)
        children_mass_scale = (1, 1)

        parent_clamped_selection = torch.tensor((0, 1, -2, -1))
        child1_clamped_selection = torch.tensor((2))
        child2_clamped_selection = torch.tensor((2))

    # Decide how many batches we use in evaluation (1 batch if inference_1_batch is True, else the entire eval_set_number)
    if inference_1_batch:
        eval_batch = 1
    else:
        eval_batch = eval_set_number

    # Number of vertices in the child branches
    n_children_vertices = (n_child1_vertices, n_child2_vertices)

    # Extract parent and child vertices from the undeformed BDLO
    parent_vertices_undeform = undeformed_BDLO[:, :n_parent_vertices]
    child1_vertices_undeform = undeformed_BDLO[:, n_parent_vertices: n_parent_vertices + n_children_vertices[0] - 1]
    child2_vertices_undeform = undeformed_BDLO[:, n_parent_vertices + n_children_vertices[0] - 1:]

    # DEFT_initialization returns scaled mass, MOI, rod orientations, nominal length, etc.
    b_DLO_mass, parent_MOI, children_MOI, parent_rod_orientation, children_rod_orientation, b_nominal_length = DEFT_initialization(
        parent_vertices_undeform,
        child1_vertices_undeform,
        child2_vertices_undeform,
        n_branch,
        n_parent_vertices,
        cs_n_vert,
        rigid_body_coupling_index,
        parent_mass_scale,
        parent_moment_scale,
        children_moment_scale,
        children_mass_scale,
        moment_ratio
    )

    # Construct the branched BDLO from data for the training batch
    # b_DLOs_vertices_undeform_untransform is the original set of undeformed states across the batch
    # The second return is an optional placeholder for transformations or expansions
    b_DLOs_vertices_undeform_untransform, _ = construct_b_DLOs(
        train_batch,
        rigid_body_coupling_index,
        n_parent_vertices,
        cs_n_vert,
        n_branch,
        parent_vertices_undeform,
        parent_vertices_undeform,
        child1_vertices_undeform,
        child1_vertices_undeform,
        child2_vertices_undeform,
        child2_vertices_undeform
    )

    # Transform the axis from local coordinate to global by re-indexing the coordinate axes
    b_DLOs_vertices_undeform_transform = torch.zeros_like(b_DLOs_vertices_undeform_untransform)
    b_DLOs_vertices_undeform_transform[:, :, :, 0] = -b_DLOs_vertices_undeform_untransform[:, :, :, 2]
    b_DLOs_vertices_undeform_transform[:, :, :, 1] = -b_DLOs_vertices_undeform_untransform[:, :, :, 0]
    b_DLOs_vertices_undeform_transform[:, :, :, 2] = b_DLOs_vertices_undeform_untransform[:, :, :, 1]

    # The first sample in the batch of undeformed vertices (reshape to [n_branch, n_vert, 3])
    b_undeformed_vert = b_DLOs_vertices_undeform_transform[0].view(n_branch, -1, 3)

    # Initialize index selection for parent MOI indices, etc.
    index_selection1, index_selection2, parent_MOI_index1, parent_MOI_index2 = index_init(
        rigid_body_coupling_index,
        n_branch
    )

    # Decide which vertices get clamped (i.e., fixed in space/rotation) for the training and evaluation sets
    clamped_index, parent_theta_clamp, child1_theta_clamp, child2_theta_clamp = clamp_index(
        train_batch,
        parent_clamped_selection,
        child1_clamped_selection,
        child2_clamped_selection,
        n_branch,
        n_parent_vertices,
        clamp_parent,
        clamp_child1,
        clamp_child2
    )
    eval_clamped_index, eval_parent_theta_clamp, eval_child1_theta_clamp, eval_child2_theta_clamp = clamp_index(
        eval_batch,
        parent_clamped_selection,
        child1_clamped_selection,
        child2_clamped_selection,
        n_branch,
        n_parent_vertices,
        clamp_parent,
        clamp_child1,
        clamp_child2
    )

    # Timestep for simulation
    dt = 0.02 # changed

    # Instantiate DEFT_sim objects for training and evaluation
    DEFT_sim_train = DEFT_sim(
        batch=train_batch,
        n_branch=n_branch,
        n_vert=n_vert,
        cs_n_vert=cs_n_vert,
        b_init_n_vert=b_undeformed_vert,
        n_edge=n_vert - 1,
        b_undeformed_vert=b_undeformed_vert,
        b_DLO_mass=b_DLO_mass,
        parent_DLO_MOI=parent_MOI,
        children_DLO_MOI=children_MOI,
        device=device,
        clamped_index=clamped_index,
        rigid_body_coupling_index=rigid_body_coupling_index,
        parent_MOI_index1=parent_MOI_index1,
        parent_MOI_index2=parent_MOI_index2,
        parent_clamped_selection=parent_clamped_selection,
        child1_clamped_selection=child1_clamped_selection,
        child2_clamped_selection=child2_clamped_selection,
        clamp_parent=clamp_parent,
        clamp_child1=clamp_child1,
        clamp_child2=clamp_child2,
        index_selection1=index_selection1,
        index_selection2=index_selection2,
        bend_stiffness_parent=bend_stiffness_parent,
        bend_stiffness_child1=bend_stiffness_child1,
        bend_stiffness_child2=bend_stiffness_child2,
        twist_stiffness=twist_stiffness,
        damping=damping,
        learning_weight=learning_weight
    )
    DEFT_sim_eval = DEFT_sim(
        batch=eval_batch,
        n_branch=n_branch,
        n_vert=n_vert,
        cs_n_vert=cs_n_vert,
        b_init_n_vert=b_undeformed_vert,
        n_edge=n_vert - 1,
        b_undeformed_vert=b_undeformed_vert,
        b_DLO_mass=b_DLO_mass,
        parent_DLO_MOI=parent_MOI,
        children_DLO_MOI=children_MOI,
        device=device,
        clamped_index=eval_clamped_index,
        rigid_body_coupling_index=rigid_body_coupling_index,
        parent_MOI_index1=parent_MOI_index1,
        parent_MOI_index2=parent_MOI_index2,
        parent_clamped_selection=parent_clamped_selection,
        child1_clamped_selection=child1_clamped_selection,
        child2_clamped_selection=child2_clamped_selection,
        clamp_parent=clamp_parent,
        clamp_child1=clamp_child1,
        clamp_child2=clamp_child2,
        index_selection1=index_selection1,
        index_selection2=index_selection2,
        bend_stiffness_parent=bend_stiffness_parent,
        bend_stiffness_child1=bend_stiffness_child1,
        bend_stiffness_child2=bend_stiffness_child2,
        twist_stiffness=twist_stiffness,
        damping=damping,
        learning_weight=learning_weight
    )

    # from Shicheng: Numerical integration substeps (stability vs. compute trade-off)
    DEFT_sim_train.integration_substeps = int(integration_substeps)
    DEFT_sim_eval.integration_substeps = int(integration_substeps)

    # Load pretrained models for initialization depending on BDLO_type and clamp_type
    if load_model:
        if BDLO_type == 1 and clamp_type == "ends":
            DEFT_sim_train.load_state_dict(torch.load("../save_model/BDLO1/DEFT_1_780_1.pth"), strict=False)
        if BDLO_type == 1 and clamp_type == "middle":
            DEFT_sim_train.load_state_dict(torch.load("../save_model/BDLO1/DEFT_middle_1_2260_1.pth"), strict=False)
        if BDLO_type == 2:
            DEFT_sim_train.load_state_dict(torch.load("../save_model/BDLO2/DEFT_2_820_2.pth"), strict=False)
        if BDLO_type == 3:
            DEFT_sim_train.load_state_dict(torch.load("../save_model/BDLO3/DEFT_3_40_3.pth"), strict=False)
        if BDLO_type == 3 and clamp_type == "middle":
            DEFT_sim_train.load_state_dict(torch.load("../save_model/BDLO3/DEFT_middle_3_2320_1.pth"), strict=False)
        if BDLO_type == 4:
            DEFT_sim_train.load_state_dict(torch.load("../save_model/BDLO4/DEFT_4_40_3.pth"), strict=False)
    
    # Re-apply substeps (defensive) in case future checkpoints touch the attribute
    DEFT_sim_train.integration_substeps = int(integration_substeps)
    DEFT_sim_eval.integration_substeps = int(integration_substeps)

    # If we want to visualize the undeformed states
    if undeform_vis:
        # Visualize the first batch's undeformed state
        first_batch_vertices = b_DLOs_vertices_undeform_transform[0]  # shape: [3, n_vert, 3]
        colors = ['red', 'green', 'blue']

        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        # For each branch, plot all vertex positions (here, just one set since it's "undeformed_vis")
        for branch_idx in range(3):
            branch_positions = first_batch_vertices[branch_idx, :, :]

            for vertex_idx in range(branch_positions.shape[0]):
                vertex_positions = branch_positions[vertex_idx, :]
                x = vertex_positions[0].unsqueeze(dim=0).numpy()
                y = vertex_positions[1].unsqueeze(dim=0).numpy()
                z = vertex_positions[2].unsqueeze(dim=0).numpy()
                ax.scatter(x, y, z, color=colors[branch_idx], alpha=1.)

        ax.set_xlim(-0.5, 1.0)
        ax.set_ylim(-0.5, 1.0)
        ax.set_zlim(-0.25, 1.25)
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        ax.set_title('Trajectories of Vertices Over Time (First Batch)')

        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color=colors[i], lw=2, label=f'Branch {i}') for i in range(3)]
        ax.legend(handles=legend_elements)
        plt.show()

    # Scale factor for learning rate (reduced to prevent NaN)
    lr_scale = 10

    # Define loss function
    loss_func = torch.nn.MSELoss()

    # We separate parameter sets based on whether we are doing residual learning or not
    gnn_params = DEFT_sim_train.GNN_tree.parameters()

    if not residual_learning:
        # If not using residual learning, we optimize all or most DEFT parameters
        parameters_to_update = [
            {"params": DEFT_sim_train.p_DLO_diagonal, "lr": 1e-6 * lr_scale},
            {"params": DEFT_sim_train.c_DLO_diagonal, "lr": 1e-6 * lr_scale},
            {"params": DEFT_sim_train.integration_ratio, "lr": 1e-6 * lr_scale},
            {"params": DEFT_sim_train.velocity_ratio, "lr": 1e-6 * lr_scale},
            {"params": DEFT_sim_train.undeformed_vert, "lr": 1e-6 * lr_scale},
            {"params": DEFT_sim_train.mass_diagonal, "lr": 1e-6 * lr_scale},
            {"params": DEFT_sim_train.damping, "lr": 1e-6 * lr_scale},
            {"params": DEFT_sim_train.gravity, "lr": 1e-6 * lr_scale},
            {"params": DEFT_sim_train.DEFT_func.twist_stiffness, "lr": 1e-6 * lr_scale},
            {"params": DEFT_sim_train.DEFT_func.bend_stiffness_parent, "lr": 1e-8 * lr_scale},
            {"params": DEFT_sim_train.DEFT_func.bend_stiffness_child1, "lr": 1e-8 * lr_scale},
            {"params": DEFT_sim_train.DEFT_func.bend_stiffness_child2, "lr": 1e-8 * lr_scale},
        ]
    else:
        # If using residual learning, we mainly update the GNN and the residual learning weight
        parameters_to_update = [
            {"params": DEFT_sim_train.learning_weight, "lr": 1e-7 * lr_scale},
            {"params": gnn_params, "lr": 1e-6 * lr_scale}
        ]

    # Check for NaN in initial parameters
    for name, param in DEFT_sim_train.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"Warning: NaN/Inf found in parameter {name}, reinitializing...")
            param.data = torch.randn_like(param.data) * 0.01
    
    # Define the optimizer (here, Adam for better stability) with the chosen parameters
    optimizer = optim.Adam(parameters_to_update, eps=1e-8)

    # We'll store evaluation results after certain intervals
    eval_epochs = []
    eval_losses = []

    # We'll store training results as well
    training_epochs = []
    training_losses = []

    # Loading training / evaluation data from custom datasets
    if clamp_type == "ends":
        eval_dataset = Eval_DEFTData(
            BDLO_type,
            n_parent_vertices,
            n_children_vertices,
            n_branch,
            rigid_body_coupling_index,
            eval_set_number,
            total_time,
            eval_time_horizon,
            device,
            frame_stride=frame_stride
        )
        eval_data_len = len(eval_dataset)
        train_dataset = Train_DEFTData(
            BDLO_type,
            n_parent_vertices,
            n_children_vertices,
            n_branch,
            rigid_body_coupling_index,
            train_set_number,
            total_time,
            train_time_horizon,
            device,
            frame_stride=frame_stride
        )
        train_data_loader = DataLoader(train_dataset, batch_size=train_batch, shuffle=True, drop_last=True)

    if clamp_type == "middle":
        # We load data from the dataset variant with middle clamps
        eval_dataset = Eval_DEFTData(
            str(BDLO_type) + "_mid_clamp",
            n_parent_vertices,
            n_children_vertices,
            n_branch,
            rigid_body_coupling_index,
            eval_set_number,
            total_time,
            eval_time_horizon,
            device,
            frame_stride=frame_stride
        )
        eval_data_len = len(eval_dataset)
        train_dataset = Train_DEFTData(
            str(BDLO_type) + "_mid_clamp",
            n_parent_vertices,
            n_children_vertices,
            n_branch,
            rigid_body_coupling_index,
            train_set_number,
            total_time,
            train_time_horizon,
            device,
            frame_stride=frame_stride
        )
        train_data_loader = DataLoader(train_dataset, batch_size=train_batch, shuffle=True, drop_last=True)

    # Define number of epochs to train
    train_epoch = 100 
    save_steps = 0
    evaluate_period = 20
    model = "DEFT"
    # training_case = 2 # comment out for grid search adjustments 

    # changed: Initialize early stopping tracker
    # If eval loss hasn't improved in the last `patience` evals, stop training this param combination
    early_stopping_tracker = EarlyStoppingTracker(
        patience=early_stopping_patience,
        min_delta=1e-6  # minimum change to count as loss improvement
    )

    # The main training loop
    if model == "DEFT":
        training_iteration = 0
        for epoch in range(train_epoch):
            bar = tqdm(train_data_loader)
            for data in bar:
                # Evaluate the model on the eval set periodically
                if save_steps % evaluate_period == 0:
                    part_eval = eval_set_number
                    # Random split for partial evaluation if desired
                    eval_set, test_set = torch.utils.data.random_split(eval_dataset,
                                                                       [part_eval, eval_data_len - part_eval])
                    eval_data_loader = DataLoader(eval_set, batch_size=eval_batch, shuffle=True, drop_last=True)

                    # Save the current model
                    torch.save(
                        DEFT_sim_train.state_dict(),
                        os.path.join("../save_model/", "DEFT_%s_%s_%s_%s.pth" % (
                        clamp_type, BDLO_type, str(training_iteration), training_case))
                    )
                    # Load the saved model into the evaluation simulation object
                    DEFT_sim_eval.load_state_dict(
                        torch.load("../save_model/DEFT_%s_%s_%s_%s.pth" % (
                        clamp_type, BDLO_type, str(training_iteration), training_case))
                    )

                    eval_bar = tqdm(eval_data_loader)
                    with torch.no_grad():
                        for eval_data in eval_bar:
                            # The evaluation data has previous, current, and target states
                            previous_b_DLOs_vertices_traj, b_DLOs_vertices_traj, target_b_DLOs_vertices_traj = eval_data
                            vis_type = "DEFT_%s" % BDLO_type
                            # Visualize on first iteration if inference_vis is True
                            if training_iteration == 0:
                                vis = inference_vis
                            else:
                                vis = False
                            # Perform iterative simulation over eval_time_horizon
                            traj_loss_eval, _ = DEFT_sim_eval.iterative_sim(
                                eval_time_horizon,
                                b_DLOs_vertices_traj,
                                previous_b_DLOs_vertices_traj,
                                target_b_DLOs_vertices_traj,
                                loss_func,
                                dt,
                                parent_theta_clamp,
                                child1_theta_clamp,
                                child2_theta_clamp,
                                inference_1_batch,
                                vis_type=vis_type,
                                vis=vis
                            )

                            # changed: After evaluation, check for early stopping in grid search
                            current_eval_loss = traj_loss_eval.cpu().detach().numpy() / total_time

                            # Print and record the average loss
                            print(np.sqrt(current_eval_loss))
                            eval_losses.append(current_eval_loss)
                            eval_epochs.append(training_iteration)

                            # Save the evaluation losses to pickle
                            save_pickle(eval_losses, "../training_record/eval_%s_loss_DEFT_%s_%s.pkl" % (
                            clamp_type, training_case, BDLO_type))
                            save_pickle(eval_epochs, "../training_record/eval_%s_epoches_DEFT_%s_%s.pkl" % (
                            clamp_type, training_case, BDLO_type))
                            
                            # changed to have minimum training iterations and only run during grid search
                            if grid_search and training_iteration >= 500 and early_stopping_tracker(current_eval_loss):
                                print(f"\nEarly stopping triggered at iteration {training_iteration}")
                                print(f"\nEval loss consistently increasing. Best loss: {early_stopping_tracker.best_loss}")
                                # Close progress bars before returning to prevent display issues
                                eval_bar.close()
                                bar.close()
                                return early_stopping_tracker.best_loss, True  # Return best loss and early stop flag

                # Increment steps and iteration
                save_steps += 1
                training_iteration += 1

                # Get the input data from the loader
                vis = False
                previous_b_DLOs_vertices_traj, b_DLOs_vertices_traj, target_b_DLOs_vertices_traj, m_u0_traj = data

                # Forward pass through the DEFT model for train_time_horizon timesteps
                traj_loss, total_loss = DEFT_sim_train.iterative_sim(
                    train_time_horizon,
                    b_DLOs_vertices_traj,
                    previous_b_DLOs_vertices_traj,
                    target_b_DLOs_vertices_traj,
                    loss_func,
                    dt,
                    parent_theta_clamp,
                    child1_theta_clamp,
                    child2_theta_clamp,
                    inference_1_batch,
                    vis_type=vis_type,
                    vis=vis
                )

                # Check for NaN in loss
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print(f"NaN/Inf detected at iteration {training_iteration}, skipping batch")
                    optimizer.zero_grad()
                    continue

                # Record and print training loss
                training_losses.append(traj_loss.cpu().detach().numpy() / train_time_horizon)
                training_epochs.append(training_iteration)

                # Backprop through the total loss
                total_loss.backward(retain_graph=True)
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(DEFT_sim_train.parameters(), max_norm=1.0)
                
                optimizer.step()
                optimizer.zero_grad()

                # Save training losses to pickle
                save_pickle(training_losses,
                            "../training_record/train_%s_loss_DEFT_%s_%s.pkl" % (clamp_type, training_case, BDLO_type))
                save_pickle(training_epochs,
                            "../training_record/train_%s_step_DEFT_%s_%s.pkl" % (clamp_type, training_case, BDLO_type))
    
    # Training completed without early stopping - return final metrics
    final_eval_loss = eval_losses[-1] if eval_losses else float('inf')
    return final_eval_loss, False  # False = didn't stop early

# changed: function to loop through grid search pairings 
def grid_search_train(base_args, param_grid):
    """
    Perform grid search over hyperparameters
    
    param_grid example:
    {
        'bend_stiffness_parent': [2e-3, 3e-3, 4e-3, 5e-3],
        'bend_stiffness_child1': [2e-3, 3e-3, 4e-3],
        'bend_stiffness_child2': [2e-3, 3e-3, 4e-3],
        'damping_parent': [2.0, 2.5, 3.0, 3.5],
        'damping_child1': [2.0, 2.5, 3.0, 3.5],
        'damping_child2': [2.0, 2.5, 3.0, 3.5],
    }
    """
    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(itertools.product(*param_values)) # 729 combinations
    
    results = []
    
    print(f"Starting grid search with {len(combinations)}...")
    
    for idx, param_combo in enumerate(combinations, start=3): # try training case 5
        config = dict(zip(param_names, param_combo))
        print(f"\n{'='*60}")
        print(f"Configuration {idx+1}/{len(combinations)}") # increase training case number
        print(f"Training Case: grid_search_{idx}")
        print(f"Parameters: {config}")
        print(f"{'='*60}\n")
        
        # call train with specific args
        final_eval_loss, stopped_early = train_with_config(base_args, config, idx)
        
        results.append({
            'config': config,
            'final_eval_loss': final_eval_loss,
            'stopped_early': stopped_early
        })
        
        # Save intermediate results (includes config index)
        save_pickle(results, "../training_record/grid_search_results_%s_case%s.pkl" % (base_args.BDLO_type, idx))
    
    # Find best configuration
    best_result = min(results, key=lambda x: x['final_eval_loss'])
    print(f"\n{'='*60}")
    print("GRID SEARCH COMPLETE")
    print(f"Best configuration: {best_result['config']}")
    print(f"Best eval loss: {best_result['final_eval_loss']}")
    print(f"{'='*60}\n")
    
    return results, best_result

# changed: specific train fuction for grid search config and increment training case
def train_with_config(base_args, param_config, config_index):
    """
    Modified train function that accepts parameter configuration
    and returns final eval loss + early stopping flag
    """
    # calls train with args from current grid search configuration
    # returns: final_eval_loss, stopped_early_flag
    final_eval_loss, stopped_early = train(
        train_batch=base_args.train_batch,
        BDLO_type=base_args.BDLO_type,
        total_time=base_args.total_time,
        train_time_horizon=base_args.train_time_horizon,
        undeform_vis=base_args.undeform_vis,
        inference_vis=base_args.inference_vis,
        inference_1_batch=base_args.inference_1_batch,
        residual_learning=base_args.residual_learning,
        clamp_type=base_args.clamp_type,
        load_model=base_args.load_model,
        dt=base_args.dt,
        data_dt=base_args.data_dt,
        frame_stride=base_args.frame_stride,
        integration_substeps=base_args.integration_substeps,
        # Grid search parameters:
        bend_stiffness_parent_init=param_config.get('bend_stiffness_parent'),
        bend_stiffness_child1_init=param_config.get('bend_stiffness_child1'),
        bend_stiffness_child2_init=param_config.get('bend_stiffness_child2'),
        damping_parent_init=param_config.get('damping_parent'),
        damping_child1_init=param_config.get('damping_child1'),
        damping_child2_init=param_config.get('damping_child2'),
        early_stopping_patience=5,
        training_case=config_index,  # make a new training case with each config
        grid_search=True,  # Enable early stopping for grid search
    )
    
    return final_eval_loss, stopped_early

if __name__ == "__main__":
    # Setting up a command-line interface for hyperparameters and options

    # Make sure to use double precision for stability in the DEFT simulations
    torch.set_default_dtype(torch.float64)
    # Ensure all tensor operations use float64
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(1)
    np.random.seed(1)

    # Limit the number of threads used by PyTorch and underlying libraries for reproducibility
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    # Initialize argument parser
    parser = argparse.ArgumentParser()

    # BDLO_type controls which BDLO dataset (and initial parameter sets) to use
    parser.add_argument("--BDLO_type", type=int, default=1)

    # clamp_type indicates how the BDLO is clamped (ends or middle)
    parser.add_argument("--clamp_type", type=str, default="ends")

     # total_time is the number of *effective* timesteps returned by the dataloader / simulated by the model.
    # Example: if dataset is sampled at data_dt=0.01 and you set dt=0.05 (frame_stride=5),
    # you should set total_time=100 to cover the same 5 seconds as 500 raw frames.
    parser.add_argument("--total_time", type=int, default=500)  # changed 500 -> 250

    # train_time_horizon is how many timesteps we simulate in each training iteration
    parser.add_argument("--train_time_horizon", type=int, default=10) # changed 50 -> 10 

    # Simulation timestep (seconds per step in the DEFT integrator)
    parser.add_argument("--dt", type=float, default=0.02)
    # Dataset sampling timestep (seconds per frame in the stored pickles)
    parser.add_argument("--data_dt", type=float, default=0.01)
    # Optional explicit downsampling stride (overrides dt/data_dt if provided)
    parser.add_argument("--frame_stride", type=int, default=None)

    # Numerical integration substeps per timestep (dt is split into dt/substeps).
    # Example: dt=0.02, integration_substeps=2 => internal dt_sub=0.01 for more stable dynamics.
    parser.add_argument("--integration_substeps", type=int, default=2)

    # Whether to visualize the initial undeformed vertices
    parser.add_argument("--undeform_vis", type=bool, default=False)

    # Whether we do inference only for 1 batch (for speed) or for all eval sets
    parser.add_argument("--inference_1_batch", type=lambda x: x.lower() == 'true', default=False)

    # Whether to enable residual learning: if True, GNN-based updates are used
    parser.add_argument("--residual_learning", type=bool, default=False)

    # Training batch size
    parser.add_argument("--train_batch", type=int, default=32)

    # Whether to visualize inference results (for debugging)
    parser.add_argument("--inference_vis", type=bool, default=False)

    # load trained model
    parser.add_argument("--load_model", type=bool, default=False)

    # changed by felicia: add grid search flag
    parser.add_argument("--grid_search", type=bool, default=False)

    # Parameters for training specific configurations (optional, overrides defaults)
    parser.add_argument("--bend_stiffness_parent", type=float, default=None)
    parser.add_argument("--bend_stiffness_child1", type=float, default=None)
    parser.add_argument("--bend_stiffness_child2", type=float, default=None)
    parser.add_argument("--damping_parent", type=float, default=None)
    parser.add_argument("--damping_child1", type=float, default=None)
    parser.add_argument("--damping_child2", type=float, default=None)

    # Flags for which branches are clamped
    clamp_parent = True
    clamp_child1 = False
    clamp_child2 = False

    # Number of branches for the BDLO (1 parent branch + 2 children branches)
    n_branch = 3

    # For simplicity, everything is done on CPU in this version
    device = "cpu"

    args = parser.parse_args()

    # Activate grid search if specified 
    if args.grid_search:
        # Define your parameter grid
        param_grid = {
            'bend_stiffness_parent': [3e-3, 4e-3, 5e-3],
            'bend_stiffness_child1': [3e-3, 4e-3, 5e-3],
            'bend_stiffness_child2': [3e-3, 4e-3, 5e-3],
            'damping_parent': [2.0, 2.5, 3.0],
            'damping_child1': [2.5, 3.0, 3.5],
            'damping_child2': [2.5, 3.0, 3.5],
        }
        
        results, best_config = grid_search_train(args, param_grid)
    else:
        # Call the training function with the user-specified arguments
        train(
            train_batch=args.train_batch,
            BDLO_type=args.BDLO_type,
            total_time=args.total_time,
            train_time_horizon=args.train_time_horizon,
            undeform_vis=args.undeform_vis,
            inference_vis=args.inference_vis,
            inference_1_batch=args.inference_1_batch,
            residual_learning=args.residual_learning,
            clamp_type=args.clamp_type,
            load_model=args.load_model,
            dt=args.dt,
            data_dt=args.data_dt,
            frame_stride=args.frame_stride,
            integration_substeps=args.integration_substeps,
            # Specific parameter configuration (if provided)
            bend_stiffness_parent_init=args.bend_stiffness_parent,
            bend_stiffness_child1_init=args.bend_stiffness_child1,
            bend_stiffness_child2_init=args.bend_stiffness_child2,
            damping_parent_init=args.damping_parent,
            damping_child1_init=args.damping_child1,
            damping_child2_init=args.damping_child2,
        )
