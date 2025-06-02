import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Seeding for reproducibility
# SEED = 24
# torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps")


class FeatureAttribution:
    """
    Implements the Stieltjes feature attribution from 'Feature Attribution from First Principles'.
    """

    def __init__(self, model, input_dim, input_space, integrands = None):
        self.integrands = integrands 
        self.model = model
        self.input_space = input_space
        self.input_dim = input_dim


    def stieltjes_attribution_density(self, explained_point, N_points=10):
        """
        Another example of sampler:
        Importance sampling to compute the Stieltjes attribution for a given input x.
        The integrands should admit a density.
        """
        assert self.input_space[0] == 0 and self.input_space[1] == 1, "input space should be [0, 1]"

        grid_points = torch.rand(N_points, self.input_dim) # shape: [N_points, self.input_dim]
        model_output_tensor = self.model(grid_points) # shape: [N_points]
        density_tensor = torch.zeros((self.input_dim, N_points), device=grid_points.device) # shape: [self.input_dim = len(self.integrands), N_points]
        for i, integrand in enumerate(self.integrands):
            if integrand is None:
                continue
            density_tensor[i] = self.integrands[i](grid_points, explained_point)

        aproximate_integral = torch.sum(model_output_tensor * density_tensor, dim = -1) / N_points

        return aproximate_integral
    

    def _sampler(self, x, N_points=10):
        """
        Sampler should be overrided, with the sampler you want. E.g.: Rejection sampling, Gibbs sampling, MCMC, generative models, etc...
        We give a uniform distribution on [0,1] as an example for all the features.
        """
        # sample_points = should be a tensor having n_integrands as first dimension as the density for each dimension can vary.
        sample_points = torch.rand((self.input_dim, N_points, self.input_dim)) # shape: [input_dim = n_integrands, N_points, input_dim]
        return sample_points
    
    def stieltjes_attribution(self, explained_point, is_montecarlo=False, N_points=10, batch_size=None):
        """
        Compute the Stieltjes attribution for a given input x.
        The function approximates an integral over the input space using either:
        - A Monte Carlo approach if is_montecarlo=True, or
        - A Riemann sum approximation with batching.
        
        For the Riemann sum, the grid is built in a batched manner by converting flat indices to 
        multi-indices (yielding batch_points), and a similar batching strategy is applied to compute
        the integrand increments over the 2**(input_dim) epsilon corners.

        The computation are based on Theorem B.4 from the paper "Feature Attribution from First Principles". 
        All the variables are defined in Theorem B.4.
        """
        with torch.no_grad():
            #### Monte Carlo Approximation
            # In this case, the sampler should be build using the density of self.integrand.
            if is_montecarlo:
                grid_points_tensor = self._sampler(explained_point, N_points=N_points) # shape: [input_dim = n_integrands, N_points, input_dim]
                # treat the two first dimensions as a batch dimension and the last dimension as the input dimension.
                grid_points_tensor = grid_points_tensor.reshape(self.input_dim * N_points, self.input_dim) # shape: [input_dim * N_points, input_dim]
                # Evaluate the model on these grid points.
                model_output_tensor = self.model(grid_points_tensor).reshape(self.input_dim, N_points) # shape: [input_dim = n_integrands, N_points]
                
                # tensor of all the approximated integrals for all integrands.
                approximate_integral = torch.sum(model_output_tensor, dim = -1) / N_points # shape: [input_dim = n_integrands]

                return approximate_integral

            #### Riemann Sum Approximation with Batching
            total_cells = N_points ** self.input_dim
            if batch_size is None:
                batch_size = total_cells  # process all cells at once if memory allows

            # Create a 1D partition of the input space.
            partition = torch.linspace(self.input_space[0], self.input_space[1], N_points + 1)

            # Precompute the epsilon tensor for hyperrectangle corners.
            # Each epsilon vector is in {0, 1}^d.
            zero_one_tensors = [torch.tensor([0, 1], device=partition.device) for _ in range(self.input_dim)]
            epsilon_tensor = torch.cartesian_prod(*zero_one_tensors)  # shape: [2**d, d]
            epsilon_signs = (-1) ** torch.sum(epsilon_tensor, dim=-1)  # shape: [2**d]

            # Initialize accumulators for each integrand.
            approximate_integrals = [0.0 for _ in self.integrands]

            # Process grid cells in batches.
            for batch_start in tqdm(range(0, total_cells, batch_size)):
                batch_end = min(total_cells, batch_start + batch_size)
                batch_flat_indices = torch.arange(batch_start, batch_end, device=partition.device)

                # Convert flat indices to multi-indices.
                # For each dimension d, compute: (flat_index // (N_points**d)) % N_points.
                multi_indices = []
                for d in range(self.input_dim):
                    multi_index_d = (batch_flat_indices // (N_points ** d)) % N_points
                    multi_indices.append(multi_index_d)
                multi_indices = torch.stack(multi_indices, dim=-1)  # shape: [batch_size, input_dim]

                # Convert multi-indices to grid points (lower corners of the hyperrectangles).
                batch_points = torch.stack(
                    [partition[multi_indices[:, d]] for d in range(self.input_dim)],
                    dim=-1
                )  # shape: [batch_size, input_dim]

                # Evaluate the model on these grid points.
                model_output_batch = self.model(batch_points).squeeze()  # shape: [batch_size]

                # Compute increments using epsilon tensor in a batched manner
                # For each grid cell, determine the 2**(input_dim) corner indices.
                multi_indices_for_increment = multi_indices.unsqueeze(1) + epsilon_tensor.unsqueeze(0)
                # Map the multi-indices to actual points by indexing into the partition.
                points_for_increment = torch.stack(
                    [partition[multi_indices_for_increment[..., d]] for d in range(self.input_dim)],
                    dim=-1
                )  # shape: [batch_size, 2**(input_dim), input_dim]

                # Loop over each integrand function provided.
                for i, integrand in enumerate(self.integrands):
                    if integrand is None:
                        # ignore None integrands
                        continue
                    # Evaluate the integrand at all corner points for the current batch.
                    # Expected shape: [batch_size, 2**(input_dim)]
                    integrand_values = integrand(points_for_increment, explained_point)
                    # Compute the weighted sum over the corners using the precomputed epsilon signs.
                    # This gives a contribution for each grid cell.
                    increment_values = torch.sum(epsilon_signs * integrand_values, dim=-1)  # shape: [batch_size]
                    # Accumulate the contribution weighted by the model output at the corresponding grid cell.
                    approximate_integrals[i] += torch.sum(model_output_batch * increment_values).item()

            return approximate_integrals
