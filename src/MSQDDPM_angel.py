import numpy as np
import tensorcircuit as tc
import scipy as sp
import torch
import torch.nn as nn

K = tc.set_backend("pytorch")
tc.set_dtype("complex64")

class DiffusionModel(nn.Module):
    '''
    The model that computes the forward pass. Depolarizing channel.
    '''
    def __init__(self, n, T, Ndata, device='cpu'):
        super().__init__()
        self.n = n
        self.T = T
        self.Ndata = Ndata
        self.device=device

    def _validate_timestep(self, t):
        if not (0 <= t <= self.T):
            raise ValueError(f"Time step t must be in the range [0, {self.T}]. Got t={t}.")

    def depolarizing_channel_t(self, rhos, t):
        '''Apply the depolarizing channel to the input density matrices `rhos` at time step `t`.
        Args:
            rhos: A batch of density matrices of shape (batch_size, 2^n, 2^n).
            t: The current time step (0 <= t <= T).
        Returns:
            A batch of density matrices after applying the depolarizing channel, of shape (batch_size, 2^n, 2^n).
        '''
        self._validate_timestep(t)
        prob = t / self.T
        m = 4**self.n
        prob = prob / m
        kraus_ops = tc.channels.generaldepolarizingchannel(prob, self.n)

        def circuit_fn(rho):
            c = tc.DMCircuit(self.n, dminputs=rho)
            c.general_kraus(kraus_ops, *range(self.n))
            return c.densitymatrix()
        
        vmapped_circuit_fn = K.vmap(circuit_fn, vectorized_argnums=0)
        return vmapped_circuit_fn(rhos)
    
    def density_matrix(self, states):
        '''Convert a batch of pure states to their corresponding density matrices.
        Args:
            states: [n_data, 2^n] A batch of pure states represented as state vectors.
        Returns:
            A batch of density matrices of shape (batch_size, 2^n, 2^n).
        '''
        def outer_product(psi):
            return tc.backend.outer_product(psi, tc.backend.conj(psi))
        all_rhos = tc.backend.vmap(outer_product)(states)
        return tc.backend.mean(all_rhos, axis=0)