from networkx import omega
import numpy as np
import tensorcircuit as tc
import scipy as sp
import torch
import torch.nn as nn
import ot
from functools import partial

K = tc.set_backend("pytorch")
tc.set_dtype("complex64")

class MSQDDPMDifusser(nn.Module):
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
    
    def density_matrix_from_pure_states_ensemble(self, states):
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
    
    def density_matrices_ensemble_from_pure_states_ensemble(self, states):
        '''Convert a batch of pure states to their corresponding density matrices, without averaging.
        Args:
            states: [n_data, 2^n] A batch of pure states represented as state vectors.
        Returns:
            A batch of density matrices of shape (batch_size, 2^n, 2^n).
        '''
        def outer_product(psi):
            return tc.backend.outer_product(psi, tc.backend.conj(psi))
        all_rhos = tc.backend.vmap(outer_product)(states)
        return all_rhos




class MSQDDPM(nn.Module):
    def __init__(self, n, n_ancilla_qubits, n_haar_ancilla_qubits, T, L, seed, device='cpu'):
        super().__init__()
        self.n = n
        self.na = n_ancilla_qubits
        self.na_haar = n_haar_ancilla_qubits
        assert self.na_haar == 1 # Currently only support 1 ancilla qubit in the Haar random state.
        self.n_tot = n + n_haar_ancilla_qubits + n_ancilla_qubits
        self.T = T
        self.L = L
        self.seed = seed
        self.device = device
        # embed the circuit to a vectorized pytorch neural network layer
        self.backCircuit_vmap = K.vmap(partial(backCircuit, n_tot=self.n_tot, L=L), vectorized_argnums=0)
    
    def _generate_haar_states(self, n_data):
        '''Generate a batch of Haar random states of just 1 qubit.
        Returns: [n_data, 2]'''
        torch.manual_seed(self.seed)
        params = torch.rand(n_data, 3)
        phi = 2 * torch.pi * params[:, 0]
        omega = 2 * torch.pi * params[:, 1]
        sin_sampler = sin_prob_dist(a=0, b=np.pi)
        theta = sin_sampler.rvs(size=1)[0]

        def circuit_fn(p):
            c = tc.Circuit(1)
            c.rot(0, phi=p[0], theta=p[1], omega=p[2])
            return c.state()
        vmap_circuit = tc.vmap(circuit_fn)
        return vmap_circuit(torch.stack([phi, theta, omega], dim=-1))

    def _randomMeasure(self, inputs):
        m_probs = (torch.abs(inputs.reshape(inputs.shape[0], 2**self.na, 2**self.n))**2).sum(dim=2) # Compute probs of measuring each ancilla state (marginal probs over the data qubits)
        m_res = torch.multinomial(m_probs, num_samples=1).squeeze() # Measurement results. The index of the measured ancilla state for each input state. (e.g. 1 stands for |01> in the ancilla space)
        
        indices = 2**self.n * m_res.view(-1, 1) + torch.arange(2**self.n, device=self.device) # Compute the indices to select the corresponding data qubit states based on the measurement results
        post_state = torch.gather(inputs, 1, indices) # Select the corresponding data qubit states based on the measurement results. This gives us the post-measurement state of the data qubits, conditioned on the measurement outcome of the ancilla qubits.
        norms = torch.sqrt(torch.sum(torch.abs(post_state)**2, axis=1)).unsqueeze(dim=1)
        return 1./norms * post_state
    
    def set_diffusionSet(self, states_diff):
        self.states_diff = torch.from_numpy(states_diff).to(self.device).cfloat()

    def backwardOutput_t(self, inputs, params):
        output_full = self.backCircuit_vmap(inputs, params)
        output_t = self._randomMeasure(output_full) # Measure the ancilla qubits (and ditch the results)
        return output_t
    
    def prepareInput_t(self, inputs_T, params_tot, t, Ndata):
        '''The input for t step has to be the tensor product of a Haar random state and na ancilla qubits.'''
        if self.na_haar == 1:
            # |Haar> \otimes |0>^(n_zero_ancilla_qubits) \otimes |data>
            haar_states = self._generate_haar_states(Ndata) # 1. |Haar> : Shape [Ndata, 2]
            ancilla_zero = torch.zeros(2**self.na, device=self.device, dtype=torch.complex64) # 2. |0>^na : Shape [2**na]
            ancilla_zero[0] = 1.0
            prefix = torch.vmap(lambda h: torch.kron(h, ancilla_zero))(haar_states) # 3. |Haar> \otimes |0>^na
            self.input_tplus1 = torch.vmap(torch.kron)(prefix, inputs_T) # 4. (|Haar> \otimes |0>^na) \otimes |Data>

        elif self.na_haar == 0:
            ancilla_zero = torch.zeros(2**self.na, device=self.device, dtype=torch.complex64) # |0>^na : Shape [2**na]
            ancilla_zero[0] = 1.0
            self.input_tplus1 = torch.vmap(torch.kron)(ancilla_zero, inputs_T) # (|0>^na) \otimes |Data>
        
        else:
            # |0>^(n_zero_ancilla_qubits) \otimes |data>
            self.input_tplus1 = torch.zeros((Ndata, 2**self.n_tot), device=self.device).cfloat()
            self.input_tplus1[:,:2**self.n] = inputs_T
            with torch.no_grad():
                for tt in range(self.T-1, t, -1):
                    self.input_tplus1[:,:2**self.n] = self.backwardOutput_t(self.input_tplus1, params_tot[tt])

        return self.input_tplus1

def backCircuit(input, params, n_tot, L):
    c = tc.Circuit(n_tot, inputs=input)
    for l in range(L):
        for i in range(n_tot):
            c.rx(i, theta=params[2*n_tot*l+i])
            c.ry(i, theta=params[2*n_tot*l+n_tot+i])
        for i in range(n_tot//2):
            c.cz(2*i, 2*i+1)
        for i in range((n_tot-1)//2):
            c.cz(2*i+1, 2*i+2)
    return c.state()

def compute_superfidelity(Set1, Set2):
    '''
        calculate the superfidelity between two sets of quantum states
    '''
    term1 = torch.real(torch.einsum('imn,knm->ik', Set1, Set2))

    purity1 = torch.real(torch.einsum('imn,inm->i', Set1, Set1))
    purity2 = torch.real(torch.einsum('kmn,knm->k', Set2, Set2))
    diff1 = 1.0 - purity1  # [N1]
    diff2 = 1.0 - purity2  # [N2]
    inner_val = diff1[:, None] * diff2[None, :]
    term2 = torch.sqrt(torch.clamp(inner_val, min=0.0))

    G = term1 + term2
    return G

def WassDistance(Set1, Set2):
    '''
        calculate the Wasserstein distance between two sets of quantum states
        the cost matrix is the superfidelity between sets S1, S2
    '''
    superfidelity = compute_superfidelity(Set1, Set2)
    D = 1. - superfidelity
    emt = torch.empty(0)
    Wass_dis = ot.emd2(emt, emt, M=D)
    return Wass_dis

def sinkhornDistance(Set1, Set2, reg=0.1):
    '''
        calculate the Sinkhorn distance between two sets of quantum states
        the cost matrix is the superfidelity between sets S1, S2
    '''
    superfidelity = compute_superfidelity(Set1, Set2)
    D = 1. - superfidelity
    emt = torch.empty(0)
    Sinkhorn_dis = ot.sinkhorn2(emt, emt, M=D, reg=reg)
    return Sinkhorn_dis

from scipy.stats import rv_continuous
class sin_prob_dist(rv_continuous):
    def _pdf(self, theta):
        # The 0.5 is so that the distribution is normalized
        return 0.5 * np.sin(theta)