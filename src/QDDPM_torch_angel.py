import numpy as np
import ot
import tensorcircuit as tc
import scipy as sp
from scipy.stats import unitary_group
import torch
import torch.nn as nn
from torch.linalg import matrix_power
from opt_einsum import contract
from functools import partial
from itertools import combinations
from tqdm import tqdm

K = tc.set_backend('pytorch')
tc.set_dtype('complex64')

class QDDPMDiffusionModel(nn.Module):
    def __init__(self, n, T, Ndata, device='cpu'):
        '''
        the diffusion quantum circuit model to scramble arbitrary set of states to Haar random states
        Args:
        n: number of qubits
        T: number of diffusion steps
        Ndata: number of samples in the dataset
        '''
        super().__init__()
        self.n = n
        self.T = T
        self.Ndata = Ndata
        self.device=device

        self.scrambleCircuit_t_vmap = K.vmap(self.scrambleCircuit_t, vectorized_argnums=(1,2,3))
        self.scrambleCircuit_t_from_tminus1_vmap = K.vmap(self.scrambleCircuit_t_from_tminus1, vectorized_argnums=(1,2,3))
    
    def HaarSampleGeneration(self, Ndata, seed):
        '''
        generate random haar states,
        used as inputs in the t=T step for backward denoise
        Args:
        Ndata: number of samples in dataset
        '''
        np.random.seed(seed)
        U = unitary_group.rvs(dim=2**self.n, size=Ndata)

        if Ndata == 1:
            U = U[np.newaxis, :, :]  # force batch axis

        states_T = U[:, :, 0]  # first column
        return torch.from_numpy(states_T).cfloat()
    
    def scrambleCircuit_t(self, t, input, phis, gs=None):
        '''
        obtain the state through diffusion step t
        Args:
        t: diffusion step
        input: the input quantum state
        phis: the single-qubit rotation angles in diffusion circuit
        gs: the angle of RZZ gates in diffusion circuit when n>=2
        '''
        c = tc.Circuit(self.n, inputs=input)
        for tt in range(t):
            # single qubit rotations
            for i in range(self.n):
                c.rz(i, theta=phis[3*self.n*tt+i])
                c.ry(i, theta=phis[3*self.n*tt+self.n+i])
                c.rz(i, theta=phis[3*self.n*tt+2*self.n+i])
            # homogenous RZZ on every pair of qubits (n>=2)
            if self.n >= 2:
                for i, j in combinations(range(self.n), 2):
                    c.rzz(i, j, theta=gs[tt]/(2*np.sqrt(self.n)))
        return c.state()

    def scrambleCircuit_t_from_tminus1(self, t, input, phis, gs=None):
        '''
        obtain the state through diffusion step t
        Args:
        t: diffusion step
        input: the input quantum state, which is the output of step t-1
        phis: the single-qubit rotation angles in diffusion circuit
        gs: the angle of RZZ gates in diffusion circuit when n>=2
        '''
        c = tc.Circuit(self.n, inputs=input)
        tt = t-1 # To match the indexing of phis and gs in scrambleCircuit_t
        for i in range(self.n):
            c.rz(i, theta=phis[3*self.n*tt+i])
            c.ry(i, theta=phis[3*self.n*tt+self.n+i])
            c.rz(i, theta=phis[3*self.n*tt+2*self.n+i])
        if self.n >= 2:
            for i, j in combinations(range(self.n), 2):
                c.rzz(i, j, theta=gs[tt]/(2*np.sqrt(self.n)))
        return c.state()
    
    def set_diffusionData_t(self, t, inputs, diff_hs, seed):
        '''
        obtain the quantum data set through diffusion step t
        Args:
        t: diffusion step
        inputs: the input quantum data set
        diff_hs: the hyper-parameter to control the amplitude of quantum circuit angles
        '''
        # set single-qubit rotation angles
        torch.manual_seed(seed)
        phis = torch.rand(self.Ndata, 3*self.n*t, device=self.device)*np.pi/4. - np.pi/8.
        phis = phis*(diff_hs.repeat(3*self.n))
        if self.n > 1:
            # set homogenous RZZ gate angles
            gs = torch.rand(self.Ndata, t, device=self.device)*0.2 + 0.4
            gs *= diff_hs
        states = torch.zeros((self.Ndata, 2**self.n), device=self.device).cfloat()
        # for i in range(self.Ndata):
        #     if self.n > 1:
        #         states[i] = self.scrambleCircuit_t(t, inputs[i], phis[i], gs[i])
        #     else:
        #         states[i] = self.scrambleCircuit_t(t, inputs[i], phis[i])

        # Útil pero poco eficiente:
        # if self.n > 1:
        #     states = self.scrambleCircuit_t_vmap(t, inputs, phis, gs)
        # else:
        #     states = self.scrambleCircuit_t_vmap(t, inputs, phis)

        for tt in range(1, t+1):
            if self.n > 1:
                states = self.scrambleCircuit_t_from_tminus1_vmap(tt, inputs, phis, gs)
            else:
                states = self.scrambleCircuit_t_from_tminus1_vmap(tt, inputs, phis) # No RZZ gates

        return states

    def set_diffusionData_t_single_step(self, t, input_state, diff_hs, seed):
        """
        Apply exactly ONE step of diffusion (step t).
        """
        torch.manual_seed(seed)
        # We only need angles for the current step 't'
        # Note: Adjust indexing to ensure you pick the correct slice of diff_hs
        phis = torch.rand(self.Ndata, 3*self.n, device=self.device)*np.pi/4. - np.pi/8.
        phis = phis * diff_hs[-1] # Use the latest weight
        
        if self.n > 1:
            gs = torch.rand(self.Ndata, 1, device=self.device)*0.2 + 0.4
            gs *= diff_hs[-1]

        # Use the 'from_tminus1' logic directly for exactly one step
        if self.n > 1:
            # Note: We pass 1 here because we are only doing one step of circuit application
            return self.scrambleCircuit_t_from_tminus1_vmap(1, input_state, phis, gs)
        else:
            return self.scrambleCircuit_t_from_tminus1_vmap(1, input_state, phis)
        
    
from src.utils import get_path, get_diffusion_weights, get_diffusion_schedule_nickname, find_closest_power_of_2
from src.trainers.basic_trainers import QDDPMGeneratorInitialqstates

class QDDPMDiffuser(QDDPMDiffusionModel):
    def __init__(self, config, n_data, n_timesteps, n_features, device='cpu'):
        self.config = config
        self.device = device
        self.n_data = n_data
        self.n_timesteps = n_timesteps
        self.n_features = n_features
        self.diffusion_schedule_nickname = get_diffusion_schedule_nickname(config)
        _, self.n_qubits = find_closest_power_of_2(n_features, return_power=True)
        super().__init__(self.n_qubits, self.n_timesteps, self.n_data, device=self.device)

        self.get_path_partial = partial(get_path, config, modeltype='MLP', diffusion_schedule_nickname=self.diffusion_schedule_nickname, n_data=n_data, n_features=n_features, n_qubits=self.n_qubits, n_timesteps=n_timesteps)


    def diffuse(self):
        n_data, n_timesteps, n_features, diffusion_schedule_nickname = self.n_data, self.n_timesteps, self.n_features, self.diffusion_schedule_nickname
        _, n_qubits = find_closest_power_of_2(n_features, return_power=True)

        # Check if initial states exist
        dir, filename = get_path(self.config, type='initialqstates.npy', n_data=n_data, n_features=n_features, n_qubits=n_qubits)
        path = dir/filename
        if not path.exists() or True:
            # Generate initial states
            print("Initial quantum states not found. Generating them...")
            generator_initialqstates = QDDPMGeneratorInitialqstates(self.config)
            generator_initialqstates.generate_initialqstates()

        # Everything checked. Diffuse.
        diffuser = QDDPMDiffuser(config=self.config, n_data=n_data, n_timesteps=n_timesteps, n_features=n_features, device=self.device)
        diffusion_weights = get_diffusion_weights(self.config, device=self.device)
        states_diffused = torch.zeros((n_timesteps+1, n_data, n_features), dtype=torch.complex64, device=self.device)
        dir, filename = get_path(self.config, type='initialqstates.npy', n_data=n_data, n_features=n_features, n_qubits=n_qubits)
        initialqstates = torch.from_numpy(np.load(dir/filename))
        states_diffused[0] = initialqstates
        for t in tqdm(range(1, n_timesteps+1)):
                    # states[t] = model.set_diffusionData_t(t, states[0], diffusion_weights[:t], seed=t)
                    states_diffused[t] = diffuser.set_diffusionData_t_single_step(t, states_diffused[t-1], diffusion_weights[:t], seed=t)
                    states_diffused[t] = states_diffused[t] / torch.norm(states_diffused[t], dim=1, keepdim=True) # Avoid numerical errorsdir, filename = get_path(self.config, type='diffusedqstates.npy', diffusion_schedule=diffusion_schedule_nickname, n_data=n_data, n_features=n_features, n_qubits=n_qubits, n_timesteps=n_timesteps)
        dir, filename = get_path(self.config, type='diffusedqstates.npy', diffusion_schedule_nickname=diffusion_schedule_nickname, n_data=n_data, n_features=n_features, n_qubits=n_qubits, n_timesteps=n_timesteps)
        np.save(dir/filename, states_diffused.cpu().numpy())
        return states_diffused.cpu()

    def compute_wassdist_forward(self, config=None):
        config = self.config if config is None else config
        dir, filename = self.get_path_partial(type='diffusedqstates.npy')
        diffused_states = torch.from_numpy(np.load(dir / filename))
        wass = np.zeros(self.n_timesteps+1)
        for t in tqdm(range(self.n_timesteps+1)):
            np.random.seed()
            wass[t] = WassDistance(diffused_states[0], diffused_states[t]).detach().numpy()
        dir, filename = self.get_path_partial(type='wassdistforward.npy')
        np.save(dir / filename, wass)
        return wass



def backCircuit(input, params, n_tot, L):
    '''
    the backward denoise parameteric quantum circuits,
    designed following the hardware-efficient ansatz
    output is the state before measurmeents on ancillas
    Args:
    input: input quantum state of n_tot qubits
    params: the parameters of the circuit
    n_tot: number of qubits in the circuits
    L: layers of circuit
    '''
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


class QDDPM(nn.Module):
    def __init__(self, n, na, T, L, device='cpu'):
        '''
        the QDDPM model: backward process only work on cpu
        Args:
        n: number of data qubits
        na: number of ancilla qubits
        T: number of diffusion steps
        L: layers of circuit in each backward step
        '''
        super().__init__()
        self.n = n
        self.na = na
        self.n_tot = n + na
        self.T = T
        self.L = L
        self.device = device
        # embed the circuit to a vectorized pytorch neural network layer
        self.backCircuit_vmap = K.vmap(partial(backCircuit, n_tot=self.n_tot, L=L), vectorized_argnums=0)
        self.backCircuit_L_vmap = K.vmap(partial(backCircuit, n_tot=self.n_tot), vectorized_argnums=0)

    def set_diffusionSet(self, states_diff):
        self.states_diff = torch.from_numpy(states_diff).to(self.device).cfloat()

    def randomMeasure(self, inputs):
        '''
        Given the inputs on both data & ancilla qubits before measurmenets,
        calculate the post-measurement state.
        The measurement and state output are calculated in parallel for data samples
        Args:
        inputs: states to be measured, first na qubit is ancilla
        '''
        m_probs = (torch.abs(inputs.reshape(inputs.shape[0], 2**self.na, 2**self.n))**2).sum(dim=2)
        m_res = torch.multinomial(m_probs, num_samples=1).squeeze() # measurment results
        indices = 2**self.n * m_res.view(-1, 1) + torch.arange(2**self.n, device=self.device)
        post_state = torch.gather(inputs, 1, indices)
        norms = torch.sqrt(torch.sum(torch.abs(post_state)**2, axis=1)).unsqueeze(dim=1)
        return 1./norms * post_state

    def backwardOutput_dynamicL_t(self, inputs, params, L):
        '''
        Backward denoise process at step t
        Args:
        inputs: the input data set at step t
        '''
        # outputs through quantum circuits before measurement
        output_full = self.backCircuit_L_vmap(inputs, params, L=L) 
        # perform measurement
        output_t = self.randomMeasure(output_full)
        return output_t

    def backwardOutput_t(self, inputs, params):
        '''
        Backward denoise process at step t
        Args:
        inputs: the input data set at step t
        '''
        # outputs through quantum circuits before measurement
        output_full = self.backCircuit_vmap(inputs, params) 
        # perform measurement
        output_t = self.randomMeasure(output_full)
        return output_t
    
    def prepareInput_t(self, inputs_T, params_tot, t, Ndata):
        '''
        prepare the input samples for step t
        Args:
        inputs_T: the input state at the beginning of backward
        params_tot: all circuit parameters till step t+1
        '''
        self.input_tplus1 = torch.zeros((Ndata, 2**self.n_tot), device=self.device).cfloat()
        self.input_tplus1[:,:2**self.n] = inputs_T
        with torch.no_grad():
            for tt in range(self.T-1, t, -1):
                self.input_tplus1[:,:2**self.n] = self.backwardOutput_t(self.input_tplus1, params_tot[tt])
        return self.input_tplus1

    def prepareInput_optuna_t(self, inputs_T, params_tplus1, t, Ndata):
        '''
        prepare the input samples for step t
        Args:
        inputs_T: the input state at the beginning of backward
        params_tplus1: circuit parameters of step t+1
        '''
        self.input_tplus1 = torch.zeros((Ndata, 2**self.n_tot), device=self.device).cfloat()
        self.input_tplus1[:,:2**self.n] = inputs_T
        with torch.no_grad():
            for tt in range(self.T-1, t, -1):
                self.input_tplus1[:,:2**self.n] = self.backwardOutput_t(self.input_tplus1, params_tplus1)
        return self.input_tplus1
    
    def backDataGeneration(self, inputs_T, params_tot, Ndata):
        '''
        generate the dataset in backward denoise process with training data set
        '''
        states = torch.zeros((self.T+1, Ndata, 2**self.n_tot)).cfloat()
        states[-1, :, :2**self.n] = inputs_T
        params_tot = torch.from_numpy(params_tot).float()
        with torch.no_grad():
            for tt in range(self.T-1, -1, -1):
                states[tt, :, :2**self.n] = self.backwardOutput_t(states[tt+1], params_tot[tt])
        return states


def naturalDistance(Set1, Set2):
    '''
        a natural measure on the distance between two sets of quantum states
        definition: 2*d - r1-r2
        d: mean of inter-distance between Set1 and Set2
        r1/r2: mean of intra-distance within Set1/Set2
    '''
    # a natural measure on the distance between two sets, according to trace distance
    r11 = 1. - torch.mean(torch.abs(contract('mi,ni->mn', Set1.conj(), Set1))**2)
    r22 = 1. - torch.mean(torch.abs(contract('mi,ni->mn', Set2.conj(), Set2))**2)
    r12 = 1. - torch.mean(torch.abs(contract('mi,ni->mn', Set1.conj(), Set2))**2)
    return 2*r12 - r11 - r22


def WassDistance(Set1, Set2):
    '''
        calculate the Wasserstein distance between two sets of quantum states
        the cost matrix is the inter trace distance between sets S1, S2
    '''
    D = 1. - torch.abs(Set1.conj() @ Set2.T)**2.
    emt = torch.empty(0)
    Wass_dis = ot.emd2(emt, emt, M=D)
    return Wass_dis


def sinkhornDistance(Set1, Set2, reg=0.005, log=False):
    '''
        calculate the Sinkhorn distance between two sets of quantum states
        the cost matrix is the inter trace distance between sets S1, S2
        reg: the regularization coefficient
        log: whether to use the log-solver
    '''
    D = 1. - torch.abs(Set1.conj() @  Set2.T)**2.
    emt = torch.empty(0, device=Set1.device)
    if log == True:
        sh_dis = ot.sinkhorn2(emt, emt, M=D, reg=reg, method='sinkhorn_log')
    else:
        sh_dis = ot.sinkhorn2(emt, emt, M=D, reg=reg)
    return sh_dis
