from tqdm import tqdm

from src.utils import get_dataset, get_diffusion_weights, get_path, find_closest_power_of_2
from src.QDDPM_torch_angel import QDDPM, DiffusionModel, WassDistance, sinkhornDistance
from src.plot import Plotter
from functools import partial
import numpy as np
import torch
import yaml

class QDDPMBasicTrainer():
    def __init__(self, config, n_data, n_features, n_timesteps, device='cpu'):
        self.device = device
        self.config = config

        self.n_data = n_data
        self.n_features = n_features
        self.n_timesteps = n_timesteps

        _, self.n_qubits = find_closest_power_of_2(n_features, return_power=True)
        self.n_ancilla_qubits = self.config['model']['n_ancilla_qubits']
        self.n_backward_layers = self.config['model']['n_backward_layers']
        self.n_epochs = self.config['training']['n_epochs']

        self.model = QDDPM(self.n_qubits, self.n_ancilla_qubits, self.n_timesteps, self.n_backward_layers, device=self.device).to(self.device)

        self.n_params = 2 * self.model.n_tot * self.model.L
        self.plotter = Plotter()

    def _get_loss_fn(self):
        loss_type = self.config['training'].get('loss_fn', 'wass')
        if loss_type == 'sinkhorn':
            return partial(sinkhornDistance, reg=self.config['training']['regularization'])
        elif loss_type == 'wass':
            return WassDistance
        else:
            raise NotImplementedError('Loss function {loss_type} not implemented.')
    
    def _get_lr_scheduler(self):
        config_lr_scheduler = self.config['training'].get('lr_scheduler', 'None')
        if config_lr_scheduler['type'] == 'CosineAnnealingWarmRestarts':
            T_0 = config_lr_scheduler.get('T_0', 50)
            T_mult = config_lr_scheduler.get('T_mult', 2)
            return partial(torch.optim.lr_scheduler.CosineAnnealingWarmRestarts, T_0=T_0, T_mult=T_mult)
        elif config_lr_scheduler['type'] == 'ctt':
            return NotImplementedError(f"Learning rate scheduler {config_lr_scheduler['type']} not implemented.")
        else:
            raise NotImplementedError(f"Learning rate scheduler {config_lr_scheduler['type']} not implemented.")

    def _check_pretrained_params(self, t):
        '''
        Checks if there are existing parameters for the given timestep t, to avoid re-training them.
        '''
        if self.config["training"]["overwrite_saves"]:
            return False
        dir, filename = get_path(self.config, type='bestparams.npy', n_data=self.n_data, n_features=self.n_features, n_qubits=self.n_qubits, n_ancilla_qubits=self.n_ancilla_qubits, n_timesteps=self.n_timesteps, n_backward_layers=self.n_backward_layers, t=t)
        path = dir/filename
        return path.exists()

    def _save_config(self):
        dir, filename = get_path(self.config, type='config.yaml', n_data=self.n_data, n_features=self.n_features, n_qubits=self.n_qubits, n_ancilla_qubits=self.n_ancilla_qubits, n_timesteps=self.n_timesteps, n_backward_layers=self.n_backward_layers)
        with open(dir/filename, 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False, sort_keys=False)

    def _save_results_t(self, params, loss_hist, t, verbose=True):
        dir, filename = get_path(self.config, type='bestparams.npy', n_data=self.n_data, n_features=self.n_features, n_qubits=self.n_qubits, n_ancilla_qubits=self.n_ancilla_qubits, n_timesteps=self.n_timesteps, n_backward_layers=self.n_backward_layers, t=t)
        np.save(dir/filename, params)
        if verbose:
            print(f"Saved parameters at {dir/filename}.\nCorresponding loss: {loss_hist[-1]:.5f}.")

        dir, filename = get_path(self.config, type='bestlosshist.npy', n_data=self.n_data, n_features=self.n_features, n_qubits=self.n_qubits, n_ancilla_qubits=self.n_ancilla_qubits, n_timesteps=self.n_timesteps, n_backward_layers=self.n_backward_layers, t=t)
        np.save(dir/filename, loss_hist)

    def _save_results_lastepoch(self, params, loss_hist, t, verbose=True):
        dir, filename = get_path(self.config, type='finalparams.npy', n_data=self.n_data, n_features=self.n_features, n_qubits=self.n_qubits, n_ancilla_qubits=self.n_ancilla_qubits, n_timesteps=self.n_timesteps, n_backward_layers=self.n_backward_layers, t=t)
        np.save(dir / filename, params)
        if verbose:
            print(f"Saved parameters at {dir/filename}.\nCorresponding loss: {loss_hist[-1]:.5f}.")

        dir, filename = get_path(self.config, type='finallosshist.npy', n_data=self.n_data, n_features=self.n_features, n_qubits=self.n_qubits, n_ancilla_qubits=self.n_ancilla_qubits, n_timesteps=self.n_timesteps, n_backward_layers=self.n_backward_layers, t=t)
        np.save(dir / filename, loss_hist)



class QDDPMDiffuser():
    def __init__(self, config, n_data, n_features, n_timesteps, device='cpu'):
        self.config = config
        self.device = device

        self.n_data = n_data
        self.n_features = n_features
        self.n_timesteps = n_timesteps

        self.diffusion_schedule_name = self.config["model"]["diffusion_schedule"]["name"]
        self.diffusion_schedule_slope = self.config["model"]["diffusion_schedule"]["slope"]
        _, self.n_qubits = find_closest_power_of_2(n_features, return_power=True)
    
    def diffuse(self):
        dir, filename = get_path(self.config, type='initialqstates.npy', n_data=self.n_data, n_features=self.n_features, n_qubits=self.n_qubits) # Editable
        dataset = torch.from_numpy(np.load(dir / filename)).to(self.device)
        model = DiffusionModel(self.n_qubits, self.n_timesteps, self.n_data, device=self.device)
        diffusion_weights = get_diffusion_weights(self.config, self.device)

        states = torch.zeros((self.n_timesteps+1, self.n_data, 2**self.n_qubits), device=self.device, dtype=torch.complex64)
        states[0] = dataset
        for t in tqdm(range(1, self.n_timesteps+1)):
            states[t] = model.set_diffusionData_t(t, states[0], diffusion_weights[:t], seed=t)
            states[t] = states[t] / torch.norm(states[t], dim=1, keepdim=True) # Avoid numerical errors

        name = self.diffusion_schedule_name + self.diffusion_schedule_slope
        dir, filename = get_path(self.config, type='diffusedqstates.npy', diffusion_schedule=name, n_data=self.n_data, n_features=self.n_features, n_qubits=self.n_qubits, n_timesteps=self.n_timesteps)
        np.save(dir / filename, states.detach().cpu().numpy())

        print(f"Saved diffused quantum states in {dir / filename}")


class QDDPMGeneratorInitialqstates():
    def __init__(self, config):
        self.config = config

    def generate_initialqstates(self):
        dataset = get_dataset(self.config)

        # Fill the rest of the states with zeroes, to match the shape of our states with the dimensionality of our Hilbert space.
        n_data = dataset.shape[0]
        n_features = dataset.shape[1]
        _, n_qubits = find_closest_power_of_2(n_features, return_power=True)
        dims_to_pad = 2**n_qubits-dataset.shape[1]
        dataset = np.pad(dataset, ((0,0), (0,dims_to_pad, 0)), 'constant', constant_values=0) if dims_to_pad > 0 else dataset
        
        # Save the generated quantum states
        dir, filename = get_path(self.config, type='initialqstates.npy', n_data=n_data, n_features=n_features, n_qubits=n_qubits)
        np.save(dir / filename, dataset)

        print(f"Dataset saved in {dir / filename}")