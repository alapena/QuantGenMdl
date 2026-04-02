import optuna

from src.utils import get_path, find_closest_power_of_2, set_device
from src.QDDPM_torch_angel import QDDPMDiffusionModel, QDDPM, WassDistance, sinkhornDistance
from plotly.subplots import make_subplots
from tqdm import tqdm
from functools import partial
from typing import Dict
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
import torch
import yaml
import time

def main():
    config = yaml.safe_load(open('config_debug.yaml', 'r'))
    print("")
    print("Running Optuna optimization...")
    device = set_device(config.get('device', 'cpu'))
    torch.set_default_device(device)

    n_data = config['dataset']['maxsize'] # EDITABLE
    n_pixels = config['dataset']['transforms']['resize']**2 # EDITABLE
    n_timesteps = config['model']['n_timesteps'] # EDITABLE

    _, n_qubits = find_closest_power_of_2(n_pixels, return_power=True)
    n_backward_layers = config['model']['n_backward_layers']
    n_ancilla_qubits = config['model']['n_ancilla_qubits']

    # Initialize model

    architect = QuantumDiffusionArchitect(config, device=device)
    architect.run_sequential_optimization()

    # trainer = Trainer(model, config, n_data, n_pixels, n_timesteps, n_qubits, n_ancilla_qubits, n_backward_layers)
    # trainer.train()
    




######################################################################
#                                                                    #
#                          HELPER FUNCTIONS                          #
#                                                                    #
######################################################################

class QuantumDiffusionArchitect():
    def __init__(self, config, device='cpu'):
        self.config = config
        self.seed = self.config['training']['seed']
        self.n_data = config['dataset']['maxsize']
        self.n_pixels = config['dataset']['transforms']['resize']**2
        _, self.n_qubits = find_closest_power_of_2(self.n_pixels, return_power=True)
        self.n_ancilla_qubits = config['model']['n_ancilla_qubits']
        self.n_timesteps = config['model']['n_timesteps']
        self.device = device
        self.dataset_name = self.config['dataset']['name']

        self.best_layers: Dict[int, int] = {}

    def _get_loss_fn(self):
        loss_type = self.config['training'].get('loss_fn', 'wass')
        if loss_type == 'sinkhorn':
            return partial(sinkhornDistance, reg=self.config['training']['regularization'])
        return WassDistance
    
    def _get_lr_scheduler(self):
        config_lr_scheduler = self.config['training'].get('lr_scheduler', 'None')

        if config_lr_scheduler['type'] == 'CosineAnnealingWarmRestarts':
            T_0 = config_lr_scheduler.get('T_0', 50)
            T_mult = config_lr_scheduler.get('T_mult', 2)
            return partial(torch.optim.lr_scheduler.CosineAnnealingWarmRestarts, T_0=T_0, T_mult=T_mult)
        
        elif config_lr_scheduler['type'] == 'ctt':
            return None
        
        else:
            raise NotImplementedError(f"Learning rate scheduler {config_lr_scheduler['type']} not implemented.")

    def run_sequential_optimization(self):
        diffusion_schedule = self.config['model'].get('diffusion_schedule', None)

        # Load diffused states
        dir, filename = get_path(self.config, type='diffusedqstates', diffusion_schedule=diffusion_schedule, n_data=self.n_data, n_pixels=self.n_pixels, n_qubits=self.n_qubits, n_timesteps=self.n_timesteps)
        states_diffused = np.load(dir / filename) # Must be numpy array

        # params_tot = torch.zeros((self.n_timesteps, 2*(self.n_qubits+self.n_ancilla_qubits)*self.n_backward_layers), device=self.device)

        for t in range(self.n_timesteps, 0, -1): # From T to 1
            print(f"\n--- Optimizing Structure for Timestep t={t} ---")

            # Optimize the timestep
            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=self.seed),
                pruner=optuna.pruners.HyperbandPruner(min_resource=5, max_resource=self.config['training']['n_epochs'], reduction_factor=3)
                )
            study.optimize(lambda trial: self.timestep_objective(trial, t, states_diffused), 
                           n_trials=11)
            
            best_L = study.best_params[f"L_t{t}"]
            best_na = study.best_params[f"na_t{t}"]
            best_loss = study.best_value
            
            
            self.best_layers[t] = {
                "L": best_L,
                "na": best_na,
                "loss": float(best_loss)
            }
            
            print(f"Results for timestep {t}: L={best_L}, na={best_na} (Loss: {best_loss:.6f})")
            print(f"Current Architecture Map: {self.best_layers}")

            # Save results. Overwrites the file after new timestep finishes.
            dir = Path(f"data/{self.dataset_name}/results_optuna")
            dir.mkdir(parents=True, exist_ok=True)
            filename = f"results.yaml"
            with open(dir / filename, 'w') as f:
                yaml.dump(self.best_layers, f, default_flow_style=False)

            best_params_tensor = study.best_trial.user_attrs["trained_weights"]
            self.save_results(best_params_tensor.detach().cpu(), t, last_epoch=True)

        print("\nOptimization Complete.")
        print("Layer Map:", self.best_layers)


    def save_results(self, params, t, prefix='', last_epoch=False, verbose=True):
            
        if not last_epoch:
            dir, filename = get_path(self.config, type='modelparams', n_data=self.n_data, n_pixels=self.n_pixels, n_qubits=self.n_qubits, n_ancilla_qubits=self.n_ancilla_qubits, n_timesteps=self.n_timesteps, n_backward_layers=self.n_backward_layers, t=t)
            np.save(dir / (prefix+filename), params)

            if verbose:
                print(f"Saved parameters at {dir/(prefix+filename)}.")

        else:
            dir = Path(f"data/{self.dataset_name}/results_optuna")
            filename = f"optimalparams_t{t}.npy"
            dir.mkdir(parents=True, exist_ok=True)
            np.save(dir / (prefix+filename), params)

            if verbose:
                print(f"Saved parameters at {dir/(prefix+filename)}.")


    def timestep_objective(self, trial, t, states_diffused):
        try:
            L_t = trial.suggest_int(f"L_t{t}", 1, 12) 
            na_t = trial.suggest_int(f"na_t{t}", 1, 2*self.n_qubits) 

            n_tot = self.n_qubits + na_t

            model = QDDPM(self.n_qubits, na_t, self.n_timesteps, L_t, device=self.device).to(self.device)

            model.set_diffusionSet(states_diffused) # This already converts the states to torch tensors in the device
            inputs_last_timestep = torch.from_numpy(states_diffused[-1]).to(self.device)
            
            model.train()

            # Prepare input for the current timestep
            params_list = [[] for _ in range(self.n_timesteps)] # To store the parameters of each timestep, which will be used as input for the next timesteps. This is needed for the optuna version where we optimize each timestep sequentially and need to use the optimized parameters of the previous timesteps as input for the next timesteps.
            if t < self.n_timesteps:
                for tt in range(t+1, self.n_timesteps+1):
                    dir = Path(f"data/{self.dataset_name}/results_optuna")
                    filename = f"optimalparams_t{tt}.npy"
                    params_list[tt-1] = torch.from_numpy(np.load(dir / filename)).to(self.device)
                    print(inputs_last_timestep.shape, params_list[tt-1].shape)
                input_tplus1 = model.prepareInput_t(inputs_last_timestep, params_list, t, self.n_data)
            else:
                input_tplus1 = torch.zeros((self.n_data, 2**(self.n_qubits+na_t)), device=self.device, dtype=torch.cfloat)
                input_tplus1[:,:2**self.n_qubits] = inputs_last_timestep

            states_diffused = model.states_diff

            generator = torch.Generator(device=self.device)
            generator.manual_seed(self.seed)
            params_t = torch.randn(2 * n_tot * L_t, generator=generator, device=model.device, requires_grad=True)
            optimizer = torch.optim.Adam([params_t], lr=self.config['training']['learning_rate'])
            lr_scheduler = self._get_lr_scheduler()(optimizer)
            lossfn = self._get_loss_fn()

            best_loss = float('inf')
            best_params_local = None

            for epoch in tqdm(range(self.config['training']['n_epochs'])):
                indices = np.random.choice(states_diffused.shape[1], size=self.n_data, replace=False)
                true_data = states_diffused[t, indices]


                output_t = model.backwardOutput_dynamicL_t(input_tplus1, params_t, L_t)
                loss = lossfn(output_t, true_data)
                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                loss_value = loss.detach().cpu()

                # Track the best weights within THIS trial
                if loss_value < best_loss:
                    best_loss = loss_value
                    best_params_local = params_t.detach().clone()

                # Optuna's Pruners logic
                trial.report(loss_value, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            trial.set_user_attr("trained_weights", best_params_local)

            return loss_value

        except torch.cuda.OutOfMemoryError:
            # Clear cache to prevent the next trial from failing immediately
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print(f"Trial {trial.number} failed: OOM with na_t={na_t}, L_t={L_t}. Returning penalty.")
            return float('inf')


if __name__ == "__main__":
    main()
