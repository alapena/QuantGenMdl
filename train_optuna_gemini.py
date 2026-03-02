import torch
import numpy as np
import optuna
import yaml
from typing import Dict, List, Tuple
from tqdm import tqdm
from functools import partial

# Constants/Utils (Assumed)
from src.utils import get_path, find_closest_power_of_2, set_device
from src.QDDPM_torch_angel import QDDPM, WassDistance, sinkhornDistance

class QuantumDiffusionArchitect:
    def __init__(self, config: dict, n_qubits: int, n_ancilla: int, n_timesteps: int, device: str):
        self.config = config
        self.device = device
        self.n_qubits = n_qubits
        self.n_ancilla = n_ancilla
        self.n_tot = n_qubits + n_ancilla
        self.n_timesteps = n_timesteps
        
        self.best_layers: Dict[int, int] = {}
        self.best_params: Dict[int, torch.Tensor] = {}

    def _get_loss_fn(self):
        loss_type = self.config['training'].get('loss_fn', 'wass')
        if loss_type == 'sinkhorn':
            return partial(sinkhornDistance, reg=self.config['training']['regularization'])
        return WassDistance

    def timestep_objective(self, trial, t: int, input_state: torch.Tensor, model: QDDPM) -> float:
        """Optuna objective for a single timestep."""
        L_t = trial.suggest_int(f"L_t{t}", 1, 12) 
        
        # Initialize parameters for this specific architecture depth
        params_t = torch.randn(2 * self.n_tot * L_t, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([params_t], lr=self.config['training']['learning_rate'])
        lossfn = self._get_loss_fn()
        
        trial_epochs = self.config['training'].get('optuna_epochs', 50)
        
        for _ in range(trial_epochs):
            indices = np.random.choice(model.states_diff.shape[1], 
                                     size=self.config['dataset']['maxsize'], replace=False)
            target = model.states_diff[t, indices]
            
            # Use the dynamic method we discussed
            output = model.backwardOutput_dynamic(input_state, params_t, L_t)
            loss = lossfn(output, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        return loss.item()

    def run_sequential_optimization(self, model: QDDPM):
        # 1. Initialize input at T (Max noise)
        # MUST BE CFLOAT for quantum states
        init_data = model.states_diff[-1]
        current_input = torch.zeros((init_data.shape[0], 2**self.n_tot), 
                                  device=self.device, dtype=torch.cfloat)
        current_input[:, :init_data.shape[1]] = init_data

        # 2. Iterate backwards from T-1 down to 0
        for t in range(self.n_timesteps - 1, -1, -1):
            print(f"\n--- Optimizing Structure for Timestep t={t} ---")
            
            study = optuna.create_study(direction="minimize")
            study.optimize(lambda trial: self.timestep_objective(trial, t, current_input, model), 
                           n_trials=11)
            
            best_L = study.best_params[f"L_t{t}"]
            self.best_layers[t] = best_L
            
            # 3. Final training for this step
            print(f"Locked L={best_L}. Finalizing weights...")
            params_t, next_state = self.train_final_t(t, current_input, best_L, model)
            
            self.best_params[t] = params_t
            # Detach and pad for next timestep (t-1)
            current_input = torch.zeros((next_state.shape[0], 2**self.n_tot), 
                                      device=self.device, dtype=torch.cfloat)
            current_input[:, :next_state.shape[1]] = next_state.detach()

        print("\nOptimization Complete.")
        print("Layer Map:", self.best_layers)

    def train_final_t(self, t, input_state, L, model):
        params = torch.randn(2 * self.n_tot * L, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([params], lr=self.config['training']['learning_rate'])
        lossfn = self._get_loss_fn()

        for _ in tqdm(range(self.config['training']['n_epochs']), desc=f"Finalizing t={t}"):
            indices = np.random.choice(model.states_diff.shape[1], 
                                     size=self.config['dataset']['maxsize'], replace=False)
            target = model.states_diff[t, indices]
            output = model.backwardOutput_dynamic(input_state, params, L)
            loss = lossfn(output, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        with torch.no_grad():
            final_output = model.backwardOutput_dynamic(input_state, params, L)
        return params.detach(), final_output

def main():
    with open('config_debug.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    device = set_device(config.get('device', 'cpu'))
    
    n_pixels = config['dataset']['transforms']['resize']**2
    _, n_qubits = find_closest_power_of_2(n_pixels, return_power=True)
    n_ancilla = config['model']['n_ancilla_qubits']
    n_t = config['model']['n_timesteps']

    # Initialize Model once
    model = QDDPM(n_qubits, n_ancilla, n_t, L=1, device=device).to(device)
    # Load data into model (Assumes set_diffusionSet exists)
    # states_diffused = np.load(...) 
    # model.set_diffusionSet(states_diffused)

    architect = QuantumDiffusionArchitect(
        config=config,
        n_qubits=n_qubits,
        n_ancilla=n_ancilla,
        n_timesteps=n_t,
        device=device
    )
    
    architect.run_sequential_optimization(model)


if __name__ == "__main__":
    main()