from src.utils import get_path, find_closest_power_of_2, set_device
from src.QDDPM_torch_angel import DiffusionModel, QDDPM, WassDistance, sinkhornDistance
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
import optuna

TIMESTEP = 40

def main():
    config = yaml.safe_load(open('config.yaml', 'r'))
    print("")
    device = set_device(config.get('device', 'cpu'))
    torch.set_default_device(device)

    n_data = config['dataset']['maxsize'] # EDITABLE
    n_pixels = config['dataset']['transforms']['resize']**2 # EDITABLE
    n_timesteps = config['model']['n_timesteps'] # EDITABLE

    _, n_qubits = find_closest_power_of_2(n_pixels, return_power=True)
    n_backward_layers = config['model']['n_backward_layers']
    n_ancilla_qubits = config['model']['n_ancilla_qubits']

    architect = QuantumDiffusionArchitect(config, device=device)
    architect.run_sequential_optimization()

    # # Entrena varios modelos variando el n ancilla qubits
    # values = list(range(1, 15))
    # for n_backward_layers in values:
    #     print(f"---TRAINING WITH n_backward_layers={n_backward_layers}---")
    #     # Initialize model
    #     model = QDDPM(n_qubits, n_ancilla_qubits, n_timesteps, n_backward_layers, device=device).to(device)

    #     trainer = Trainer(model, config, n_data, n_pixels, n_timesteps, n_qubits, n_ancilla_qubits, n_backward_layers)
    #     trainer.train()




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
        self.n_timesteps = config['model']['n_timesteps']
        self.device = device
        self.dataset_name = self.config['dataset']['name']
        self.diffusion_schedule = self.config['model'].get('diffusion_schedule', None)
        self.n_epochs = self.config['training']['n_epochs']

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

        # Load diffused states
        dir, filename = get_path(self.config, type='diffusedqstates', diffusion_schedule=self.diffusion_schedule, n_data=self.n_data, n_pixels=self.n_pixels, n_qubits=self.n_qubits, n_timesteps=self.n_timesteps)
        states_diffused = np.load(dir / filename) # Must be numpy array

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.seed),
            pruner=optuna.pruners.HyperbandPruner(min_resource=5, max_resource=self.config['training']['n_epochs'], reduction_factor=3)
            )
        study.optimize(lambda trial: self.timestep_objective(trial, TIMESTEP, states_diffused), 
                        n_trials=11)
        
        # Print and save results
        best_L = study.best_params[f"L_t{TIMESTEP}"]
        best_na = study.best_params[f"na_t{TIMESTEP}"]
        best_loss = study.best_value
        
        self.best_layers[TIMESTEP] = {
            "L": best_L,
            "na": best_na,
            "loss": float(best_loss)
        }
        
        print(f"Results for timestep {TIMESTEP}: L={best_L}, na={best_na} (Loss: {best_loss:.6f})")
        print(f"Current Architecture Map: {self.best_layers}")

        # Save results. Overwrites the file after new timestep finishes.
        dir, filename = get_path(self.config, type='optunadict')
        with open(dir / filename, 'w') as f:
            yaml.dump(self.best_layers, f, default_flow_style=False)

        best_params_tensor = study.best_trial.user_attrs["trained_weights"]
        self.save_results(best_params_tensor.detach().cpu().numpy(), best_na, best_L)

        print("\nOptimization Complete.")
        print("Layer Map:", self.best_layers)

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
            with torch.no_grad():
                states_diffused = model.states_diff
                input_tplus1 = torch.zeros((self.n_data, 2**(self.n_qubits + na_t)), device=self.device).cfloat()
                input_tplus1[:,:2**self.n_qubits] = states_diffused[t]


            # initialize parameters
            np.random.seed()
            params_t = torch.tensor(np.random.normal(size=2 * n_tot * model.L), device=self.device, requires_grad=True)
            optimizer = torch.optim.Adam([params_t], lr=self.config['training']['learning_rate'])
            lr_scheduler = self._get_lr_scheduler()(optimizer)
            lossfn = self._get_loss_fn()

            best_loss = float('inf')
            best_params_local = None
            last_save = 0 # Epoch where results were last saved
            pbar = tqdm(range(self.n_epochs))
            for epoch in pbar:
                indices = np.random.choice(states_diffused.shape[1], size=self.n_data, replace=False)
                true_data = states_diffused[t, indices]

                output_t = model.backwardOutput_t(input_tplus1, params_t)
                loss = lossfn(output_t, true_data)
                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                loss_value = loss.detach().cpu()

                pbar.set_postfix({
                    'ℒ (loss)': f"{loss.item():.4f}",
                    '💾 (last saved)': f"{last_save}"
                })

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

    def save_results(self, params, best_na, best_L):
        dir, filename = get_path(self.config, type='modelparams', n_data=self.n_data, n_pixels=self.n_pixels, n_qubits=self.n_qubits, n_ancilla_qubits=best_na, n_timesteps=self.n_timesteps, n_backward_layers=best_L, t=TIMESTEP)
        np.save(dir / (filename), params)


# class Trainer():
#     def __init__(self, model: QDDPM, config, n_data, n_pixels, n_timesteps, n_qubits, n_ancilla_qubits, n_backward_layers):
#         self.model = model
#         self.config = config
#         self.device = model.device

#         self.n_data = n_data
#         self.n_pixels = n_pixels
#         self.n_timesteps = n_timesteps

#         self.n_qubits = n_qubits
#         self.n_ancilla_qubits = n_ancilla_qubits
#         self.n_backward_layers = n_backward_layers
#         self.n_epochs = self.config['training']['n_epochs']
#         self.reg = config['training']['regularization']

#         self.history = {
#             'lr': [],
#             'loss': [],
#         }

#     def train(self):
#         n_data = self.n_data
#         n_pixels = self.n_pixels
#         n_timesteps = self.n_timesteps

#         n_qubits = self.n_qubits
#         n_backward_layers = self.n_backward_layers
#         n_ancilla_qubits = self.n_ancilla_qubits
#         learning_rate = self.config['training']['learning_rate']
#         diffusion_schedule = self.config['model'].get('diffusion_schedule', 'linear')
        

#         # Load diffused states
#         dir, filename = get_path(self.config, type='diffusedqstates', diffusion_schedule=diffusion_schedule, n_data=n_data, n_pixels=n_pixels, n_qubits=n_qubits, n_timesteps=n_timesteps)
#         states_diffused = np.load(dir / filename) # Must be numpy array

#         self.model.set_diffusionSet(states_diffused) # This already converts the states to torch tensors in the device

#         params, loss_hist = self.train_timestep_t(TIMESTEP, n_data, learning_rate)

#         self.save_results(params.detach().cpu(), loss_hist, TIMESTEP, last_epoch=True)

    
#     def train_timestep_t(self, t, n_data, lr):
#         with torch.no_grad():
#             # input_tplus1 = self.model.prepareInput_t(inputs_last_timestep, params_tot, t, n_data)
#             states_diffused = self.model.states_diff
#             input_tplus1 = torch.zeros((n_data, 2**(self.n_qubits + self.n_ancilla_qubits)), device=self.device).cfloat()
#             input_tplus1[:,:2**self.n_qubits] = states_diffused[t+1]

#         # initialize parameters
#         np.random.seed()
#         params_t = torch.tensor(np.random.normal(size=2 * self.model.n_tot * self.model.L), device=self.device, requires_grad=True)
#         optimizer = torch.optim.Adam([params_t], lr=lr)
#         lr_scheduler =  torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
#         loss_hist = []

#         t0 = time.time()
#         last_save = 0 # Epoch where results were last saved
#         pbar = tqdm(range(self.n_epochs))
#         for epoch in pbar:
#             indices = np.random.choice(states_diffused.shape[1], size=n_data, replace=False)
#             true_data = states_diffused[t, indices]

#             output_t = self.model.backwardOutput_t(input_tplus1, params_t)
#             loss = WassDistance(output_t, true_data)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             loss_value = loss.detach().cpu()

#             pbar.set_postfix({
#                 'ℒ (loss)': f"{loss.item():.4f}",
#                 '💾 (last saved)': f"{last_save}"
#             })

#             # Check if current step is best
#             if len(loss_hist) == 0 or loss_value < min(loss_hist):
#                 self.save_results(params_t.detach().cpu(), loss_hist, t, verbose=False)
#                 last_save = epoch

#             # Save and plot stats
#             self.history['lr'].append(lr_scheduler.get_last_lr()[0])
#             self.history['loss'].append(loss_value)
#             loss_hist.append(loss_value)

#             if (epoch+1)%50 == 0:
#                 self.plot_loss(t)
            
#             lr_scheduler.step()
            

#         return params_t, torch.stack(loss_hist)
    
#     def save_results(self, params, loss_hist, t, prefix='', last_epoch=False, verbose=True):
            
#         if not last_epoch:
#             dir, filename = get_path(self.config, type='modelsingleparams', n_data=self.n_data, n_pixels=self.n_pixels, n_qubits=self.n_qubits, n_ancilla_qubits=self.n_ancilla_qubits, n_timesteps=self.n_timesteps, n_backward_layers=self.n_backward_layers, t=t)
#             np.save(dir / (prefix+filename), params)

#             if verbose:
#                 print(f"Saved parameters at {dir/(prefix+filename)}.")

#             dir, filename = get_path(self.config, type='modelsinglelosshist', n_data=self.n_data, n_pixels=self.n_pixels, n_qubits=self.n_qubits, n_ancilla_qubits=self.n_ancilla_qubits, n_timesteps=self.n_timesteps, n_backward_layers=self.n_backward_layers, t=t)
#             np.save(dir / (prefix+filename), loss_hist)

#         else:
#             dir, filename = get_path(self.config, type='modelsinglefinalparams', n_data=self.n_data, n_pixels=self.n_pixels, n_qubits=self.n_qubits, n_ancilla_qubits=self.n_ancilla_qubits, n_timesteps=self.n_timesteps, n_backward_layers=self.n_backward_layers, t=t)
#             np.save(dir / (prefix+filename), params)

#             if verbose:
#                 print(f"Saved parameters at {dir/(prefix+filename)}.")

#             dir, filename = get_path(self.config, type='modelsinglefinallosshist', n_data=self.n_data, n_pixels=self.n_pixels, n_qubits=self.n_qubits, n_ancilla_qubits=self.n_ancilla_qubits, n_timesteps=self.n_timesteps, n_backward_layers=self.n_backward_layers, t=t)
#             np.save(dir / (prefix+filename), loss_hist)

#     def plot_loss(self, t):
#         fig = make_subplots(specs=[[{"secondary_y": True}]])

#         fig.add_trace(
#             go.Scatter(
#                 y = self.history['loss'],
#                 name = 'Loss'
#             ),
#             secondary_y=False
#         )

#         fig.add_trace(
#             go.Scatter(
#                 y = self.history['lr'],
#                 name = 'Learning rate',
#                 line=dict(color="lightgreen", dash="solid"),
#             ),
#             secondary_y=True
#         )

#         fig.update_layout(
#             title = f'Loss plot of timestep {t}',
#             xaxis_title = 'Epoch',
#             yaxis = dict(
#                 title = 'Loss'
#             ),
#             yaxis2 = dict(
#                 title = 'Learning rate',
#                 showgrid = False,
#                 side = 'right',
#                 type="log",
#                 tickformat=".0e",
#             )
#         )

#         dir, filename = get_path(self.config, type='lossplot', n_data=self.n_data, n_pixels=self.n_pixels, n_qubits=self.n_qubits, n_ancilla_qubits=self.n_ancilla_qubits, n_timesteps=self.n_timesteps, n_backward_layers=self.n_backward_layers, t=t)
#         fig.write_html(str(dir/filename))



if __name__ == "__main__":
    main()
