import optuna

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

def main():
    config = yaml.safe_load(open('config.yaml', 'r'))
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
    model = QDDPM(n_qubits, n_ancilla_qubits, n_timesteps, n_backward_layers, device=device).to(device)

    architect = QuantumDiffusionArchitect(config, device=device)
    architect.run_sequential_optimization(model)

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

        self.best_layers: Dict[int, int] = {}
        pass

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

    def run_sequential_optimization(self, model: QDDPM):
        learning_rate = self.config['training']['learning_rate']
        diffusion_schedule = self.config['model'].get('diffusion_schedule', None)

        # Load diffused states
        dir, filename = get_path(self.config, type='diffusedqstates', diffusion_schedule=diffusion_schedule, n_data=self.n_data, n_pixels=self.n_pixels, n_qubits=self.n_qubits, n_timesteps=self.n_timesteps)
        states_diffused = np.load(dir / filename) # Must be numpy array

        model.set_diffusionSet(states_diffused) # This already converts the states to torch tensors in the device
        inputs_last_timestep = torch.from_numpy(states_diffused[-1]).to(self.device)

        # params_tot = torch.zeros((self.n_timesteps, 2*(self.n_qubits+self.n_ancilla_qubits)*self.n_backward_layers), device=self.device)
        params_list = [[] for _ in range(self.n_timesteps)] # To store the parameters of each timestep, which will be used as input for the next timesteps. This is needed for the optuna version where we optimize each timestep sequentially and need to use the optimized parameters of the previous timesteps as input for the next timesteps.

        model.train()
        for t in range(self.n_timesteps, 0, -1): # From T to 1
            print(f"\n--- Optimizing Structure for Timestep t={t} ---")
            
            # Prepare input for the current timestep
            if t < self.n_timesteps:
                for tt in range(t+1, self.n_timesteps+1):
                    dir = Path(f"data/MNIST0/results_optuna")
                    filename = f"optimalparams_t{tt}.npy"
                    params_list[tt-1] = torch.from_numpy(np.load(dir / filename)).to(self.device)
                input_tplus1 = model.prepareInput_t(inputs_last_timestep, params_list, t, self.n_data)
            else:
                input_tplus1 = torch.zeros((self.n_data, 2**(self.n_qubits+self.n_ancilla_qubits)), device=self.device, dtype=torch.cfloat)
                input_tplus1[:,:2**self.n_qubits] = inputs_last_timestep

            # Optimize the timestep
            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.CmaEsSampler(seed=self.seed),
                pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=0)
                )
            study.optimize(lambda trial: self.timestep_objective(trial, t, input_tplus1, model), 
                           n_trials=11)
            
            best_L = study.best_params[f"L_t{t}"]
            best_loss = study.best_value
            
            
            self.best_layers[t] = {
                "L": best_L,
                "loss": float(best_loss)
            }
            
            print(f"Best L for timestep {t}: {best_L} (Loss: {best_loss:.6f})")
            print(f"Current Architecture Map: {self.best_layers}")

            # Save results
            dir = Path(f"data/MNIST0/results_optuna")
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
            dir = Path(f"data/MNIST0/results_optuna")
            filename = f"optimalparams_t{t}.npy"
            dir.mkdir(parents=True, exist_ok=True)
            np.save(dir / (prefix+filename), params)

            if verbose:
                print(f"Saved parameters at {dir/(prefix+filename)}.")


    def timestep_objective(self, trial, t, input_tplus1, model: QDDPM):
        L_t = trial.suggest_int(f"L_t{t}", 1, 11) 

        states_diffused = model.states_diff

        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.seed)
        params_t = torch.randn(2 * model.n_tot * L_t, generator=generator, device=model.device, requires_grad=True)
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


class Trainer():
    def __init__(self, model: QDDPM, config, n_data, n_pixels, n_timesteps, n_qubits, n_ancilla_qubits, n_backward_layers):
        self.model = model
        self.config = config
        self.device = model.device

        self.n_data = n_data
        self.n_pixels = n_pixels
        self.n_timesteps = n_timesteps

        self.n_qubits = n_qubits
        self.n_ancilla_qubits = n_ancilla_qubits
        self.n_backward_layers = n_backward_layers
        self.n_epochs = self.config['training']['n_epochs']
        self.reg = config['training']['regularization']

        self.n_params = 2 * self.model.n_tot * self.model.L
        self.loss_fn = self.config['training'].get('loss_fn', 'wass')

    def train(self):
        dir, filename = get_path(self.config, type='config', n_data=self.n_data, n_pixels=self.n_pixels, n_qubits=self.n_qubits, n_ancilla_qubits=self.n_ancilla_qubits, n_timesteps=self.n_timesteps, n_backward_layers=self.n_backward_layers)
        with open(dir/filename, 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False, sort_keys=False)

        learning_rate = self.config['training']['learning_rate']
        diffusion_schedule = self.config['model'].get('diffusion_schedule', None)

        # Load diffused states
        dir, filename = get_path(self.config, type='diffusedqstates', diffusion_schedule=diffusion_schedule, n_data=self.n_data, n_pixels=self.n_pixels, n_qubits=self.n_qubits, n_timesteps=self.n_timesteps)
        states_diffused = np.load(dir / filename) # Must be numpy array

        self.model.set_diffusionSet(states_diffused) # This already converts the states to torch tensors in the device
        inputs_last_timestep = torch.from_numpy(states_diffused[-1]).to(self.device)

        self.model.train()
        for t in range(self.n_timesteps, 0, -1): # From T to 1
            print(f"--- Training timestep {t} ---")
            params_tot = torch.zeros((self.n_timesteps, 2*(self.n_qubits+self.n_ancilla_qubits)*self.n_backward_layers), device=self.device)
            if t < self.n_timesteps:
                for tt in range(t+1, self.n_timesteps+1):
                    dir, filename = get_path(self.config, type='modelparams', n_data=self.n_data, n_pixels=self.n_pixels, n_qubits=self.n_qubits, n_ancilla_qubits=self.n_ancilla_qubits, n_timesteps=self.n_timesteps, n_backward_layers=self.n_backward_layers, t=tt)
                    params_tot[tt-1] = torch.from_numpy(np.load(dir / filename)).to(self.device)
            params, loss_hist = self.train_timestep_t(t, inputs_last_timestep, params_tot, self.n_data, learning_rate)

            self.save_results(params.detach().cpu(), loss_hist, t, last_epoch=True)

    
    def train_timestep_t(self, t, inputs_last_timestep, params_tot, n_data, lr):
        self.history = {
            'lr': [],
            'loss': [],
        }

        # To save the gradients
        self.grad_history_np = np.zeros((self.n_epochs, self.n_params), dtype=np.float32)
        
        # Prepare input
        input_tplus1 = self.model.prepareInput_t(inputs_last_timestep, params_tot, t, n_data)
        states_diffused = self.model.states_diff

        # initialize parameters
        np.random.seed()
        params_t = torch.tensor(np.random.normal(size=2 * self.model.n_tot * self.model.L), device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([params_t], lr=lr)
        lr_scheduler =  torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
        loss_hist = []

        # Select loss function
        if self.loss_fn == 'sinkhorn':
            reg = self.config['training']['regularization']
            lossfn = partial(sinkhornDistance, reg=reg)
        else:
            lossfn = WassDistance

        # Start training loop
        t0 = time.time()
        last_save = 0 # Epoch where results were last saved
        pbar = tqdm(range(self.n_epochs))
        for epoch in pbar:
            indices = np.random.choice(states_diffused.shape[1], size=n_data, replace=False)
            true_data = states_diffused[t, indices]

            output_t = self.model.backwardOutput_t(input_tplus1, params_t)
            loss = lossfn(output_t, true_data)
            optimizer.zero_grad()
            loss.backward()

            if self.config['training']['save_grads']:
                self.save_grads(epoch, params_t.grad.detach().cpu().numpy())

            optimizer.step()
            lr_scheduler.step()

            loss_value = loss.detach().cpu()

            pbar.set_postfix({
                'ℒ (loss)': f"{loss.item():.4f}",
                '💾 (last saved)': f"{last_save}"
            })

            # Check if current step is best
            if len(loss_hist) == 0 or loss_value < min(loss_hist):
                self.save_results(params_t.detach().cpu(), loss_hist, t, verbose=False)
                last_save = epoch

            # Save and plot stats
            self.history['lr'].append(lr_scheduler.get_last_lr()[0])
            self.history['loss'].append(loss_value)
            loss_hist.append(loss_value) # record the current loss
            
            if self.config['training']['live_plot'] and epoch%50 == 0:
                self.plot_loss(t)

        return params_t, torch.stack(loss_hist)
    
    def save_results(self, params, loss_hist, t, prefix='', last_epoch=False, verbose=True):
            
        if not last_epoch:
            dir, filename = get_path(self.config, type='modelparams', n_data=self.n_data, n_pixels=self.n_pixels, n_qubits=self.n_qubits, n_ancilla_qubits=self.n_ancilla_qubits, n_timesteps=self.n_timesteps, n_backward_layers=self.n_backward_layers, t=t)
            np.save(dir / (prefix+filename), params)

            if verbose:
                print(f"Saved parameters at {dir/(prefix+filename)}.")

            dir, filename = get_path(self.config, type='modellosshist', n_data=self.n_data, n_pixels=self.n_pixels, n_qubits=self.n_qubits, n_ancilla_qubits=self.n_ancilla_qubits, n_timesteps=self.n_timesteps, n_backward_layers=self.n_backward_layers, t=t)
            np.save(dir / (prefix+filename), loss_hist)

        else:
            dir, filename = get_path(self.config, type='modelfinalparams', n_data=self.n_data, n_pixels=self.n_pixels, n_qubits=self.n_qubits, n_ancilla_qubits=self.n_ancilla_qubits, n_timesteps=self.n_timesteps, n_backward_layers=self.n_backward_layers, t=t)
            np.save(dir / (prefix+filename), params)

            if verbose:
                print(f"Saved parameters at {dir/(prefix+filename)}.")

            dir, filename = get_path(self.config, type='modelfinallosshist', n_data=self.n_data, n_pixels=self.n_pixels, n_qubits=self.n_qubits, n_ancilla_qubits=self.n_ancilla_qubits, n_timesteps=self.n_timesteps, n_backward_layers=self.n_backward_layers, t=t)
            np.save(dir / (prefix+filename), loss_hist)

    def save_grads(self, epoch, grads):
        pass

    def plot_loss(self, t):
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Scatter(
                y = self.history['loss'],
                name = 'Loss'
            ),
            secondary_y=False
        )

        fig.add_trace(
            go.Scatter(
                y = self.history['lr'],
                name = 'Learning rate',
                line=dict(color="lightgreen", dash="solid"),
            ),
            secondary_y=True
        )

        fig.update_layout(
            title = f'Loss plot of timestep {t}',
            xaxis_title = 'Epoch',
            yaxis = dict(
                title = 'Loss'
            ),
            yaxis2 = dict(
                title = 'Learning rate',
                showgrid = False,
                side = 'right',
                type="log",
                tickformat=".0e",
            )
        )

        dir, filename = get_path(self.config, type='lossplot', n_data=self.n_data, n_pixels=self.n_pixels, n_qubits=self.n_qubits, n_ancilla_qubits=self.n_ancilla_qubits, n_timesteps=self.n_timesteps, n_backward_layers=self.n_backward_layers, t=t)
        fig.write_html(str(dir/filename))




if __name__ == "__main__":
    main()
