from src.utils import get_path, find_closest_power_of_2, set_device
from src.QDDPM_torch_angel import DiffusionModel, QDDPM, WassDistance, sinkhornDistance
from plotly.subplots import make_subplots
from tqdm import tqdm
from functools import partial
import numpy as np
import plotly.graph_objects as go
import torch
import yaml
import time

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

    # Entrena varios modelos variando el n ancilla qubits
    values = list(range(7, 17))
    for n_ancilla_qubits in values:
        print(f"---TRAINING WITH n_ancilla_qubits={n_ancilla_qubits}---")

        try:
            # Initialize model
            model = QDDPM(n_qubits, n_ancilla_qubits, n_timesteps, n_backward_layers, device=device).to(device)

            trainer = Trainer(model, config, n_data, n_pixels, n_timesteps, n_qubits, n_ancilla_qubits, n_backward_layers)
            trainer.train()

        except torch.cuda.OutOfMemoryError:
            # Clear cache to prevent the next trial from failing immediately
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print(f"Trial {n_ancilla_qubits} failed: OOM with na={n_ancilla_qubits}.")




######################################################################
#                                                                    #
#                          HELPER FUNCTIONS                          #
#                                                                    #
######################################################################

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

        self.history = {
            'lr': [],
            'loss': [],
        }

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

    def train(self):
        n_data = self.n_data
        n_pixels = self.n_pixels
        n_timesteps = self.n_timesteps

        n_qubits = self.n_qubits
        n_backward_layers = self.n_backward_layers
        n_ancilla_qubits = self.n_ancilla_qubits
        learning_rate = self.config['training']['learning_rate']
        diffusion_schedule = self.config['model'].get('diffusion_schedule', 'linear')
        

        # Load diffused states
        dir, filename = get_path(self.config, type='diffusedqstates', diffusion_schedule=diffusion_schedule, n_data=n_data, n_pixels=n_pixels, n_qubits=n_qubits, n_timesteps=n_timesteps)
        states_diffused = np.load(dir / filename) # Must be numpy array

        self.model.set_diffusionSet(states_diffused) # This already converts the states to torch tensors in the device

        params, loss_hist = self.train_timestep_t(TIMESTEP, n_data, learning_rate)

        self.save_results(params.detach().cpu(), loss_hist, TIMESTEP, last_epoch=True)

    
    def train_timestep_t(self, t, n_data, lr):
        with torch.no_grad():
            # input_tplus1 = self.model.prepareInput_t(inputs_last_timestep, params_tot, t, n_data)
            states_diffused = self.model.states_diff
            input_tplus1 = torch.zeros((n_data, 2**(self.n_qubits + self.n_ancilla_qubits)), device=self.device).cfloat()
            input_tplus1[:,:2**self.n_qubits] = states_diffused[t]

        # initialize parameters
        np.random.seed()
        params_t = torch.tensor(np.random.normal(size=2 * self.model.n_tot * self.model.L), device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([params_t], lr=lr)
        lr_scheduler =  self._get_lr_scheduler()(optimizer)
        loss_fn = self._get_loss_fn()
        loss_hist = []

        t0 = time.time()
        last_save = 0 # Epoch where results were last saved
        pbar = tqdm(range(self.n_epochs))
        for epoch in pbar:
            indices = np.random.choice(states_diffused.shape[1], size=n_data, replace=False)
            true_data = states_diffused[t, indices]

            output_t = self.model.backwardOutput_t(input_tplus1, params_t)
            loss = loss_fn(output_t, true_data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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
            loss_hist.append(loss_value)

            if (epoch+1)%50 == 0:
                self.plot_loss(t)
            
            lr_scheduler.step()
            

        return params_t, torch.stack(loss_hist)
    
    def save_results(self, params, loss_hist, t, prefix='', last_epoch=False, verbose=True):
            
        if not last_epoch:
            dir, filename = get_path(self.config, type='modelsingleparams', n_data=self.n_data, n_pixels=self.n_pixels, n_qubits=self.n_qubits, n_ancilla_qubits=self.n_ancilla_qubits, n_timesteps=self.n_timesteps, n_backward_layers=self.n_backward_layers, t=t)
            np.save(dir / (prefix+filename), params)

            if verbose:
                print(f"Saved parameters at {dir/(prefix+filename)}.")

            dir, filename = get_path(self.config, type='modelsinglelosshist', n_data=self.n_data, n_pixels=self.n_pixels, n_qubits=self.n_qubits, n_ancilla_qubits=self.n_ancilla_qubits, n_timesteps=self.n_timesteps, n_backward_layers=self.n_backward_layers, t=t)
            np.save(dir / (prefix+filename), loss_hist)

        else:
            dir, filename = get_path(self.config, type='modelsinglefinalparams', n_data=self.n_data, n_pixels=self.n_pixels, n_qubits=self.n_qubits, n_ancilla_qubits=self.n_ancilla_qubits, n_timesteps=self.n_timesteps, n_backward_layers=self.n_backward_layers, t=t)
            np.save(dir / (prefix+filename), params)

            if verbose:
                print(f"Saved parameters at {dir/(prefix+filename)}.")

            dir, filename = get_path(self.config, type='modelsinglefinallosshist', n_data=self.n_data, n_pixels=self.n_pixels, n_qubits=self.n_qubits, n_ancilla_qubits=self.n_ancilla_qubits, n_timesteps=self.n_timesteps, n_backward_layers=self.n_backward_layers, t=t)
            np.save(dir / (prefix+filename), loss_hist)

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
