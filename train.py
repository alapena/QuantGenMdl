from src.utils import get_path, find_closest_power_of_2, set_device
from src.QDDPM_torch_angel import DiffusionModel, QDDPM, WassDistance, sinkhornDistance
from plotly.subplots import make_subplots
from tqdm import tqdm
import numpy as np
import plotly.graph_objects as go
import torch
import yaml
import time

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

    # Initialize model
    model = QDDPM(n_qubits, n_ancilla_qubits, n_timesteps, n_backward_layers, device=device).to(device)

    trainer = Trainer(model, config, n_data, n_pixels, n_timesteps, n_qubits, n_ancilla_qubits, n_backward_layers)
    trainer.train()




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

        self.n_params = 2 * self.model.n_tot * self.model.L

    def train(self):
        dir, filename = get_path(self.config, type='config', n_data=self.n_data, n_pixels=self.n_pixels, n_qubits=self.n_qubits, n_ancilla_qubits=self.n_ancilla_qubits, n_timesteps=self.n_timesteps, n_backward_layers=self.n_backward_layers)
        with open(dir/filename, 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False, sort_keys=False)
        
        n_data = self.n_data
        n_pixels = self.n_pixels
        n_timesteps = self.n_timesteps

        n_qubits = self.n_qubits
        n_backward_layers = self.n_backward_layers
        n_ancilla_qubits = self.n_ancilla_qubits
        learning_rate = self.config['training']['learning_rate']
        diffusion_schedule = self.config['model'].get('diffusion_schedule', None)
        

        # Load diffused states
        dir, filename = get_path(self.config, type='diffusedqstates', diffusion_schedule=diffusion_schedule, n_data=n_data, n_pixels=n_pixels, n_qubits=n_qubits, n_timesteps=n_timesteps)
        states_diffused = np.load(dir / filename) # Must be numpy array

        self.model.set_diffusionSet(states_diffused) # This already converts the states to torch tensors in the device
        inputs_last_timestep = torch.from_numpy(states_diffused[-1]).to(self.device)

        for t in range(n_timesteps, 0, -1): # From T to 1
            print(f"--- Training timestep {t} ---")
            params_tot = torch.zeros((n_timesteps, 2*(n_qubits+n_ancilla_qubits)*n_backward_layers), device=self.device)
            if t < n_timesteps:
                for tt in range(t+1, n_timesteps+1):
                    dir, filename = get_path(self.config, type='modelparams', n_data=n_data, n_pixels=n_pixels, n_qubits=n_qubits, n_ancilla_qubits=self.n_ancilla_qubits, n_timesteps=n_timesteps, n_backward_layers=self.n_backward_layers, t=tt)
                    params_tot[tt-1] = torch.from_numpy(np.load(dir / filename)).to(self.device)
            params, loss_hist = self.train_timestep_t(t, inputs_last_timestep, params_tot, n_data, learning_rate)

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

        t0 = time.time()
        last_save = 0 # Epoch where results were last saved
        pbar = tqdm(range(self.n_epochs))
        for epoch in pbar:
            indices = np.random.choice(states_diffused.shape[1], size=n_data, replace=False)
            true_data = states_diffused[t, indices]

            output_t = self.model.backwardOutput_t(input_tplus1, params_t)
            loss = WassDistance(output_t, true_data)
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
