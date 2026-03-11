from src.utils import get_path, find_closest_power_of_2, set_device
from src.QDDPM_torch_angel import DiffusionModel, QDDPM, WassDistance, sinkhornDistance
from src.plot import Plotter
from tqdm import tqdm
from functools import partial
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
    n_features = config['dataset']['transforms']['resize']**2 # EDITABLE
    n_timesteps = config['model']['n_timesteps'] # EDITABLE

    _, n_qubits = find_closest_power_of_2(n_features, return_power=True)
    n_backward_layers = config['model']['n_backward_layers']
    n_ancilla_qubits = config['model']['n_ancilla_qubits']

    # Initialize model
    model = QDDPM(n_qubits, n_ancilla_qubits, n_timesteps, n_backward_layers, device=device).to(device)

    trainer = Trainer(model, config, n_data, n_features, n_timesteps, n_qubits, n_ancilla_qubits, n_backward_layers)
    trainer.train()




######################################################################
#                                                                    #
#                          HELPER FUNCTIONS                          #
#                                                                    #
######################################################################

class Trainer():
    def __init__(self, model: QDDPM, config, n_data, n_features, n_timesteps, n_qubits, n_ancilla_qubits, n_backward_layers):
        self.model = model
        self.config = config
        self.device = model.device

        self.n_data = n_data
        self.n_features = n_features
        self.n_timesteps = n_timesteps

        self.n_qubits = n_qubits
        self.n_ancilla_qubits = n_ancilla_qubits
        self.n_backward_layers = n_backward_layers
        self.n_epochs = self.config['training']['n_epochs']
        self.reg = config['training']['regularization']

        self.n_params = 2 * self.model.n_tot * self.model.L
        self.loss_fn = self.config['training'].get('loss_fn', 'wass')

        self.plotter = Plotter()

    def _get_loss_fn(self):
        loss_type = self.config['training'].get('loss_fn', 'wass')
        if loss_type == 'sinkhorn':
            return partial(sinkhornDistance, reg=self.config['training']['regularization'])
        elif loss_type == 'wass'
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
        if not self.config["training"]["overwrite_saves"]:
            return False
        dir, filename = get_path(self.config, type='bestparams.npy', n_data=self.n_data, n_features=self.n_features, n_qubits=self.n_qubits, n_ancilla_qubits=self.n_ancilla_qubits, n_timesteps=self.n_timesteps, n_backward_layers=self.n_backward_layers, t=tt)
        path = dir/filename
        return path.exists()

    def _save_config(self):
        dir, filename = get_path(self.config, type='config.yaml', n_data=self.n_data, n_features=self.n_features, n_qubits=self.n_qubits, n_ancilla_qubits=self.n_ancilla_qubits, n_timesteps=self.n_timesteps, n_backward_layers=self.n_backward_layers)
        with open(dir/filename, 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False, sort_keys=False)


    def train(self):
        self._save_config()

        learning_rate = self.config['training']['learning_rate']
        diffusion_schedule = self.config['model'].get('diffusion_schedule', None)

        # Load diffused states
        dir, filename = get_path(self.config, type='diffusedqstates.npy', diffusion_schedule=diffusion_schedule, n_data=self.n_data, n_features=self.n_features, n_qubits=self.n_qubits, n_timesteps=self.n_timesteps)
        states_diffused = np.load(dir / filename) # Must be numpy array

        self.model.set_diffusionSet(states_diffused) # This already converts the states to torch tensors in the device
        inputs_last_timestep = torch.from_numpy(states_diffused[-1]).to(self.device)

        self.model.train()
        for t in range(self.n_timesteps, 0, -1): # From T to 1
            print(f"--- Training timestep {t} ---")
            if self._check_pretrained_params(): 
                print("Found already trained parameters. Skipping this timestep...")
                continue

            params_tot = torch.zeros((self.n_timesteps, 2*(self.n_qubits+self.n_ancilla_qubits)*self.n_backward_layers), device=self.device)
            if t < self.n_timesteps:
                for tt in range(t+1, self.n_timesteps+1):
                    dir, filename = get_path(self.config, type='bestparams.npy', n_data=self.n_data, n_features=self.n_features, n_qubits=self.n_qubits, n_ancilla_qubits=self.n_ancilla_qubits, n_timesteps=self.n_timesteps, n_backward_layers=self.n_backward_layers, t=tt)
                    params_tot[tt-1] = torch.from_numpy(np.load(dir / filename)).to(self.device)
            params, loss_hist = self.train_timestep_t(t, inputs_last_timestep, params_tot, self.n_data, learning_rate)

            self.save_results_t(params.detach().cpu(), loss_hist, t, last_epoch=True)

    
    def train_timestep_t(self, t, inputs_last_timestep, params_tot, n_data, lr):
        self.history = {
            'lr': [],
            'loss': [],
        }
        
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
                self.save_results_t(params_t.detach().cpu(), loss_hist, t, verbose=False)
                last_save = epoch

            # Save and plot stats
            self.history['lr'].append(lr_scheduler.get_last_lr()[0])
            self.history['loss'].append(loss_value)
            loss_hist.append(loss_value) # record the current loss
            
            if self.config['training']['live_plot'] and epoch%50 == 0:
                fig = self.plotter.plot_loss(t, history=self.history)
                dir, filename = get_path(self.config, type='lossplot.html', n_data=self.n_data, n_features=self.n_features, n_qubits=self.n_qubits, n_ancilla_qubits=self.n_ancilla_qubits, n_timesteps=self.n_timesteps, n_backward_layers=self.n_backward_layers, t=t)
                fig.write_html(str(dir/filename))

        return params_t, torch.stack(loss_hist)
    
    def save_results_t(self, params, loss_hist, t, verbose=True):
        dir, filename = get_path(self.config, type='bestparams.npy', n_data=self.n_data, n_features=self.n_features, n_qubits=self.n_qubits, n_ancilla_qubits=self.n_ancilla_qubits, n_timesteps=self.n_timesteps, n_backward_layers=self.n_backward_layers, t=t)
        np.save(dir/filename, params)
        if verbose:
            print(f"Saved parameters at {dir/filename}.")

        dir, filename = get_path(self.config, type='bestlosshist.npy', n_data=self.n_data, n_features=self.n_features, n_qubits=self.n_qubits, n_ancilla_qubits=self.n_ancilla_qubits, n_timesteps=self.n_timesteps, n_backward_layers=self.n_backward_layers, t=t)
        np.save(dir/filename, loss_hist)


    def save_results_lastepoch(self, params, loss_hist, t, verbose=True):
        dir, filename = get_path(self.config, type='finalparams.npy', n_data=self.n_data, n_features=self.n_features, n_qubits=self.n_qubits, n_ancilla_qubits=self.n_ancilla_qubits, n_timesteps=self.n_timesteps, n_backward_layers=self.n_backward_layers, t=t)
        np.save(dir / filename, params)
        if verbose:
            print(f"Saved parameters at {dir/filename}.")

        dir, filename = get_path(self.config, type='finallosshist.npy', n_data=self.n_data, n_features=self.n_features, n_qubits=self.n_qubits, n_ancilla_qubits=self.n_ancilla_qubits, n_timesteps=self.n_timesteps, n_backward_layers=self.n_backward_layers, t=t)
        np.save(dir / filename, loss_hist)

    def save_grads(self, epoch, grads):
        pass



if __name__ == "__main__":
    main()
