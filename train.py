from src.utils import get_path, find_closest_power_of_2, set_device
from src.QDDPM_torch_angel import DiffusionModel, QDDPM, WassDistance, sinkhornDistance
from src.plot import Plotter
from src.trainer import QDDPMBasicTrainer
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
    n_features = 4 #config['dataset']['transforms']['resize']**2 # EDITABLE
    n_timesteps = config['model']['n_timesteps'] # EDITABLE

    trainer = QDDPMTrainer(config, n_data, n_features, n_timesteps, device=device)
    trainer.train_all_timesteps()
    # trainer.train_single_timestep(t=40)




######################################################################
#                                                                    #
#                          HELPER FUNCTIONS                          #
#                                                                    #
######################################################################

class QDDPMTrainer(QDDPMBasicTrainer):
    def __init__(self, config, n_data, n_features, n_timesteps, device='cpu'):
        super().__init__(config, n_data, n_features, n_timesteps, device=device)

        self.learning_rate = self.config['training']['learning_rate']
        self.diffusion_schedule = self.config['model']['diffusion_schedule']

    def _save_config_single_timestep(self, t):
        dir, filename = get_path(self.config, type='config.yaml', suffix=f'_t{t}', n_data=self.n_data, n_features=self.n_features, n_qubits=self.n_qubits, n_ancilla_qubits=self.n_ancilla_qubits, n_timesteps=self.n_timesteps, n_backward_layers=self.n_backward_layers)
        with open(dir/filename, 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False, sort_keys=False)

    def train_single_timestep(self, t):
        '''(Must exist the previous timesteps trained parameters)'''
        self._save_config_single_timestep(t)

        # Load diffused states
        dir, filename = get_path(self.config, type='diffusedqstates.npy', diffusion_schedule=self.diffusion_schedule, n_data=self.n_data, n_features=self.n_features, n_qubits=self.n_qubits, n_timesteps=self.n_timesteps)
        states_diffused = np.load(dir / filename) # Must be numpy array
        self.model.set_diffusionSet(states_diffused) # This already converts the states to torch tensors in the device
        inputs_last_timestep = torch.from_numpy(states_diffused[-1]).to(self.device)

        self.model.train()
        print(f"--- Training timestep {t} ---")
        params_tot = torch.zeros((self.n_timesteps, 2*(self.n_qubits+self.n_ancilla_qubits)*self.n_backward_layers), device=self.device)
        if t < self.n_timesteps:
            for tt in range(t+1, self.n_timesteps+1):
                dir, filename = get_path(self.config, type='bestparams.npy', n_data=self.n_data, n_features=self.n_features, n_qubits=self.n_qubits, n_ancilla_qubits=self.n_ancilla_qubits, n_timesteps=self.n_timesteps, n_backward_layers=self.n_backward_layers, t=tt)
                params_tot[tt-1] = torch.from_numpy(np.load(dir / filename)).to(self.device)
        params, loss_hist = self.train_timestep_t(t, inputs_last_timestep, params_tot, self.n_data, self.learning_rate)

        self._save_results_t(params.detach().cpu(), loss_hist, t)
        

    
    def train_all_timesteps(self):
        self._save_config()

        # Load diffused states
        dir, filename = get_path(self.config, type='diffusedqstates.npy', diffusion_schedule=self.diffusion_schedule, n_data=self.n_data, n_features=self.n_features, n_qubits=self.n_qubits, n_timesteps=self.n_timesteps)
        states_diffused = np.load(dir / filename) # Must be numpy array
        self.model.set_diffusionSet(states_diffused) # This already converts the states to torch tensors in the device
        inputs_last_timestep = torch.from_numpy(states_diffused[-1]).to(self.device)

        self.model.train()
        for t in range(self.n_timesteps, 0, -1): # From T to 1
            print(f"--- Training timestep {t} ---")
            if self._check_pretrained_params(t): 
                print("Found already trained parameters. Skipping this timestep...")
                continue

            params_tot = torch.zeros((self.n_timesteps, 2*(self.n_qubits+self.n_ancilla_qubits)*self.n_backward_layers), device=self.device)
            if t < self.n_timesteps:
                for tt in range(t+1, self.n_timesteps+1):
                    dir, filename = get_path(self.config, type='bestparams.npy', n_data=self.n_data, n_features=self.n_features, n_qubits=self.n_qubits, n_ancilla_qubits=self.n_ancilla_qubits, n_timesteps=self.n_timesteps, n_backward_layers=self.n_backward_layers, t=tt)
                    params_tot[tt-1] = torch.from_numpy(np.load(dir / filename)).to(self.device)
            params, loss_hist = self.train_timestep_t(t, inputs_last_timestep, params_tot, self.n_data, self.learning_rate)

            self._save_results_lastepoch(params.detach().cpu(), loss_hist, t)

    
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
        lr_scheduler = self._get_lr_scheduler()(optimizer)
        lossfn = self._get_loss_fn()
        loss_hist = []

        # Start training loop
        t0 = time.time()
        last_save = 0 # Epoch where results were last saved
        pbar = tqdm(range(self.n_epochs))
        for epoch in pbar:
            optimizer.zero_grad()

            indices = np.random.choice(states_diffused.shape[1], size=n_data, replace=False)
            true_data = states_diffused[t, indices]

            output_t = self.model.backwardOutput_t(input_tplus1, params_t)
            loss = lossfn(output_t, true_data)

            loss_value = loss.detach().cpu()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            pbar.set_postfix({
                'ℒ (loss)': f"{loss_value:.4f}",
                '💾 (last saved)': f"{last_save}"
            })

            # Check if current step is best
            if len(loss_hist) == 0 or loss_value < min(loss_hist):
                self._save_results_t(params_t.detach().cpu(), loss_hist, t, verbose=False)
                last_save = epoch

            # Save and plot stats
            self.history['lr'].append(lr_scheduler.get_last_lr()[0])
            self.history['loss'].append(loss_value)
            loss_hist.append(loss_value) # record the current loss
            if self.config['training']['live_plot']['save'] and epoch % self.config['training']['live_plot']['frequency'] == 0:
                fig = self.plotter.plot_loss(t, history=self.history)
                dir, filename = get_path(self.config, type='lossplot.html', n_data=self.n_data, n_features=self.n_features, n_qubits=self.n_qubits, n_ancilla_qubits=self.n_ancilla_qubits, n_timesteps=self.n_timesteps, n_backward_layers=self.n_backward_layers, t=t)
                fig.write_html(str(dir/filename))

        return params_t, torch.stack(loss_hist)


if __name__ == "__main__":
    main()
