# from src.trainers.default_trainer import MSQDDPMTrainer
from src.utils import find_closest_power_of_2, get_diffusion_schedule_nickname, get_diffusion_weights
from src.utils import get_path
from src.QDDPM_torch_angel import QDDPM, QDDPMDiffusionModel
from src.trainers.basic_trainers import QDDPMGeneratorInitialqstates
from src.plot import Plotter
from functools import partial
from tqdm import tqdm
import numpy as np
import torch
import yaml
from src.utils import set_device
import torch
import yaml

def main():
    config = yaml.safe_load(open('config.yaml', 'r'))
    print("")
    device = set_device(config.get('device', 'cpu'))
    torch.set_default_device(device)

    n_data = config['dataset']['maxsize'] # EDITABLE
    n_timesteps = config['model']['n_timesteps'] # EDITABLE
    # n_qubits = int(config['dataset']['name'].split('_')[1])
    n_features = config['dataset']['transforms']['resize']**2 # EDITABLE
    
    # values = [2,3,4,5,6,7,8]
    # for i, value in enumerate(values):
        # print(f"---TRAINING WITH n_ancilla_qubits={value}---")
        # config["model"]["n_ancilla_qubits"] = value

    trainer = QDDPMTrainer(config, n_data, n_features, n_timesteps, device=device)
    trainer.train_all_timesteps()
        
    # trainer.train_single_timestep(t=40)


class QDDPMTrainer():
    def __init__(self, config, n_data, n_features, n_timesteps, device='cpu'):
        self.device = device
        self.config = config
        self.n_data = n_data
        self.n_features = n_features
        self.n_timesteps = n_timesteps

        _, self.n_qubits = find_closest_power_of_2(n_features, return_power=True)
        self.n_zero_ancilla_qubits = self.config['model']['n_zero_ancilla_qubits']
        self.n_haar_ancilla_qubits = self.config['model']['n_haar_ancilla_qubits']
        self.n_ancilla_qubits = self.n_zero_ancilla_qubits + self.n_haar_ancilla_qubits
        self.n_backward_layers = self.config['model']['n_backward_layers']
        self.n_epochs = self.config['training']['n_epochs']
        self.seed = self.config['seed']
        self.learning_rate = self.config['training']['learning_rate']
        self.diffusion_schedule_nickname = get_diffusion_schedule_nickname(self.config)

        self.model = QDDPM(self.n_qubits, self.n_zero_ancilla_qubits, self.n_haar_ancilla_qubits, self.n_timesteps, self.n_backward_layers, seed=self.seed, device=self.device).to(self.device)

        self.n_params = 2 * self.model.n_tot * self.model.L
        self.plotter = Plotter()

        self.get_path = partial(get_path, self.config, diffusion_schedule_nickname=self.diffusion_schedule_nickname, n_data=self.n_data, n_features=self.n_features, n_qubits=self.n_qubits, n_zero_ancilla_qubits=self.n_zero_ancilla_qubits, n_haar_ancilla_qubits=self.n_haar_ancilla_qubits, n_timesteps=self.n_timesteps, n_backward_layers=self.n_backward_layers)

    def _get_loss_fn(self):
        from src.QDDPM_torch_angel import WassDistance, sinkhornDistance, maximum_mean_discrepancy
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
            T_0 = config_lr_scheduler['T_0']
            T_mult = config_lr_scheduler['T_mult']
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
        dir, filename = self.get_path(type='bestparams.npy', t=t)
        path = dir/filename
        return path.exists()

    def _save_config(self):
        dir, filename = self.get_path(type='config.yaml')
        with open(dir/filename, 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False, sort_keys=False)

    def _save_results_t(self, params, loss_hist, t, verbose=True):
        dir, filename = self.get_path(type='bestparams.npy', t=t)
        np.save(dir/filename, params)
        if verbose:
            print(f"Saved parameters at {dir/filename}.\nCorresponding loss: {loss_hist[-1]:.3e}.")

        dir, filename = self.get_path(type='bestlosshist.npy', t=t)
        np.save(dir/filename, loss_hist)

    def _save_results_lastepoch(self, params, loss_hist, t, verbose=True):
        dir, filename = self.get_path(type='finalparams.npy', t=t)
        np.save(dir / filename, params)
        if verbose:
            print(f"Saved parameters at {dir/filename}.\nCorresponding loss: {loss_hist[-1]:.3e}.")

        dir, filename = self.get_path(type='finallosshist.npy', t=t)
        np.save(dir / filename, loss_hist)

    def _generate_diffusedstates(self):
        dir, filename = self.get_path(type='diffusedqstates.npy', diffusion_schedule=self.diffusion_schedule_nickname)
        path = dir/filename
        if path.exists():
            return path
        else:
            print("Forward diffused states not found. Generating them...")

            # Check if initial states exist
            dir, filename = self.get_path(type='initialqstates.npy')
            path = dir/filename
            if not path.exists():
                # Generate initial states
                print("Initial quantum states not found. Generating them...")
                generator_initialqstates = QDDPMGeneratorInitialqstates(self.config)
                generator_initialqstates.generate_initialqstates()

            # Everything checked. Diffuse.
            dir, filename = self.get_path(type='initialqstates.npy')
            dataset = torch.from_numpy(np.load(dir / filename)).to(self.device)

            diffuser = QDDPMDiffusionModel(self.n_qubits, self.n_timesteps, self.n_data, device=self.device)
            diffusion_weights = get_diffusion_weights(self.config, self.device)
            states = torch.zeros((self.n_timesteps+1, self.n_data, 2**self.n_qubits), device=self.device, dtype=torch.complex64)
            states[0] = dataset
            for t in tqdm(range(1, self.n_timesteps+1)):
                # states[t] = model.set_diffusionData_t(t, states[0], diffusion_weights[:t], seed=t)
                states[t] = diffuser.set_diffusionData_t(t, states[t-1], diffusion_weights[:t], seed=t)
                states[t] = states[t] / torch.norm(states[t], dim=1, keepdim=True) # Avoid numerical errors

            dir, filename = self.get_path(type='diffusedqstates.npy')
            np.save(dir/filename, states.cpu().numpy())
            print(f"Saved diffused quantum states in {dir / filename}")


    def _train_timestep_t(self, t, inputs_last_timestep, params_tot, n_data, lr):
        self.history = {
            'lr': [],
            'loss': [],
        }
        seed = self.config['seed']
        input_tplus1 = self.model.prepareInput_t(inputs_last_timestep, params_tot, t, n_data) # Same Haar for all epochs, but different for each data state.
        states_diffused = self.model.states_diff

        # initialize parameters
        np.random.seed(seed)
        params_t = torch.tensor(np.random.normal(size=2 * self.model.n_tot * self.model.L), device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([params_t], lr=lr)
        lr_scheduler = self._get_lr_scheduler()(optimizer)
        lossfn = self._get_loss_fn()
        loss_hist = []

        # Start training loop
        last_save = 0 # Epoch where results were last saved
        pbar = tqdm(range(self.n_epochs))
        self.model.train()
        for epoch in pbar:
            optimizer.zero_grad()

            indices = np.random.choice(states_diffused.shape[1], size=n_data, replace=False)
            true_data = states_diffused[t, indices]

            measured_full = self.model.backwardOutput_t(input_tplus1, params_t)
            output_t = self.model._trace_out_ancilla_vmap(measured_full)
            loss = lossfn(output_t, true_data)

            loss_value = loss.detach().cpu()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            pbar.set_postfix({
                'ℒ (loss)': f"{loss_value:.3e}",
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
                fig = self.plotter.plot_loss(t, history=self.history, logscale=True)
                dir, filename = self.get_path(type='lossplot.html', t=t)
                fig.write_html(str(dir/filename))

        return params_t, torch.stack(loss_hist)

    
    def train_all_timesteps(self):
        self._save_config()
        self._generate_diffusedstates()
        dir, filename = self.get_path(type='diffusedqstates.npy', diffusion_schedule=self.diffusion_schedule_nickname)
        states_diffused = np.load(dir/filename) # Must be numpy array
        self.model.set_diffusionSet(states_diffused) # This already converts the states to torch tensors in the device

        inputs_last_timestep = torch.from_numpy(states_diffused[-1]).to(self.device)
        for t in range(self.n_timesteps, 0, -1): # From T to 1
            print(f"--- Training timestep {t} ---")
            if self._check_pretrained_params(t): 
                print("Found already trained parameters. Skipping this timestep...")
                continue

            params_tot = torch.zeros((self.n_timesteps, 2*(self.n_qubits+self.n_zero_ancilla_qubits+self.n_haar_ancilla_qubits)*self.n_backward_layers), device=self.device)
            
            # Load previous parameters (from t+1 to T)
            if t < self.n_timesteps:
                for tt in range(t+1, self.n_timesteps+1):
                    dir, filename = self.get_path(type='bestparams.npy', t=tt)
                    params_tot[tt-1] = torch.from_numpy(np.load(dir / filename)).to(self.device)
            
            # Train current timestep t
            params, loss_hist = self._train_timestep_t(t, inputs_last_timestep, params_tot, self.n_data, self.learning_rate)

            self._save_results_lastepoch(params.detach().cpu(), loss_hist, t)

class QDDPMGeneratorInitialqstates():
    def __init__(self, config):
        self.config = config

    def _ndim_cluster0Gen(self, n_data, n_qubits, epsilon, seed=None):
        np.random.seed(seed)
        states0 = np.zeros((n_data, 2**n_qubits), dtype=np.complex64)
        states0[:, 0] = 1.
        statesi = np.zeros((2**n_qubits-1, n_data, 2**n_qubits), dtype=np.complex64)
        for i in range(1, 2**n_qubits):
            statesi[i-1, :, i] = 1. + 0.j

        rng = np.random.default_rng(seed=seed)
        re_c = rng.normal(loc=0.0, scale=1.0, size=(2**n_qubits-1, n_data))
        im_c = rng.normal(loc=0.0, scale=1.0, size=(2**n_qubits-1, n_data))
        c = re_c + 1j*im_c
        c = c[:, :, np.newaxis]
        statesi = epsilon*c*statesi
        states = states0 + np.sum(statesi, axis=0)
        states /= np.linalg.norm(states, axis=1, keepdims=True)
        return states

    def _get_dataset(self):
        seed = self.config['seed']
        dataset_config = self.config["dataset"]
        name = dataset_config["name"]
        parts = name.split('_')
        type = parts[0]

        if type == 'CLUSTER0':
            size = dataset_config['maxsize']
            size = size if size is not None else 60000
            n_qubits = parts[1]
            dataset = self._ndim_cluster0Gen(size, int(n_qubits), epsilon=0.06, seed=seed)
        
        return dataset

    def generate_initialqstates(self):
        dataset = self._get_dataset()

        # Save the generated quantum states
        n_data = dataset.shape[0]
        n_features = dataset.shape[1]
        _, n_qubits = find_closest_power_of_2(n_features, return_power=True)
        dims_to_pad = 2**n_qubits-dataset.shape[1]
        dataset = np.pad(dataset, ((0,0), (0,dims_to_pad, 0)), 'constant', constant_values=0) if dims_to_pad > 0 else dataset
        dir, filename = get_path(self.config, type='initialqstates.npy', n_data=n_data, n_features=n_features, n_qubits=n_qubits)
        np.save(dir / filename, dataset)
        print(f"Dataset saved in {dir / filename}")

if __name__ == "__main__":
    main()
