import numpy as np
from tqdm import tqdm

from src.trainers.default_trainer import QDDPMTrainer
from src.utils import get_path, set_device
import torch
import yaml
import optuna

TIMESTEP = 39

def main():
    config = yaml.safe_load(open('config.yaml', 'r'))
    print("")
    device = set_device(config.get('device', 'cpu'))
    torch.set_default_device(device)

    n_data = config['dataset']['maxsize'] # EDITABLE
    n_timesteps = config['model']['n_timesteps'] # EDITABLE
    n_qubits = int(config['dataset']['name'].split('_')[1])
    n_features = 2**n_qubits #config['dataset']['transforms']['resize']**2 # EDITABLE
    
    optuna_trainer = OptunaObjective(config, n_data, n_features, n_timesteps, device=device)
    n_ancilla_qubits_list = list(range(1, 2*n_qubits+1))
    optuna_trainer(n_ancilla_qubits_list)
    
    # trainer.train_all_timesteps()
    

class OptunaObjective():
    def __init__(self, t, config, n_data, n_features, n_timesteps, device='cpu'):
        self.t = t
        self.config = config
        self.device = device

        self.n_data = n_data
        self.n_features = n_features
        self.n_timesteps = n_timesteps

    def study_objective(self, n_ancilla_qubits_list):
        for n_ancilla_qubits in n_ancilla_qubits_list:
            study = optuna.create_study(
                direction = "minimize",
                sampler = optuna.samplers.TPESampler(seed=0),
                pruner = optuna.pruners.HyperbandPruner(min_resource=5)
            )
            study.optimize(lambda trial: self.objective(trial, n_ancilla_qubits), n_trials=11)
        
        best_L = study.best_params[f"L_t{TIMESTEP}"]
        best_loss = study.best_value

        self.best_layers = {
            "L": best_L,
            "loss": float(best_loss)
        }

        # Save results. Overwrites the file after new timestep finishes.
        dir = Path(f"data/{self.dataset_name}/results_optuna") # ! ARREGLAR
        dir.mkdir(parents=True, exist_ok=True)
        filename = f"results.yaml"
        with open(dir / filename, 'w') as f:
            yaml.dump(self.best_layers, f, default_flow_style=False)

    def objective(self, trial, n_ancilla_qubits):
        try:
            n_backward_layers = trial.suggest_int(f"L_t{TIMESTEP}", 1, 12)

            trainer = QDDPMTrainerOptuna(self.config, self.n_data, self.n_features, self.n_timesteps, n_ancilla_qubits=n_ancilla_qubits, device=self.device)
            loss_value =trainer.train_single_timestep_optuna(self.t, trainer.inputs_last_timestep, trainer.params_tot, self.n_data, trainer.lr)
            return loss_value
        
        except torch.cuda.OutOfMemoryError:
            # Clear cache to prevent the next trial from failing immediately
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print(f"Trial {trial.number} failed: OOM with na_t={n_ancilla_qubits}, L_t={n_backward_layers}. Returning penalty.")
            return float('inf')
    
class QDDPMTrainerOptuna(QDDPMTrainer):
    def __init__(self, config, n_data, n_features, n_timesteps, n_ancilla_qubits, device='cpu'):
        super().__init__(config, n_data, n_features, n_ancilla_qubits, n_timesteps, device)

    def train_single_timestep_optuna(self, t, inputs_last_timestep, params_tot, n_data, lr):
        # Load diffused states
        dir, filename = get_path(self.config, type='diffusedqstates.npy', diffusion_schedule=self.diffusion_schedule_nickname, n_data=self.n_data, n_features=self.n_features, n_qubits=self.n_qubits, n_timesteps=self.n_timesteps)
        states_diffused = np.load(dir / filename) # Must be numpy array
        self.model.set_diffusionSet(states_diffused) # This already converts the states to torch tensors in the device
        inputs_last_timestep = torch.from_numpy(states_diffused[-1]).to(self.device)
        
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
                # self._save_results_t(params_t.detach().cpu(), loss_hist, t, verbose=False)
                last_save = epoch

            # Save and plot stats
            self.history['lr'].append(lr_scheduler.get_last_lr()[0])
            self.history['loss'].append(loss_value)
            loss_hist.append(loss_value) # record the current loss
            if self.config['training']['live_plot']['save'] and epoch % self.config['training']['live_plot']['frequency'] == 0:
                fig = self.plotter.plot_loss(t, history=self.history)
                dir, filename = get_path(self.config, type='lossplot.html', new_subfolder="optuna", n_data=self.n_data, n_features=self.n_features, n_qubits=self.n_qubits, n_ancilla_qubits=self.n_ancilla_qubits, n_timesteps=self.n_timesteps, n_backward_layers=self.n_backward_layers, t=t)
                fig.write_html(str(dir/filename))

        return loss_value

if __name__ == "__main__":
    main()