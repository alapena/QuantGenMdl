from src.utils import find_closest_power_of_2, get_diffusion_schedule_nickname, get_diffusion_weights
from src.utils import get_path
from src.QDDPM_torch_angel import QDDPM, QDDPMDiffuser
from src.trainers.basic_trainers import QDDPMGeneratorInitialqstates
from src.plot import Plotter
from functools import partial
from tqdm import tqdm
from diffusers import UNet1DModel
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
import yaml
from src.utils import set_device
import torch
import yaml

def main():
    config = yaml.safe_load(open('config_unet.yaml', 'r'))
    print("")
    device = set_device(config.get('device', 'cpu'))
    torch.set_default_device(device)

    n_data = config['dataset']['maxsize'] # EDITABLE
    n_timesteps = config['model']['n_timesteps'] # EDITABLE
    n_qubits = 1 #int(config['dataset']['name'].split('_')[1])
    n_features = 2**n_qubits # config['dataset']['transforms']['resize']**2 # EDITABLE
    
    # values = [2,3,4,5,6,7,8]
    # for i, value in enumerate(values):
        # print(f"---TRAINING WITH n_ancilla_qubits={value}---")
        # config["model"]["n_ancilla_qubits"] = value

    trainer = UNetTrainer(config, n_data, n_features, n_timesteps, device=device)
    trainer.train_all_timesteps()


class UNetTrainer():
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
        self.n_params = 2 * self.model.n_tot * self.model.L
        self.plotter = Plotter()
        self.get_path = partial(get_path, self.config, diffusion_schedule_nickname=self.diffusion_schedule_nickname, n_data=self.n_data, n_features=self.n_features, n_qubits=self.n_qubits, n_zero_ancilla_qubits=self.n_zero_ancilla_qubits, n_haar_ancilla_qubits=self.n_haar_ancilla_qubits, n_timesteps=self.n_timesteps, n_backward_layers=self.n_backward_layers)

        self.model = UNet1DModel(sample_size=self.n_features, **self.config['model']['unet_config']).to(self.device)

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

    def _save_config(self):
        dir, filename = self.get_path(type='config.yaml')
        with open(dir/filename, 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False, sort_keys=False)

    def _save_best_checkpoint(self, optimizer, lr_scheduler, epoch, best_loss):
        dir, filename = self.get_path(type='bestparams.npy')
        # self.model.save_pretrained(dir) # We save both: unet architecture (with parameters)
        torch.save({ 
            'model_state_dict': self.model.state_dict(), # and parameters inside torch file.
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'epoch': epoch,
            'best_loss': best_loss,
        }, dir / "bestmodel.pt") 

    def _save_last_checkpoint(self, optimizer, lr_scheduler, epoch, final_loss, best_loss):
        dir, filename = self.get_path(type='finalparams.npy')
        # self.model.save_pretrained(dir) # We save both: unet architecture (with parameters)
        torch.save({ 
            'model_state_dict': self.model.state_dict(), # and parameters inside torch file.
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'epoch': epoch,
            'best_loss': best_loss,
            'final_loss': final_loss,
        }, dir / "finalmodel.pt") 

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

            diffuser = QDDPMDiffuser(self.n_qubits, self.n_timesteps, self.n_data, device=self.device)
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

    def training_loop(self):
        self._save_config()
        self._generate_diffusedstates()
        writer = SummaryWriter(log_dir=self.get_path(type='tensorboard_logs'))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        lr_scheduler = self._get_lr_scheduler()(optimizer)
        lossfn = self._get_loss_fn()
        dir, filename = self.get_path(type='diffusedqstates.npy')
        targets = torch.from_numpy(np.load(dir/filename)).to(self.device) # [T+1, n_data, n_features]
        targets = targets[:, :, None, :] # [T+1, n_data, n_channels=1, n_features]
        inputs_last_timestep = targets[-1] # The most diffused states as inputs

        last_save = 0
        best_loss = float('inf')
        pbar = tqdm(range(self.n_epochs))
        self.model.train()
        for epoch in pbar:
            pbar.set_description(f"Epoch {epoch}/{self.n_epochs}")

            timesteps = torch.randint(
                0, self.n_timesteps, (self.n_data,), device=inputs_last_timestep.device,
                dtype=torch.int64
            )
            noisy_states = targets[timesteps, :, :, :]
            pred = self.model(noisy_states, timesteps, return_dict=False)[0] # [n_data, n_channels=1, n_features]
            loss = lossfn(pred, noisy_states)

            loss_value = loss.detach().cpu()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            # Logs

            pbar.set_postfix({
                'ℒ (loss)': f"{loss_value:.3e}",
                '💾 (last saved)': f"{last_save}"
            })

            # Check if current step is best
            if loss_value < best_loss:
                self._save_best_checkpoint(optimizer, lr_scheduler, epoch, best_loss)
                last_save = epoch
                best_loss = loss_value

            # Save and plot stats
            writer.add_scalar('Loss/train', loss_value, epoch)
            writer.add_scalar('Learning Rate', lr_scheduler.get_last_lr()[0], epoch)
        self._save_last_checkpoint(optimizer, lr_scheduler, epoch, final_loss=loss_value, best_loss=best_loss)