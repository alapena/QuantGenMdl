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
import pytorch_optimizer

def main():
    config = yaml.safe_load(open('config_unet.yaml', 'r'))
    print("")
    device = set_device(config.get('device', 'cpu'))
    # torch.set_default_device(device)

    n_data = config['dataset']['maxsize'] # EDITABLE
    n_timesteps = config['model']['n_timesteps'] # EDITABLE
    n_qubits = int(config['dataset']['name'].split('_')[1])
    n_features = 2**n_qubits#config['dataset']['transforms']['resize']**2 # EDITABLE
    
    # values = [2,3,4,5,6,7,8]
    # for i, value in enumerate(values):
        # print(f"---TRAINING WITH n_ancilla_qubits={value}---")
        # config["model"]["n_ancilla_qubits"] = value

    trainer = MLPTrainer(config, n_data, n_features, n_timesteps, device=device)
    trainer.training_loop()


class UNetTrainer():
    def __init__(self, config, n_data, n_features, n_timesteps, device='cpu'):
        self.device = device
        self.config = config
        self.n_data = n_data
        self.n_features = n_features
        self.n_timesteps = n_timesteps

        _, self.n_qubits = find_closest_power_of_2(n_features, return_power=True)
        self.n_epochs = self.config['training']['n_epochs']
        self.seed = self.config['seed']
        self.learning_rate = self.config['training']['learning_rate']
        self.diffusion_schedule_nickname = get_diffusion_schedule_nickname(self.config)
        self.plotter = Plotter()
        self.get_path = partial(get_path, self.config, modeltype='UNet', diffusion_schedule_nickname=self.diffusion_schedule_nickname, n_data=self.n_data, n_features=self.n_features, n_qubits=self.n_qubits, n_timesteps=self.n_timesteps)
        self.batch_size = self.config['training']['batch_size']
        self.batch_size = self.batch_size if self.batch_size is not None else self.n_data

        self.model = self._get_model().to(self.device)

    def _get_model(self):
        model_type = self.config['model']['type']
        if model_type == 'MLP':
            return MLP(
                input_dim=self.config['model']['input_dim'],
                n_hidden_layers=self.config['model']['n_hidden_layers'],
                hidden_dim=self.config['model']['hidden_dim'],
                output_dim=self.config['model']['output_dim']
            )
        elif model_type == 'UNet':
            return UNet1DModel(sample_size=self.n_features, **self.config['model']['unet_config'])
        else:
            raise NotImplementedError(f"Model type {model_type} not implemented.")
        
    def _get_loss_fn(self):
        from src.QDDPM_torch_angel import WassDistance, sinkhornDistance
        loss_type = self.config['training'].get('loss_fn', 'wass')
        if loss_type == 'sinkhorn':
            return NotImplementedError('Loss function {loss_type} not implemented.')
        elif loss_type == 'wass':
            return WassDistance
        elif loss_type == 'mse':
            return torch.nn.MSELoss(**self.config['training']['loss_fn_kwargs'])
        elif loss_type == 'mse_and_norm':
            return partial(mse_and_norm_loss, **self.config['training']['loss_fn_kwargs'])
        elif loss_type == 'quantum_mean_infidelity':
            return quantum_mean_infidelity
        else:
            raise NotImplementedError('Loss function {loss_type} not implemented.')
        
    def _get_optimizer(self):
        optimizer_type = self.config['training']['optimizer']['type']
        if optimizer_type == 'Adam':
            return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif optimizer_type == 'AdamW':
            return torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        elif optimizer_type == 'SGD':
            return torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        elif optimizer_type == 'RMSprop':
            return torch.optim.RMSprop(self.model.parameters(), lr=self.learning_rate)
        elif optimizer_type == 'SOAP':
            return pytorch_optimizer.optimizer.SOAP(self.model.parameters(), lr=self.learning_rate)
        else:
            raise NotImplementedError(f"Optimizer {optimizer_type} not implemented.")
    
    def _get_lr_scheduler(self):
        config_lr_scheduler = self.config['training'].get('lr_scheduler', 'None')
        type = config_lr_scheduler['type']
        kwargs = {k: v for k, v in config_lr_scheduler.items() if k != 'type'}
        if config_lr_scheduler['type'] == 'CosineAnnealingWarmRestarts':
            return partial(torch.optim.lr_scheduler.CosineAnnealingWarmRestarts, **kwargs)
        elif config_lr_scheduler['type'] == 'OneCycleLR':
            return partial(torch.optim.lr_scheduler.OneCycleLR, total_steps=self.n_epochs, max_lr=self.learning_rate, **kwargs)        
        elif type == 'CosineAnnealingLR':
            return partial(torch.optim.lr_scheduler.CosineAnnealingLR, T_max=self.n_epochs, **kwargs)
        elif type == 'ExponentialLR':
            return partial(torch.optim.lr_scheduler.ExponentialLR, **kwargs)
        elif type == 'LinearLR':
            return partial(torch.optim.lr_scheduler.LinearLR, **kwargs)
        elif type == 'ReduceLROnPlateau':
            return partial(torch.optim.lr_scheduler.ReduceLROnPlateau, **kwargs)
        elif type == 'sequential_2cosine':
            kwargs1 = self.config['training']['lr_scheduler']['sched1']
            sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, **kwargs1)
            kwargs2 = self.config['training']['lr_scheduler']['sched2']
            sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, **kwargs2)
            sched2.base_lrs = [kwargs1['eta_min']]
            return torch.optim.lr_scheduler.SequentialLR(self.optimizer, schedulers=[sched1, sched2], milestones=[kwargs1['T_max']])
        elif config_lr_scheduler['type'] == 'ctt':
            return partial(torch.optim.lr_scheduler.CosineAnnealingLR, T_max=self.n_epochs, eta_min=self.learning_rate)
        else:
            raise NotImplementedError(f"Learning rate scheduler {config_lr_scheduler['type']} not implemented.")

    def _save_config(self):
        dir, filename = self.get_path(type='config.yaml')
        with open(dir/filename, 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False, sort_keys=False)

    def _save_best_checkpoint(self, optimizer, lr_scheduler, epoch, best_loss):
        dir, filename = self.get_path(type='bestparams.npy', t=epoch)
        # self.model.save_pretrained(dir) # We save both: unet architecture (with parameters)
        torch.save({ 
            'model_state_dict': self.model.state_dict(), # and parameters inside torch file.
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'epoch': epoch,
            'best_loss': best_loss,
        }, dir / "bestmodel.pt") 

    def _save_last_checkpoint(self, optimizer, lr_scheduler, epoch, final_loss, best_loss):
        dir, filename = self.get_path(type='finalparams.npy', t=epoch)
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
            torch.set_default_device(self.device)
            dir, filename = self.get_path(type='initialqstates.npy')
            dataset = torch.from_numpy(np.load(dir / filename)).to(self.device)

            diffuser = QDDPMDiffuser(self.n_qubits, self.n_timesteps, self.n_data, device=self.device)
            diffusion_weights = get_diffusion_weights(self.config, self.device)
            states = torch.zeros((self.n_timesteps+1, self.n_data, 2**self.n_qubits), device=self.device, dtype=torch.complex64)
            states[0] = dataset
            for t in tqdm(range(1, self.n_timesteps+1)):
                # states[t] = model.set_diffusionData_t(t, states[0], diffusion_weights[:t], seed=t)
                states[t] = diffuser.set_diffusionData_t_single_step(t, states[t-1], diffusion_weights[:t], seed=t)
                states[t] = states[t] / torch.norm(states[t], dim=1, keepdim=True) # Avoid numerical errors

            dir, filename = self.get_path(type='diffusedqstates.npy')
            np.save(dir/filename, states.cpu().numpy())
            print(f"Saved diffused quantum states in {dir / filename}")
            torch.set_default_device('cpu')

    def training_loop(self):
        self._save_config()
        self._generate_diffusedstates()
        dir, filename = self.get_path(type='tensorboard_logs')
        writer = SummaryWriter(log_dir=dir)
        optimizer = self._get_optimizer()
        self.optimizer = optimizer
        lr_scheduler = self._get_lr_scheduler()
        # lr_scheduler = self._get_lr_scheduler()(optimizer)
        lossfn = self._get_loss_fn()
        dir, filename = self.get_path(type='diffusedqstates.npy')
        targets = torch.from_numpy(np.load(dir/filename)).to(self.device) # [T+1, n_data, n_features (complex)]
        targets = torch.view_as_real(targets) # [T+1, n_data, n_features (real), 2]
        targets = targets.permute(1, 0, 3, 2) # [n_data, T+1, n_channels=2, n_features]
        inputs_last_timestep = targets[:,-1] # The most diffused states as inputs
        train_dataloader = torch.utils.data.DataLoader(targets, batch_size=self.batch_size, shuffle=True)

        last_save = 0
        best_loss = float('inf')
        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        global_step = 0
        pbar = tqdm(range(self.n_epochs))
        self.model.train()
        for epoch in pbar:
            pbar.set_description(f"Epoch {epoch}/{self.n_epochs}")
            epoch_loss_sum = 0.0
            for batch in train_dataloader:
                batch = batch.to(self.device) # [n_data, T+1, n_channels=2, n_features]
                optimizer.zero_grad()

                timesteps = torch.randint(
                    1, self.n_timesteps+1, (self.batch_size,), generator=generator, device=inputs_last_timestep.device,
                    dtype=torch.int64
                )
                batch_indices = torch.arange(self.batch_size)
                # indexs = torch.randint(0, self.n_data, (self.batch_size,), generator=generator, device=inputs_last_timestep.device)
                noisy_states = batch[batch_indices, timesteps, :, :] # [n_data, n_channels=2, n_features]
                pred = self.model(noisy_states, timesteps, return_dict=False)[0] # [n_data, n_channels=2, n_features]
                # Normalize
                norms = torch.sqrt(torch.pow(pred, 2).sum(dim=1).sum(dim=1))
                pred = pred / norms.reshape(pred.shape[0], 1, 1)

                pred_complex = pred[:,0,:] + 1j*pred[:,1,:] # [n_data, n_features] complex tensor
                true_complex = noisy_states[:,0,:] + 1j*noisy_states[:,1,:]
                loss = lossfn(pred_complex, true_complex)
                loss_value = loss.detach().cpu()
                epoch_loss_sum += loss_value

                loss.backward()
                optimizer.step()

                # Save and plot stats
                writer.add_scalar('Loss/train_step', loss_value, global_step)
                writer.add_scalar('Average norm before hardcoding', norms.mean(), global_step)
                # writer.add_scalar('Loss_mse', mse_term, epoch)
                # writer.add_scalar('Loss_norm', norm_term, epoch)
                global_step += 1

            pbar.set_postfix({
                'ℒ (loss)': f"{loss_value:.3e}",
                '💾 (last saved)': f"{last_save}"
            })

            # Check if current step is best
            avg_epoch_loss = epoch_loss_sum / len(train_dataloader)
            lr_scheduler.step()
            writer.add_scalar('Loss/train_epoch', avg_epoch_loss, epoch)
            writer.add_scalar('Learning Rate', lr_scheduler.get_last_lr()[0], epoch)
            if avg_epoch_loss < best_loss:
                self._save_best_checkpoint(optimizer, lr_scheduler, epoch, best_loss)
                last_save = epoch
                best_loss = avg_epoch_loss

        self._save_last_checkpoint(optimizer, lr_scheduler, epoch, final_loss=loss_value, best_loss=best_loss)
        writer.flush()


def mse_and_norm_loss(pred, true, lambd=0.01, return_terms=False):
    mse = torch.nn.MSELoss()

    norms = torch.sqrt(torch.pow(pred, 2).sum(dim=1).sum(dim=1))
    norms = norms.reshape(pred.shape[0], 1, 1)
    norm_term = lambd * mse(norms, torch.ones(pred.shape[0], 1, 1))

    mse_term = mse(pred, true)

    if return_terms:
        return mse_term + norm_term, mse_term, norm_term
    else:
        return mse_term + norm_term
    
def quantum_mean_infidelity(pred, true):
    # pred and true are [n_data, n_features] complex tensors
    inner_product = torch.linalg.vecdot(true, pred, dim=-1) # [n_data]
    fidelity = inner_product.abs().pow(2)
    infidelity = 1 - fidelity
    return infidelity.mean()

def complex_to_interleaved_real(z):
    return torch.stack([z.real, z.imag], dim=-1).view(*z.shape[:-1], -1)

def interleaved_real_to_complex(z):
    z = z.view(*z.shape[:-1], -1, 2)
    return torch.complex(z[..., 0], z[..., 1])


class MLP(torch.nn.Module):
    def __init__(self, input_dim, n_hidden_layers, hidden_dim, output_dim):
        super().__init__()
        layers = []
        for _ in range(n_hidden_layers):
            layers.append(torch.nn.Linear(input_dim, hidden_dim))
            layers.append(torch.nn.Sigmoid())
            input_dim = hidden_dim
        layers.append(torch.nn.Linear(input_dim, output_dim))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x, t):
        x_input = torch.cat([x, t.unsqueeze(-1).float()], dim=-1)
        return self.mlp(x_input)
    
class MLPTrainer(UNetTrainer):
    def __init__(self, config, n_data, n_features, n_timesteps, device='cpu'):
        super().__init__(config, n_data, n_features, n_timesteps, device=device)
    
    def training_loop(self):
        self._save_config()
        self._generate_diffusedstates()
        dir, filename = self.get_path(type='tensorboard_logs')
        writer = SummaryWriter(log_dir=dir)
        optimizer = self._get_optimizer()
        self.optimizer = optimizer
        # lr_scheduler = self._get_lr_scheduler()
        lr_scheduler = self._get_lr_scheduler()(optimizer)
        lossfn = self._get_loss_fn()
        dir, filename = self.get_path(type='diffusedqstates.npy')
        targets = torch.from_numpy(np.load(dir/filename)).to(self.device) # [T+1, n_data, n_features (complex)]
        targets_interleaved = complex_to_interleaved_real(targets) # [T+1, n_data, n_features (real*2)]
        targets_interleaved = targets_interleaved.permute(1, 0, 2) # [n_data, T+1, n_features*2]

        inputs_last_timestep = targets_interleaved[:,-1] # The most diffused states as inputs
        train_dataloader = torch.utils.data.DataLoader(targets_interleaved, batch_size=self.batch_size, shuffle=True)

        last_save = 0
        best_loss = float('inf')
        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        global_step = 0
        pbar = tqdm(range(self.n_epochs))
        self.model.train()
        for epoch in pbar:
            pbar.set_description(f"Epoch {epoch}/{self.n_epochs}")
            epoch_loss_sum = 0.0
            for batch in train_dataloader:
                batch = batch.to(self.device) # [n_data, T+1, n_features*2]
                optimizer.zero_grad()

                timesteps = torch.randint(
                    1, self.n_timesteps+1, (self.batch_size,), generator=generator, device=batch.device,
                    dtype=torch.int64
                )
                batch_indices = torch.arange(self.batch_size)
                # indexs = torch.randint(0, self.n_data, (self.batch_size,), generator=generator, device=inputs_last_timestep.device)
                noisy_states = batch[batch_indices, timesteps, :] # [n_data, n_features]

                pred = self.model(noisy_states, timesteps) # [n_data, 2*n_features]
                pred = interleaved_real_to_complex(pred) # [n_data, n_features] complex tensor
                # Normalize
                norms = torch.linalg.norm(pred, dim=1, keepdim=True)
                pred = pred / norms
                # norms = torch.sqrt(torch.pow(pred, 2).sum(dim=1).sum(dim=1))
                # pred = pred / norms.reshape(pred.shape[0], 1, 1)

                # pred_complex = pred[:,0,:] + 1j*pred[:,1,:] # [n_data, n_features] complex tensor
                # true_complex = noisy_states[:,0,:] + 1j*noisy_states[:,1,:]

                timesteps_true = timesteps-1
                true = batch[batch_indices, timesteps_true,:]
                true = interleaved_real_to_complex(true)
                loss = lossfn(pred, true)
                loss_value = loss.detach().cpu()
                epoch_loss_sum += loss_value

                loss.backward()
                optimizer.step()

                # Save and plot stats
                writer.add_scalar('Loss/train_step', loss_value, global_step)
                writer.add_scalar('Average norm before hardcoding', norms.mean(), global_step)
                # writer.add_scalar('Loss_mse', mse_term, epoch)
                # writer.add_scalar('Loss_norm', norm_term, epoch)
                global_step += 1

            pbar.set_postfix({
                'ℒ (loss)': f"{loss_value:.3e}",
                '💾 (last saved)': f"{last_save}"
            })

            # Check if current step is best
            avg_epoch_loss = epoch_loss_sum / len(train_dataloader)
            lr_scheduler.step()
            writer.add_scalar('Loss/train_epoch', avg_epoch_loss, epoch)
            writer.add_scalar('Learning Rate', lr_scheduler.get_last_lr()[0], epoch)
            if avg_epoch_loss < best_loss:
                self._save_best_checkpoint(optimizer, lr_scheduler, epoch, best_loss)
                last_save = epoch
                best_loss = avg_epoch_loss

        self._save_last_checkpoint(optimizer, lr_scheduler, epoch, final_loss=loss_value, best_loss=best_loss)
        writer.flush()
        

if __name__ == "__main__":
    main()