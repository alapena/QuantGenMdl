from src.QDDPM_torch_angel import QDDPMDiffuser
from src.utils import set_device, get_dataset_type_and_number, get_diffusion_weights, save_config
from src.lossfns import quantum_mean_infidelity
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from functools import partial
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import math
import yaml

activation_dict = {
    'ReLU': nn.ReLU,
    'Sigmoid': nn.Sigmoid,
    'Tanh': nn.Tanh,
}

class TimeConditionedMLP(nn.Module):
    def __init__(self, input_dim: int, n_hidden_layers: int, hidden_dim: int, 
                 output_dim: int, time_embed_dim: int, activation=nn.ReLU, time_base: int = 100):
        super().__init__()
        
        # Dimension setup
        real_input_dim = input_dim #2 * input_dim
        real_output_dim = output_dim #2 * output_dim
        
        # 1. Time Embedding Network
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_embed_dim, time_base=time_base), # Non-learnable. But we want to transform our sinusoidal representation to the network's data representation.
            nn.Linear(time_embed_dim, time_embed_dim * 2),
            activation(),
            nn.Linear(time_embed_dim * 2, time_embed_dim)
        )

        # 2. Main Layers and their corresponding Time Projections (FiLM)
        self.layers = nn.ModuleList()
        self.time_projections = nn.ModuleList()
        curr_dim = real_input_dim
        for _ in range(n_hidden_layers):
            self.layers.append(nn.Linear(curr_dim, hidden_dim))
            self.time_projections.append(nn.Linear(time_embed_dim, 2 * hidden_dim)) # We concatenate the gamma and beta dimensions.
            curr_dim = hidden_dim
        self.final_layer = nn.Linear(curr_dim, real_output_dim)
        self.activation = activation()

    def forward(self, x, t):
        # x: (batch_size, n_features) complex
        # t: (batch_size,) integer/float
        
        batch_size = x.shape[0]
        x = torch.view_as_real(x).view(batch_size, -1)

        t_emb = self.time_mlp(t) # (batch_size, time_embed_dim)
        for layer, time_proj in zip(self.layers, self.time_projections):
            x = layer(x)

            condition = time_proj(t_emb)
            gamma, beta = torch.chunk(condition, 2, dim=-1)
            
            # Apply FiLM: (1 + gamma) * x + beta. El 1 es para inicializarlo en ~1, en vez de en ~0.
            x = (1 + gamma) * x + beta
            x = self.activation(x)

        x_output = self.final_layer(x)

        x_output = x_output.view(batch_size, -1, 2)
        x_output = torch.view_as_complex(x_output)
        norm = torch.linalg.norm(x_output, dim=1, keepdim=True)
        x_output = x_output / norm.clamp(min=1e-8) # Avoid division by zero
        return x_output
    

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, time_base: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.time_base = time_base

    def forward(self, t: torch.Tensor):
        # t: (batch_size,) 
        device = t.device
        half_dim = self.embedding_dim // 2
        
        # Create frequencies: omega_i = 1 / (base^(2i/D))
        frequencies = torch.exp(torch.arange(half_dim, device=device) * -(math.log(self.time_base) / half_dim)) # (half_dim)
        
        # Compute arguments: (batch_size, 1) * (1,  half_dim) -> (batch_size, half_dim)
        args = t[:, None] * frequencies[None, :]
        
        # Concatenate [sin, cos]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1) # (batch_size, embedding_dim)
        return embedding

def main():
    config = yaml.safe_load(open('config_timeconditionedmlp.yaml', 'r'))
    print("")
    device = set_device(config['device'])
    torch.set_default_device(device)

    model_config = config['model']
    model_params_config, activation = model_config['params'], activation_dict[model_config['activation']]
    model = TimeConditionedMLP(**model_params_config, activation=activation).to(device)
    lossfn = quantum_mean_infidelity
    savedir = Path(config['savedir'])

    save_config(config, savedir)

    train(config, model, lossfn, savedir, device=device)



def train(config, model: TimeConditionedMLP, lossfn, savedir: Path, device='cpu'):
    # Load from config
    model_config = config['model']
    n_data = config['dataset']['n_data']
    n_timesteps = config['diffusion']['n_timesteps']
    n_features = config['dataset']['n_features']
    batch_size = config['dataset']['batch_size']
    avg_val_loss_goal = config['training']['avg_val_loss_goal']

    training_config = config['training']
    save_interval = training_config['save_interval']
    lr = float(training_config['learning_rate'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=200)
    writer = SummaryWriter(log_dir=savedir/'tensorboard')
    avg_train_loss_epoch_hist = []
    avg_val_loss_epoch_hist = []
    epochs = []
    val_epochs = []

    diffuser = QDDPMDiffuser(config=config, n_data=n_data, n_timesteps=n_timesteps, n_features=n_features, device=device)
    diffusion_weights = get_diffusion_weights(config, device=device)
    seed = config["seed"]
    rng = np.random.default_rng(seed)
    datagen = DataGenerator(config, diffuser, diffusion_weights, rng, n_features, device=device)
    states_diffused = datagen.generate(n_timesteps)
    torch_rng = torch.Generator(device=device).manual_seed(seed)
    timesteps = torch.randint(
        1, n_timesteps+1, (n_timesteps, n_data), device=device,
        dtype=torch.int64, generator=torch_rng
    ) 
    sample_indices = torch.arange(n_data).repeat(n_timesteps, 1)
    flat_timesteps = timesteps.flatten()
    flat_offsets = sample_indices.flatten()
    flat_indices = (flat_timesteps * n_data) + flat_offsets
    flat_timesteps_minus1 = (flat_timesteps - 1)
    flat_indices_tminus1 = (flat_timesteps_minus1 * n_data) + flat_offsets
    y_t, y_tminus1 = states_diffused.view(-1, n_features)[flat_indices], states_diffused.view(-1, n_features)[flat_indices_tminus1]
    val_dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(y_t, y_tminus1, flat_timesteps), batch_size=batch_size, shuffle=False, generator=torch_rng)

    avg_val_loss_epoch = float('inf')
    best_loss = float('inf')
    last_save_epoch = 0
    epoch = 1
    pbar = tqdm()
    while avg_val_loss_epoch > avg_val_loss_goal and (epoch-last_save_epoch) < save_interval:
        model.train()
        train_loss_epoch = 0

        # Generate new data at each epoch (reproducible)
        states_diffused = datagen.generate(n_timesteps) # [t+1, n_data, n_features]     
        timesteps = torch.randint(
            1, n_timesteps+1, (n_timesteps, n_data), device=device,
            dtype=torch.int64, generator=torch_rng
        ) 
        sample_indices = torch.arange(n_data).repeat(n_timesteps, 1)
        flat_timesteps = timesteps.flatten()
        flat_offsets = sample_indices.flatten()
        flat_indices = (flat_timesteps * n_data) + flat_offsets
        flat_timesteps_minus1 = (flat_timesteps - 1)
        flat_indices_tminus1 = (flat_timesteps_minus1 * n_data) + flat_offsets
        y_t, y_tminus1 = states_diffused.view(-1, n_features)[flat_indices], states_diffused.view(-1, n_features)[flat_indices_tminus1]
        train_dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(y_t, y_tminus1, flat_timesteps), batch_size=batch_size, shuffle=True, generator=torch_rng)

        # Train
        for y_t, y_tminus1, timesteps in train_dataloader:
            y_t, y_tminus1, timesteps = y_t.to(device), y_tminus1.to(device), timesteps.to(device)
            
            optimizer.zero_grad()
            x_tminus1 = model(y_t, timesteps)
            loss = lossfn(x_tminus1, y_tminus1)
            loss.backward()
            optimizer.step()
            train_loss_value = loss.item()
            train_loss_epoch += train_loss_value
        avg_train_loss_epoch = train_loss_epoch / len(train_dataloader)
        avg_train_loss_epoch_hist.append(avg_train_loss_epoch)

        # Validate
        val_loss_epoch = 0
        if epoch % 1 == 0:
            with torch.no_grad():
                for y_t, y_tminus1, timesteps in val_dataloader:
                    y_t, y_tminus1, timesteps = y_t.to(device), y_tminus1.to(device), timesteps.to(device)
                    x_tminus1 = model(y_t, timesteps)
                    val_loss_value = lossfn(x_tminus1, y_tminus1).item()
                    val_loss_epoch += val_loss_value
            avg_val_loss_epoch = val_loss_epoch / len(val_dataloader)
            val_epochs.append(epoch)
            avg_val_loss_epoch_hist.append(avg_val_loss_epoch)

        # Save
        if avg_val_loss_epoch < best_loss:
            best_loss = avg_val_loss_epoch
            last_save_epoch = epoch
            savedir_model = savedir / 'models'
            filename = f'val_bestmodel.pt'
            path = savedir_model/filename
            savedir_model.mkdir(parents=True, exist_ok=True)
            torch.save({
                    'model_state_dict': model.state_dict(),
                    'avg_train_loss_hist': avg_train_loss_epoch_hist,
                    'avg_val_loss_hist': avg_val_loss_epoch_hist,
                    'epochs_hist': epochs,
                    'val_epochs_hist': val_epochs
                    }, path)
            
        # Log
        writer.add_scalar('Loss/Train', avg_train_loss_epoch, epoch)
        writer.add_scalar('Loss/Validation', avg_val_loss_epoch, epoch)
        writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)
        pbar.set_description(f'Epoch {epoch}, Val loss: {avg_val_loss_epoch:.3e}/{avg_val_loss_goal:.1e}, LR: {scheduler.get_last_lr()[0]:.1e}, Last_save: {last_save_epoch}')
        pbar.update(1)
        scheduler.step(avg_train_loss_epoch)
        epochs.append(epoch)
        epoch += 1
    writer.flush()
    pbar.close()

    # Last save
    savedir_model = savedir / 'models'
    filename = f'final_model.pt'
    path = savedir_model/filename
    savedir_model.mkdir(parents=True, exist_ok=True)
    torch.save({
            'model_state_dict': model.state_dict(),
            'avg_train_loss_hist': avg_train_loss_epoch_hist,
            'avg_val_loss_hist': avg_val_loss_epoch_hist,
            'epochs_hist': epochs,
            'val_epochs_hist': val_epochs
            }, path)
    
    print(f'Training finished.')

def sample(model: TimeConditionedMLP, X):
    model.eval()
    with torch.no_grad():
        for t in range(model.n_timesteps, 0, -1):
            X = model(X, t)
    return X

class DataGenerator():
    def __init__(self, config, diffuser, diffusion_weights, rng, n_features, device='cpu'):
        self.config = config
        self.diffuser = diffuser
        self.diffusion_weights = diffusion_weights
        self.rng = rng
        self.device = device
        self.batch_size = config['dataset']['batch_size']
        self.n_data = self.config['dataset']['n_data']
        self.n_features = n_features
        self.generatorfn = self._get_generatorfn()

    def _get_generatorfn(self):
        from src.generate_dataset import ndim_cluster0Gen_rng, ndim_circleYGen_rng
        type, num = get_dataset_type_and_number(self.config)
        generatorfns_dict = {
            'CLUSTER0': partial(ndim_cluster0Gen_rng, self.n_data, num, epsilon=0.08),
            'CIRCLEY': partial(ndim_circleYGen_rng, self.n_data, num)
        }
        return generatorfns_dict[type]

    def generate(self, t):
        initial_states = self.generatorfn(rng=self.rng)
        with torch.no_grad():
            states_diffused = torch.zeros((t+1, self.n_data, self.n_features), dtype=torch.complex64, device=self.device)
            states_diffused[0] = torch.from_numpy(initial_states).to(self.device)
            for tt in (range(1, t+1)):
                states_diffused[tt] = self.diffuser.set_diffusionData_t_single_step(tt, states_diffused[tt-1], self.diffusion_weights[:tt], seed=tt)
                states_diffused[tt] = states_diffused[tt] / torch.norm(states_diffused[tt], dim=1, keepdim=True) # Avoid numerical errorsdir, filename = get_path(self.config, type='diffusedqstates.npy', diffusion_schedule=diffusion_schedule_nickname, n_data=n_data, n_features=n_features, n_qubits=n_qubits, n_timesteps=n_timesteps)
            # y_t, y_tminus1, y_last = states_diffused[t].cpu(), states_diffused[t-1].cpu(), states_diffused[-1].cpu() # The two steps involved in the current timestep t, and the last timestep
        return states_diffused
    
    def generate_initial_states(self):
        states_initial = self.generatorfn(epsilon=0.08, rng=self.rng)
        with torch.no_grad():
            return torch.from_numpy(states_initial).to(self.device)
    
    def diffuse(self, states_initial, diffusion_weights, timesteps):
        with torch.no_grad():
            states_diffused = torch.zeros((t+1, self.n_data, self.n_features), dtype=torch.complex64, device=self.device)
            states_diffused[0] = torch.from_numpy(states_initial).to(self.device)

if __name__ == '__main__':
    main()