from src.QDDPM_torch_angel import DiffusionModel
from src.utils import get_path, set_device, find_closest_power_of_2
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import yaml
import re

def main():
    config = yaml.safe_load(open('config.yaml', 'r'))
    dataset_config = config['dataset']
    model_config = config['model']
    device = set_device(config.get('device', 'cpu'))
    torch.set_default_device(device)

    # Load the states
    n_data = dataset_config['maxsize'] # EDITABLE
    n_pixels = int(np.square(dataset_config['transforms']['resize'])) # EDITABLE
    _, n_qubits = find_closest_power_of_2(n_pixels, return_power=True)

    dir, filename = get_path(config, type='initialqstates', n_data=n_data, n_pixels=n_pixels, n_qubits=n_qubits) # Editable
    dataset = torch.from_numpy(np.load(dir / filename)).to(device)

    # Initialize the model
    n_timesteps, n_data = config['model']['n_timesteps'], dataset.shape[0]
    model = DiffusionModel(n_qubits, n_timesteps, n_data, device=device)

    # Get the diffused states
    max_diffusion_weight = torch.tensor(model_config.get('max_diffusion_weight', 4.0), device=device)
    diffusion_schedule = model_config.get('diffusion_schedule', 'linear')

    match_pow = re.match(r'pow(\d+)', diffusion_schedule)
    match_int = re.match(r'powint(\d+)', diffusion_schedule)
    if diffusion_schedule == 'linear':
        diffusion_weights = torch.linspace(1., max_diffusion_weight, n_timesteps, device=device) # Hyperparameter that controls the 'size' of diffusion steps.
    elif diffusion_schedule == 'quadratic':
        linear_steps = torch.linspace(1., torch.sqrt(max_diffusion_weight), n_timesteps, device=device)
        diffusion_weights = torch.pow(linear_steps, 2)
    elif diffusion_schedule == 'pow6':
        # y = 3* (x/4)**6 + 1 \in [1, 4]
        x = torch.linspace(1., torch.tensor(4), n_timesteps, device=device)
        diffusion_weights = 1 + 3*torch.pow((x/4), 6)
    elif match_pow:
        n = int(match_pow.group(1)) 
        x = torch.linspace(1., torch.tensor(4), n_timesteps, device=device)
        diffusion_weights = 1 + 3*torch.pow((x/4), n)
    elif match_int:
        n = int(match_int.group(1)) 
        linear_steps = torch.linspace(1., torch.tensor(n_timesteps+1), n_timesteps+1, device=device)
        diffusion_weights = torch.pow((linear_steps/(n_timesteps+1)), n)
    elif diffusion_schedule == 'linT':
        diffusion_weights = 1/(n_timesteps+1)*torch.linspace(1., torch.tensor(2*n_timesteps+1), n_timesteps+1, device=device)
    elif diffusion_schedule == 'lin2T':
        diffusion_weights = 1/(n_timesteps+1)*torch.linspace(1., 2*torch.tensor(n_timesteps+1), n_timesteps+1, device=device)
    elif diffusion_schedule == 'lin3T':
        diffusion_weights = 1/(n_timesteps+1)*torch.linspace(1., 3*torch.tensor(n_timesteps+1), n_timesteps+1, device=device)
    elif diffusion_schedule == 'lin2T+2':
        diffusion_weights = 1/(n_timesteps+1)*torch.linspace(1., 2*torch.tensor(n_timesteps+2), n_timesteps+3, device=device)
    elif diffusion_schedule == 'lin2.5T':
        diffusion_weights = 1/(n_timesteps+1)*torch.linspace(1., 2.5*torch.tensor(n_timesteps+1), n_timesteps+1, device=device)
    else:
        raise NotImplementedError('Diffusion schedule not implemented.')

    # states = np.zeros((n_timesteps+1, n_data, 2**n_qubits), dtype=np.complex64)
    states = torch.zeros((n_timesteps+1, n_data, 2**n_qubits), device=device, dtype=torch.complex64)
    states[0] = dataset
    for t in tqdm(range(1, n_timesteps+1)):
        states[t] = model.set_diffusionData_t(t, states[0], diffusion_weights[:t], seed=t)
        # states[t] = states[t] / np.linalg.norm(states[t], axis=1, keepdims=True) # Avoid numerical errors
        states[t] = states[t] / torch.norm(states[t], dim=1, keepdim=True) # Avoid numerical errors

    dir, filename = get_path(config, type='diffusedqstates', diffusion_schedule=diffusion_schedule, n_data=n_data, n_pixels=n_pixels, n_qubits=n_qubits, n_timesteps=n_timesteps)
    np.save(dir / filename, states.detach().cpu().numpy())

    print(f"Saved diffused quantum states in {dir / filename}")

if __name__ == "__main__":
    main()