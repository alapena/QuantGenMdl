from src.QDDPM_torch_angel import DiffusionModel
from src.utils import get_path, set_device, find_closest_power_of_2
from pathlib import Path
from tqdm import tqdm
import numpy as np
import yaml

def main():
    config = yaml.safe_load(open('config_debug.yaml', 'r'))
    dataset_config = config['dataset']
    model_config = config['model']
    device = set_device(config.get('device', 'cpu'))

    # Load the states
    n_data = dataset_config['maxsize'] # EDITABLE
    n_pixels = int(np.sqrt(dataset_config['transforms']['resize'])) # EDITABLE
    _, n_qubits = find_closest_power_of_2(n_pixels, return_power=True)

    dir, filename = get_path(config, type='initialqstates', n_data=n_data, n_pixels=n_pixels, n_qubits=n_qubits) # Editable
    dataset = np.load(dir / filename)

    # Initialize the model
    n_timesteps, n_data = config['model']['n_timesteps'], dataset.shape[0]
    model = DiffusionModel(n_qubits, n_timesteps, n_data)

    # Get the diffused states
    max_diffusion_weight = model_config.get('max_diffusion_weight', 4.0)
    diffusion_weights = np.linspace(1., max_diffusion_weight, n_timesteps) # Hyperparameter that controls the 'size' of diffusion steps.

    states = np.zeros((n_timesteps+1, n_data, 2**n_qubits), dtype=np.complex64)
    states[0] = dataset
    for t in tqdm(range(1, n_timesteps+1)):
        states[t] = model.set_diffusionData_t(t, states[0], diffusion_weights[:t], seed=t)
        states[t] = states[t] / np.linalg.norm(states[t], axis=1, keepdims=True) # Avoid numerical errors

    dir, filename = get_path(config, type='diffusedqstates', n_data=n_data, n_pixels=n_pixels, n_qubits=n_qubits, n_timesteps=n_timesteps)
    np.save(dir / filename, states)

    print(f"Saved diffused quantum stated in {dir / filename}")

if __name__ == "__main__":
    main()