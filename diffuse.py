from src.QDDPM_torch_angel import DiffusionModel
from src.utils import get_path, set_device, get_n_qubits_from_data
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
    n_data = 10 # EDITABLE
    n_pixels = 1 # EDITABLE

    dir, filename = get_path(config, type='initialqstates', n_data=n_data, n_pixels=n_pixels) # Editable
    dataset = np.load(dir / filename)

    # Initialize the model
    n_qubits, n_timesteps, n_data = get_n_qubits_from_data(dataset[0]), config['model']['n_timesteps'], dataset.shape[0]
    model = DiffusionModel(n_qubits, n_timesteps, n_data)

    # Get the diffused states
    max_diffusion_weight = model_config.get('max_diffusion_weight', 4.0)
    diffusion_weights = np.linspace(1., max_diffusion_weight, n_timesteps) # Hyperparameter that controls the 'size' of diffusion steps.

    states = np.zeros((n_timesteps+1, n_data, 2**n_qubits), dtype=np.complex64)
    states[0] = dataset
    for t in tqdm(range(1, n_timesteps+1)):
        states[t] = model.set_diffusionData_t(t, states[0], diffusion_weights[:t], seed=t)

    dir, filename = get_path(config, type='diffusedqstates', n_data=n_data, n_pixels=n_pixels, n_timesteps=n_timesteps)
    np.save(dir / filename, states)

if __name__ == "__main__":
    main()