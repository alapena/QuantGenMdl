from src.QDDPM_torch_angel import DiffusionModel
from src.trainers.basic_trainers import QDDPMDiffuser, QDDPMGeneratorInitialqstates
from src.utils import get_path, set_device, find_closest_power_of_2
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import yaml
import re

def main(config, n_data, n_features, n_timesteps, diffusion_schedule_nickname, overwrite=False):
    # config = yaml.safe_load(open('config.yaml', 'r'))
    dataset_config = config['dataset']
    model_config = config['model']
    device = set_device(config.get('device', 'cpu'))
    torch.set_default_device(device)

    # Load the states
    # n_data = dataset_config['maxsize'] # EDITABLE
    # n_features = 4 #int(np.square(dataset_config['transforms']['resize'])) # EDITABLE
    # n_timesteps = model_config['n_timesteps']
    # diffusion_schedule_nickname = model_config['diffusion_schedule']['name'] + str(model_config['diffusion_schedule']['slope'])
    _, n_qubits = find_closest_power_of_2(n_features, return_power=True)

    dir, filename = get_path(config, type='diffusedqstates.npy', diffusion_schedule=diffusion_schedule_nickname, n_data=n_data, n_features=n_features, n_qubits=n_qubits, n_timesteps=n_timesteps)
    path = dir/filename
    if path.exists() and not overwrite:
        print("Forward diffused states found. Skipping diffusion...")
    else:
        print("Forward diffused states not found. Generating them...")

        # Check if initial states exist
        dir, filename = get_path(config, type='initialqstates.npy', n_data=n_data, n_features=n_features, n_qubits=n_qubits)
        path = dir/filename
        if not path.exists():
            # Generate initial states
            print("Initial quantum states not found. Generating them...")
            generator_initialqstates = QDDPMGeneratorInitialqstates(config)
            generator_initialqstates.generate_initialqstates()

        # Everything checked. Diffuse.
        diffuser = QDDPMDiffuser(config, n_data, n_features, n_timesteps, device=device)
        diffuser.diffuse()

if __name__ == "__main__":
    main()