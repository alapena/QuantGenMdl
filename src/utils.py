from pathlib import Path
import numpy as np
import torch

def set_device(device='cpu', verbose=True):
    if device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device =  torch.device('cpu')

    if verbose:
        print(f"Using device: {device}")

    return device

def get_path(config, type='qstates', **kwargs):
    '''
    Docstring for get_path
    
    :param config: 
        type: str, the type of path to get. Can be 'qstates'.
    :param type: Description
    '''
    dataset_config = config['dataset']

    if type == 'initialqstates':
        datasetname = dataset_config['name']
        dir = Path(dataset_config.get('dir', './data')) / datasetname / 'initialqstates'
        filename = f"initialqstates_{datasetname}_N{kwargs['n_data']}_M{kwargs['n_pixels']}_n{kwargs['n_qubits']}.npy"
    
    elif type == 'diffusedqstates':
        datasetname = dataset_config['name']
        dir = Path(dataset_config.get('dir', './data')) / datasetname / 'diffusedqstates'
        filename = f"diffusedqstates_{datasetname}_N{kwargs['n_data']}_M{kwargs['n_pixels']}_n{kwargs['n_qubits']}_T{kwargs['n_timesteps']}.npy"
    
    elif type == 'wassdist':
        datasetname = dataset_config['name']
        dir = Path(dataset_config.get('dir', './data')) / datasetname / 'wassdist'
        filename = f"wassdist_{datasetname}_N{kwargs['n_data']}_M{kwargs['n_pixels']}_n{kwargs['n_qubits']}_T{kwargs['n_timesteps']}.npy"

    else:
        raise NotImplementedError(f"Path type '{type}' not implemented.")
    
    dir.mkdir(parents=True, exist_ok=True)
    return dir, filename

def get_n_qubits_from_data(data):
    n_pixels = data.reshape(-1).shape[0]
    n_qubits = np.int(np.ceil(np.log2(n_pixels)))
    n_qubits = 1 if n_qubits == 0 else n_qubits
    return n_qubits

def find_closest_power_of_2(x, return_power=False):
    a = 2 if x == 1 else 1<<(x-1).bit_length()
    return a, int(np.ceil(np.log2(a)))