from pathlib import Path
import numpy as np
import torch
from typing import Dict

def set_device(device='cpu', verbose=True):
    if device != 'cpu' and torch.cuda.is_available():
        device = torch.device(device)
    else:
        device =  torch.device('cpu')

    if verbose:
        print(f"Using device: {device}")

    return device

def get_path(config: Dict, type: str = 'qstates', new_subfolder: str = None, suffix: str = None, **kwargs):
    dataset_config = config['dataset']

    if type == 'initialqstates.npy':
        datasetname = dataset_config['name']
        dir = Path(dataset_config.get('dir', './data')) / datasetname / 'initialqstates'
        filename = f"initialqstates_{datasetname}_N{kwargs['n_data']}_M{kwargs['n_features']}_n{kwargs['n_qubits']}.npy"
    
    elif type == 'diffusedqstates.npy':
        datasetname = dataset_config['name']
        dir = Path(dataset_config.get('dir', './data')) / datasetname / 'diffusedqstates'
        filename = f"diffusedqstates_{datasetname}_schedule{kwargs['diffusion_schedule']}_N{kwargs['n_data']}_M{kwargs['n_features']}_n{kwargs['n_qubits']}_T{kwargs['n_timesteps']}.npy"
    
    elif type == 'wassdistforward.npy':
        datasetname = dataset_config['name']
        dir = Path(dataset_config.get('dir', './data')) / datasetname / 'diffusedqstates'
        filename = f"wassdistforward_{datasetname}_schedule{kwargs['diffusion_schedule']}_N{kwargs['n_data']}_M{kwargs['n_features']}_n{kwargs['n_qubits']}_T{kwargs['n_timesteps']}.npy"

    elif type == 'wassdistbackwardtrain.npy':
        datasetname = dataset_config['name']
        dir = Path(dataset_config.get('dir', './data')) / datasetname / f"modelresults_schedule{kwargs['diffusion_schedule']}_N{kwargs['n_data']}_M{kwargs['n_features']}_n{kwargs['n_qubits']}_na{kwargs['n_ancilla_qubits']}_T{kwargs['n_timesteps']}_L{kwargs['n_backward_layers']}"
        filename = f"wassdistbackwardtrain_{datasetname}_schedule{kwargs['diffusion_schedule']}_N{kwargs['n_data']}_M{kwargs['n_features']}_n{kwargs['n_qubits']}_na{kwargs['n_ancilla_qubits']}_T{kwargs['n_timesteps']}_L{kwargs['n_backward_layers']}.npy"

    elif type == 'wassdistbackwardtest.npy':
        datasetname = dataset_config['name']
        dir = Path(dataset_config.get('dir', './data')) / datasetname / f"modelresults_schedule{kwargs['diffusion_schedule']}_N{kwargs['n_data']}_M{kwargs['n_features']}_n{kwargs['n_qubits']}_na{kwargs['n_ancilla_qubits']}_T{kwargs['n_timesteps']}_L{kwargs['n_backward_layers']}"
        filename = f"wassdistbackwardtest_{datasetname}_schedule{kwargs['diffusion_schedule']}_N{kwargs['n_data']}_M{kwargs['n_features']}_n{kwargs['n_qubits']}_na{kwargs['n_ancilla_qubits']}_T{kwargs['n_timesteps']}_L{kwargs['n_backward_layers']}.npy"

    elif type == 'sinkdistbackwardtrain.npy':
        datasetname = dataset_config['name']
        dir = Path(dataset_config.get('dir', './data')) / datasetname / f"modelresults_schedule{kwargs['diffusion_schedule']}_N{kwargs['n_data']}_M{kwargs['n_features']}_n{kwargs['n_qubits']}_na{kwargs['n_ancilla_qubits']}_T{kwargs['n_timesteps']}_L{kwargs['n_backward_layers']}"
        filename = f"sinkdistbackwardtest_{datasetname}_schedule{kwargs['diffusion_schedule']}_N{kwargs['n_data']}_M{kwargs['n_features']}_n{kwargs['n_qubits']}_na{kwargs['n_ancilla_qubits']}_T{kwargs['n_timesteps']}_L{kwargs['n_backward_layers']}.npy"


    elif type == 'config.yaml':
        datasetname = dataset_config['name']
        dir = Path(dataset_config.get('dir', './data')) / datasetname / f"modelresults_N{kwargs['n_data']}_M{kwargs['n_features']}_n{kwargs['n_qubits']}_na{kwargs['n_ancilla_qubits']}_T{kwargs['n_timesteps']}_L{kwargs['n_backward_layers']}"
        filename = f"config.yaml"


    elif type == 'bestparams.npy':
        datasetname = dataset_config['name']
        dir = Path(dataset_config.get('dir', './data')) / datasetname / f"modelresults_N{kwargs['n_data']}_M{kwargs['n_features']}_n{kwargs['n_qubits']}_na{kwargs['n_ancilla_qubits']}_T{kwargs['n_timesteps']}_L{kwargs['n_backward_layers']}" / "bestresults"
        filename = f"bestparams_t{kwargs['t']}.npy"
    
    elif type == 'bestlosshist.npy':
        datasetname = dataset_config['name']
        dir = Path(dataset_config.get('dir', './data')) / datasetname / f"modelresults_N{kwargs['n_data']}_M{kwargs['n_features']}_n{kwargs['n_qubits']}_na{kwargs['n_ancilla_qubits']}_T{kwargs['n_timesteps']}_L{kwargs['n_backward_layers']}" / "bestresults"
        filename = f"bestlosshist_t{kwargs['t']}.npy"

    elif type == 'finalparams.npy':
        datasetname = dataset_config['name']
        dir = Path(dataset_config.get('dir', './data')) / datasetname / f"modelresults_N{kwargs['n_data']}_M{kwargs['n_features']}_n{kwargs['n_qubits']}_na{kwargs['n_ancilla_qubits']}_T{kwargs['n_timesteps']}_L{kwargs['n_backward_layers']}" / "finalresults"
        filename = f"finalparams_t{kwargs['t']}.npy"

    elif type == 'finallosshist.npy':
        datasetname = dataset_config['name']
        dir = Path(dataset_config.get('dir', './data')) / datasetname / f"modelresults_N{kwargs['n_data']}_M{kwargs['n_features']}_n{kwargs['n_qubits']}_na{kwargs['n_ancilla_qubits']}_T{kwargs['n_timesteps']}_L{kwargs['n_backward_layers']}" / "finalresults"
        filename = f"finallosshist_t{kwargs['t']}.npy"

    elif type == 'lossplot.html':
        datasetname = dataset_config['name']
        dir = Path(dataset_config.get('dir', './data')) / datasetname / f"modelresults_N{kwargs['n_data']}_M{kwargs['n_features']}_n{kwargs['n_qubits']}_na{kwargs['n_ancilla_qubits']}_T{kwargs['n_timesteps']}_L{kwargs['n_backward_layers']}"
        filename = f"lossplot_t{kwargs['t']}.html"

    # elif type == 'deprecatedmodelfinalparams':
    #     datasetname = dataset_config['name']
    #     dir = Path(dataset_config.get('dir', './data')) / datasetname / f"modelresults_N{kwargs['n_data']}_M{kwargs['n_features']}_n{kwargs['n_qubits']}_na{kwargs['n_ancilla_qubits']}_T{kwargs['n_timesteps']}_L{kwargs['n_backward_layers']}" / "finalresults"
    #     filename = f"finalparams_{datasetname}_N{kwargs['n_data']}_M{kwargs['n_features']}_n{kwargs['n_qubits']}_na{kwargs['n_ancilla_qubits']}_T{kwargs['n_timesteps']}_L{kwargs['n_backward_layers']}_t{kwargs['t']}.npy"

    # elif type == 'deprecatedmodelfinallosshist':
    #     datasetname = dataset_config['name']
    #     dir = Path(dataset_config.get('dir', './data')) / datasetname / f"modelresults_N{kwargs['n_data']}_M{kwargs['n_features']}_n{kwargs['n_qubits']}_na{kwargs['n_ancilla_qubits']}_T{kwargs['n_timesteps']}_L{kwargs['n_backward_layers']}" / "finalresults"
    #     filename = f"finallosshist_{datasetname}_N{kwargs['n_data']}_M{kwargs['n_features']}_n{kwargs['n_qubits']}_na{kwargs['n_ancilla_qubits']}_T{kwargs['n_timesteps']}_L{kwargs['n_backward_layers']}_t{kwargs['t']}.npy"
    
    # elif type == 'modelsingleparams':
    #     datasetname = dataset_config['name']
    #     dir = Path(dataset_config.get('dir', './data')) / datasetname / f"modelresults_N{kwargs['n_data']}_M{kwargs['n_features']}_n{kwargs['n_qubits']}_na{kwargs['n_ancilla_qubits']}_T{kwargs['n_timesteps']}_L{kwargs['n_backward_layers']}"
    #     filename = f"SINGLEbestparams_t{kwargs['t']}.npy"

    # elif type == 'modelsinglelosshist':
    #     datasetname = dataset_config['name']
    #     dir = Path(dataset_config.get('dir', './data')) / datasetname / f"modelresults_N{kwargs['n_data']}_M{kwargs['n_features']}_n{kwargs['n_qubits']}_na{kwargs['n_ancilla_qubits']}_T{kwargs['n_timesteps']}_L{kwargs['n_backward_layers']}"
    #     filename = f"SINGLEbestlosshist_t{kwargs['t']}.npy"

    # elif type == 'modelsinglefinalparams':
    #     datasetname = dataset_config['name']
    #     dir = Path(dataset_config.get('dir', './data')) / datasetname / f"modelresults_N{kwargs['n_data']}_M{kwargs['n_features']}_n{kwargs['n_qubits']}_na{kwargs['n_ancilla_qubits']}_T{kwargs['n_timesteps']}_L{kwargs['n_backward_layers']}"
    #     filename = f"SINGLEfinalparams_t{kwargs['t']}.npy"

    # elif type == 'modelsinglefinallosshist':
    #     datasetname = dataset_config['name']
    #     dir = Path(dataset_config.get('dir', './data')) / datasetname / f"modelresults_N{kwargs['n_data']}_M{kwargs['n_features']}_n{kwargs['n_qubits']}_na{kwargs['n_ancilla_qubits']}_T{kwargs['n_timesteps']}_L{kwargs['n_backward_layers']}"
    #     filename = f"SINGLEfinallosshist_t{kwargs['t']}.npy"

    # elif type == 'modelcustom':
    #     datasetname = dataset_config['name']
    #     dir = Path(dataset_config.get('dir', './data')) / datasetname / f"modelresults_N{kwargs['n_data']}_M{kwargs['n_features']}_n{kwargs['n_qubits']}_na{kwargs['n_ancilla_qubits']}_T{kwargs['n_timesteps']}_L{kwargs['n_backward_layers']}"
    #     filename = f"SINGLEfinalSINGLEbestlosshist_t{kwargs['t']}.npy"

    # elif type == 'optunadict':
    #     datasetname = dataset_config['name']
    #     dir = Path(dataset_config.get('dir', './data')) / datasetname / "results_optuna"
    #     filename = f"results.yaml"

    # elif type == 'optunaparams':
    #     datasetname = dataset_config['name']
    #     dir = Path(dataset_config.get('dir', './data')) / datasetname / "results_optuna"
    #     filename = f"results.yaml"

    else:
        raise NotImplementedError(f"Path type '{type}' not implemented.")

    
    # Append a new folder at the end of the directory if specified:
    if new_subfolder is not None:
        parts = list(dir.parts)
        parts.insert(-1, new_subfolder)
        dir = Path(*parts)

    # Append a suffix before the extension of the filename
    if suffix is not None:
        parts = filename.split('.')
        filename = f"{parts[0]}_{suffix}.{parts[1]}"
    
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