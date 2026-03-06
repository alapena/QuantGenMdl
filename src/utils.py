from pathlib import Path
import numpy as np
import torch

def set_device(device='cpu', verbose=True):
    if device != 'cpu' and torch.cuda.is_available():
        device = torch.device(device)
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
        filename = f"diffusedqstates_{datasetname}_schedule{kwargs['diffusion_schedule']}_N{kwargs['n_data']}_M{kwargs['n_pixels']}_n{kwargs['n_qubits']}_T{kwargs['n_timesteps']}.npy"
    
    elif type == 'wassdistforward':
        datasetname = dataset_config['name']
        dir = Path(dataset_config.get('dir', './data')) / datasetname / 'diffusedqstates'
        filename = f"wassdistforward_{datasetname}_schedule{kwargs['diffusion_schedule']}_N{kwargs['n_data']}_M{kwargs['n_pixels']}_n{kwargs['n_qubits']}_T{kwargs['n_timesteps']}.npy"

    elif type == 'wassdistbackwardtrain':
        datasetname = dataset_config['name']
        dir = Path(dataset_config.get('dir', './data')) / datasetname / f"modelresults_schedule{kwargs['diffusion_schedule']}_N{kwargs['n_data']}_M{kwargs['n_pixels']}_n{kwargs['n_qubits']}_na{kwargs['n_ancilla_qubits']}_T{kwargs['n_timesteps']}_L{kwargs['n_backward_layers']}"
        filename = f"wassdistbackwardtrain_{datasetname}_schedule{kwargs['diffusion_schedule']}_N{kwargs['n_data']}_M{kwargs['n_pixels']}_n{kwargs['n_qubits']}_na{kwargs['n_ancilla_qubits']}_T{kwargs['n_timesteps']}_L{kwargs['n_backward_layers']}.npy"

    elif type == 'wassdistbackwardtest':
        datasetname = dataset_config['name']
        dir = Path(dataset_config.get('dir', './data')) / datasetname / f"modelresults_schedule{kwargs['diffusion_schedule']}_N{kwargs['n_data']}_M{kwargs['n_pixels']}_n{kwargs['n_qubits']}_na{kwargs['n_ancilla_qubits']}_T{kwargs['n_timesteps']}_L{kwargs['n_backward_layers']}"
        filename = f"wassdistbackwardtest_{datasetname}_schedule{kwargs['diffusion_schedule']}_N{kwargs['n_data']}_M{kwargs['n_pixels']}_n{kwargs['n_qubits']}_na{kwargs['n_ancilla_qubits']}_T{kwargs['n_timesteps']}_L{kwargs['n_backward_layers']}.npy"

    elif type == 'sinkdist':
        datasetname = dataset_config['name']
        dir = Path(dataset_config.get('dir', './data')) / datasetname / f"modelresults_schedule{kwargs['diffusion_schedule']}_N{kwargs['n_data']}_M{kwargs['n_pixels']}_n{kwargs['n_qubits']}_na{kwargs['n_ancilla_qubits']}_T{kwargs['n_timesteps']}_L{kwargs['n_backward_layers']}"
        filename = f"sinkdist_{datasetname}_schedule{kwargs['diffusion_schedule']}_N{kwargs['n_data']}_M{kwargs['n_pixels']}_n{kwargs['n_qubits']}_na{kwargs['n_ancilla_qubits']}_T{kwargs['n_timesteps']}_L{kwargs['n_backward_layers']}.npy"

    elif type == 'modelparams':
        datasetname = dataset_config['name']
        dir = Path(dataset_config.get('dir', './data')) / datasetname / f"modelresults_N{kwargs['n_data']}_M{kwargs['n_pixels']}_n{kwargs['n_qubits']}_na{kwargs['n_ancilla_qubits']}_T{kwargs['n_timesteps']}_L{kwargs['n_backward_layers']}" / "bestresults"
        filename = f"bestparams_t{kwargs['t']}.npy"
    
    elif type == 'modellosshist':
        datasetname = dataset_config['name']
        dir = Path(dataset_config.get('dir', './data')) / datasetname / f"modelresults_N{kwargs['n_data']}_M{kwargs['n_pixels']}_n{kwargs['n_qubits']}_na{kwargs['n_ancilla_qubits']}_T{kwargs['n_timesteps']}_L{kwargs['n_backward_layers']}" / "bestresults"
        filename = f"bestlosshist_t{kwargs['t']}.npy"

    elif type == 'modelfinalparams':
        datasetname = dataset_config['name']
        dir = Path(dataset_config.get('dir', './data')) / datasetname / f"modelresults_N{kwargs['n_data']}_M{kwargs['n_pixels']}_n{kwargs['n_qubits']}_na{kwargs['n_ancilla_qubits']}_T{kwargs['n_timesteps']}_L{kwargs['n_backward_layers']}" / "finalresults"
        filename = f"finalparams_t{kwargs['t']}.npy"

    elif type == 'modelfinallosshist':
        datasetname = dataset_config['name']
        dir = Path(dataset_config.get('dir', './data')) / datasetname / f"modelresults_N{kwargs['n_data']}_M{kwargs['n_pixels']}_n{kwargs['n_qubits']}_na{kwargs['n_ancilla_qubits']}_T{kwargs['n_timesteps']}_L{kwargs['n_backward_layers']}" / "finalresults"
        filename = f"finallosshist_t{kwargs['t']}.npy"

    elif type == 'deprecatedmodelfinalparams':
        datasetname = dataset_config['name']
        dir = Path(dataset_config.get('dir', './data')) / datasetname / f"modelresults_N{kwargs['n_data']}_M{kwargs['n_pixels']}_n{kwargs['n_qubits']}_na{kwargs['n_ancilla_qubits']}_T{kwargs['n_timesteps']}_L{kwargs['n_backward_layers']}" / "finalresults"
        filename = f"finalparams_{datasetname}_N{kwargs['n_data']}_M{kwargs['n_pixels']}_n{kwargs['n_qubits']}_na{kwargs['n_ancilla_qubits']}_T{kwargs['n_timesteps']}_L{kwargs['n_backward_layers']}_t{kwargs['t']}.npy"

    elif type == 'deprecatedmodelfinallosshist':
        datasetname = dataset_config['name']
        dir = Path(dataset_config.get('dir', './data')) / datasetname / f"modelresults_N{kwargs['n_data']}_M{kwargs['n_pixels']}_n{kwargs['n_qubits']}_na{kwargs['n_ancilla_qubits']}_T{kwargs['n_timesteps']}_L{kwargs['n_backward_layers']}" / "finalresults"
        filename = f"finallosshist_{datasetname}_N{kwargs['n_data']}_M{kwargs['n_pixels']}_n{kwargs['n_qubits']}_na{kwargs['n_ancilla_qubits']}_T{kwargs['n_timesteps']}_L{kwargs['n_backward_layers']}_t{kwargs['t']}.npy"
    
    elif type == 'modelsingleparams':
        datasetname = dataset_config['name']
        dir = Path(dataset_config.get('dir', './data')) / datasetname / f"modelresults_N{kwargs['n_data']}_M{kwargs['n_pixels']}_n{kwargs['n_qubits']}_na{kwargs['n_ancilla_qubits']}_T{kwargs['n_timesteps']}_L{kwargs['n_backward_layers']}"
        filename = f"SINGLEbestparams_t{kwargs['t']}.npy"

    elif type == 'modelsinglelosshist':
        datasetname = dataset_config['name']
        dir = Path(dataset_config.get('dir', './data')) / datasetname / f"modelresults_N{kwargs['n_data']}_M{kwargs['n_pixels']}_n{kwargs['n_qubits']}_na{kwargs['n_ancilla_qubits']}_T{kwargs['n_timesteps']}_L{kwargs['n_backward_layers']}"
        filename = f"SINGLEbestlosshist_t{kwargs['t']}.npy"

    elif type == 'modelsinglefinalparams':
        datasetname = dataset_config['name']
        dir = Path(dataset_config.get('dir', './data')) / datasetname / f"modelresults_N{kwargs['n_data']}_M{kwargs['n_pixels']}_n{kwargs['n_qubits']}_na{kwargs['n_ancilla_qubits']}_T{kwargs['n_timesteps']}_L{kwargs['n_backward_layers']}"
        filename = f"SINGLEfinalparams_t{kwargs['t']}.npy"

    elif type == 'modelsinglefinallosshist':
        datasetname = dataset_config['name']
        dir = Path(dataset_config.get('dir', './data')) / datasetname / f"modelresults_N{kwargs['n_data']}_M{kwargs['n_pixels']}_n{kwargs['n_qubits']}_na{kwargs['n_ancilla_qubits']}_T{kwargs['n_timesteps']}_L{kwargs['n_backward_layers']}"
        filename = f"SINGLEfinallosshist_t{kwargs['t']}.npy"

    elif type == 'lossplot':
        datasetname = dataset_config['name']
        dir = Path(dataset_config.get('dir', './data')) / datasetname / f"modelresults_N{kwargs['n_data']}_M{kwargs['n_pixels']}_n{kwargs['n_qubits']}_na{kwargs['n_ancilla_qubits']}_T{kwargs['n_timesteps']}_L{kwargs['n_backward_layers']}"
        filename = f"lossplot_t{kwargs['t']}.html"

    elif type == 'config':
        datasetname = dataset_config['name']
        dir = Path(dataset_config.get('dir', './data')) / datasetname / f"modelresults_N{kwargs['n_data']}_M{kwargs['n_pixels']}_n{kwargs['n_qubits']}_na{kwargs['n_ancilla_qubits']}_T{kwargs['n_timesteps']}_L{kwargs['n_backward_layers']}"
        filename = f"config.yaml"

    elif type == 'modelcustom':
        datasetname = dataset_config['name']
        dir = Path(dataset_config.get('dir', './data')) / datasetname / f"modelresults_N{kwargs['n_data']}_M{kwargs['n_pixels']}_n{kwargs['n_qubits']}_na{kwargs['n_ancilla_qubits']}_T{kwargs['n_timesteps']}_L{kwargs['n_backward_layers']}"
        filename = f"SINGLEfinalSINGLEbestlosshist_t{kwargs['t']}.npy"

    elif type == 'optunadict':
        datasetname = dataset_config['name']
        dir = Path(dataset_config.get('dir', './data')) / datasetname / "results_optuna"
        filename = f"results.yaml"

    elif type == 'optunaparams':
        datasetname = dataset_config['name']
        dir = Path(dataset_config.get('dir', './data')) / datasetname / "results_optuna"
        filename = f"results.yaml"

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