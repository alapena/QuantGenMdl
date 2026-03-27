from pathlib import Path
import numpy as np
import torch
from typing import Dict
from torch.utils.data import Subset, DataLoader

def set_device(device='cpu', verbose=True):
    if device != 'cpu' and torch.cuda.is_available():
        device = torch.device(device)
    else:
        device =  torch.device('cpu')

    if verbose:
        print(f"Using device: {device}")

    return device

# def get_path(config: Dict, type: str = 'qstates', new_subfolder: str = None, suffix: str = None, **kwargs):
#     dataset_config = config['dataset']

#     if type == 'initialqstates.npy':
#         datasetname = dataset_config['name']
#         dir = Path(dataset_config.get('dir', './data')) / datasetname / 'initialqstates'
#         filename = f"initialqstates_{datasetname}_N{kwargs['n_data']}_M{kwargs['n_features']}_n{kwargs['n_qubits']}.npy"
    
#     elif type == 'diffusedqstates.npy':
#         datasetname = dataset_config['name']
#         dir = Path(dataset_config.get('dir', './data')) / datasetname / 'diffusedqstates'
#         filename = f"diffusedqstates_{datasetname}_schedule{kwargs['diffusion_schedule']}_N{kwargs['n_data']}_M{kwargs['n_features']}_n{kwargs['n_qubits']}_T{kwargs['n_timesteps']}.npy"
    
#     elif type == 'wassdistforward.npy':
#         datasetname = dataset_config['name']
#         dir = Path(dataset_config.get('dir', './data')) / datasetname / 'diffusedqstates'
#         filename = f"wassdistforward_{datasetname}_schedule{kwargs['diffusion_schedule']}_N{kwargs['n_data']}_M{kwargs['n_features']}_n{kwargs['n_qubits']}_T{kwargs['n_timesteps']}.npy"

#     elif type == 'wassdistbackwardtrain.npy':
#         datasetname = dataset_config['name']
#         dir = Path(dataset_config.get('dir', './data')) / datasetname / f"modelresults_schedule{kwargs['diffusion_schedule']}_N{kwargs['n_data']}_M{kwargs['n_features']}_n{kwargs['n_qubits']}_na{kwargs['n_ancilla_qubits']}_T{kwargs['n_timesteps']}_L{kwargs['n_backward_layers']}"
#         filename = f"wassdistbackwardtrain_{datasetname}_schedule{kwargs['diffusion_schedule']}_N{kwargs['n_data']}_M{kwargs['n_features']}_n{kwargs['n_qubits']}_na{kwargs['n_ancilla_qubits']}_T{kwargs['n_timesteps']}_L{kwargs['n_backward_layers']}.npy"

#     elif type == 'wassdistbackwardtest.npy':
#         datasetname = dataset_config['name']
#         dir = Path(dataset_config.get('dir', './data')) / datasetname / f"modelresults_schedule{kwargs['diffusion_schedule']}_N{kwargs['n_data']}_M{kwargs['n_features']}_n{kwargs['n_qubits']}_na{kwargs['n_ancilla_qubits']}_T{kwargs['n_timesteps']}_L{kwargs['n_backward_layers']}"
#         filename = f"wassdistbackwardtest_{datasetname}_schedule{kwargs['diffusion_schedule']}_N{kwargs['n_data']}_M{kwargs['n_features']}_n{kwargs['n_qubits']}_na{kwargs['n_ancilla_qubits']}_T{kwargs['n_timesteps']}_L{kwargs['n_backward_layers']}.npy"

#     elif type == 'sinkdistbackwardtrain.npy':
#         datasetname = dataset_config['name']
#         dir = Path(dataset_config.get('dir', './data')) / datasetname / f"modelresults_schedule{kwargs['diffusion_schedule']}_N{kwargs['n_data']}_M{kwargs['n_features']}_n{kwargs['n_qubits']}_na{kwargs['n_ancilla_qubits']}_T{kwargs['n_timesteps']}_L{kwargs['n_backward_layers']}"
#         filename = f"sinkdistbackwardtest_{datasetname}_schedule{kwargs['diffusion_schedule']}_N{kwargs['n_data']}_M{kwargs['n_features']}_n{kwargs['n_qubits']}_na{kwargs['n_ancilla_qubits']}_T{kwargs['n_timesteps']}_L{kwargs['n_backward_layers']}.npy"


#     elif type == 'config.yaml':
#         datasetname = dataset_config['name']
#         dir = Path(dataset_config.get('dir', './data')) / datasetname / f"modelresults_N{kwargs['n_data']}_M{kwargs['n_features']}_n{kwargs['n_qubits']}_na{kwargs['n_ancilla_qubits']}_T{kwargs['n_timesteps']}_L{kwargs['n_backward_layers']}"
#         filename = f"config.yaml"


#     elif type == 'bestparams.npy':
#         datasetname = dataset_config['name']
#         dir = Path(dataset_config.get('dir', './data')) / datasetname / f"modelresults_N{kwargs['n_data']}_M{kwargs['n_features']}_n{kwargs['n_qubits']}_na{kwargs['n_ancilla_qubits']}_T{kwargs['n_timesteps']}_L{kwargs['n_backward_layers']}" / "bestresults"
#         filename = f"bestparams_t{kwargs['t']}.npy"
    
#     elif type == 'bestlosshist.npy':
#         datasetname = dataset_config['name']
#         dir = Path(dataset_config.get('dir', './data')) / datasetname / f"modelresults_N{kwargs['n_data']}_M{kwargs['n_features']}_n{kwargs['n_qubits']}_na{kwargs['n_ancilla_qubits']}_T{kwargs['n_timesteps']}_L{kwargs['n_backward_layers']}" / "bestresults"
#         filename = f"bestlosshist_t{kwargs['t']}.npy"

#     elif type == 'finalparams.npy':
#         datasetname = dataset_config['name']
#         dir = Path(dataset_config.get('dir', './data')) / datasetname / f"modelresults_N{kwargs['n_data']}_M{kwargs['n_features']}_n{kwargs['n_qubits']}_na{kwargs['n_ancilla_qubits']}_T{kwargs['n_timesteps']}_L{kwargs['n_backward_layers']}" / "finalresults"
#         filename = f"finalparams_t{kwargs['t']}.npy"

#     elif type == 'finallosshist.npy':
#         datasetname = dataset_config['name']
#         dir = Path(dataset_config.get('dir', './data')) / datasetname / f"modelresults_N{kwargs['n_data']}_M{kwargs['n_features']}_n{kwargs['n_qubits']}_na{kwargs['n_ancilla_qubits']}_T{kwargs['n_timesteps']}_L{kwargs['n_backward_layers']}" / "finalresults"
#         filename = f"finallosshist_t{kwargs['t']}.npy"

#     elif type == 'lossplot.html':
#         datasetname = dataset_config['name']
#         dir = Path(dataset_config.get('dir', './data')) / datasetname / f"modelresults_N{kwargs['n_data']}_M{kwargs['n_features']}_n{kwargs['n_qubits']}_na{kwargs['n_ancilla_qubits']}_T{kwargs['n_timesteps']}_L{kwargs['n_backward_layers']}"
#         filename = f"lossplot_t{kwargs['t']}.html"

#     # elif type == 'deprecatedmodelfinalparams':
#     #     datasetname = dataset_config['name']
#     #     dir = Path(dataset_config.get('dir', './data')) / datasetname / f"modelresults_N{kwargs['n_data']}_M{kwargs['n_features']}_n{kwargs['n_qubits']}_na{kwargs['n_ancilla_qubits']}_T{kwargs['n_timesteps']}_L{kwargs['n_backward_layers']}" / "finalresults"
#     #     filename = f"finalparams_{datasetname}_N{kwargs['n_data']}_M{kwargs['n_features']}_n{kwargs['n_qubits']}_na{kwargs['n_ancilla_qubits']}_T{kwargs['n_timesteps']}_L{kwargs['n_backward_layers']}_t{kwargs['t']}.npy"

#     # elif type == 'deprecatedmodelfinallosshist':
#     #     datasetname = dataset_config['name']
#     #     dir = Path(dataset_config.get('dir', './data')) / datasetname / f"modelresults_N{kwargs['n_data']}_M{kwargs['n_features']}_n{kwargs['n_qubits']}_na{kwargs['n_ancilla_qubits']}_T{kwargs['n_timesteps']}_L{kwargs['n_backward_layers']}" / "finalresults"
#     #     filename = f"finallosshist_{datasetname}_N{kwargs['n_data']}_M{kwargs['n_features']}_n{kwargs['n_qubits']}_na{kwargs['n_ancilla_qubits']}_T{kwargs['n_timesteps']}_L{kwargs['n_backward_layers']}_t{kwargs['t']}.npy"
    
#     # elif type == 'modelsingleparams':
#     #     datasetname = dataset_config['name']
#     #     dir = Path(dataset_config.get('dir', './data')) / datasetname / f"modelresults_N{kwargs['n_data']}_M{kwargs['n_features']}_n{kwargs['n_qubits']}_na{kwargs['n_ancilla_qubits']}_T{kwargs['n_timesteps']}_L{kwargs['n_backward_layers']}"
#     #     filename = f"SINGLEbestparams_t{kwargs['t']}.npy"

#     # elif type == 'modelsinglelosshist':
#     #     datasetname = dataset_config['name']
#     #     dir = Path(dataset_config.get('dir', './data')) / datasetname / f"modelresults_N{kwargs['n_data']}_M{kwargs['n_features']}_n{kwargs['n_qubits']}_na{kwargs['n_ancilla_qubits']}_T{kwargs['n_timesteps']}_L{kwargs['n_backward_layers']}"
#     #     filename = f"SINGLEbestlosshist_t{kwargs['t']}.npy"

#     # elif type == 'modelsinglefinalparams':
#     #     datasetname = dataset_config['name']
#     #     dir = Path(dataset_config.get('dir', './data')) / datasetname / f"modelresults_N{kwargs['n_data']}_M{kwargs['n_features']}_n{kwargs['n_qubits']}_na{kwargs['n_ancilla_qubits']}_T{kwargs['n_timesteps']}_L{kwargs['n_backward_layers']}"
#     #     filename = f"SINGLEfinalparams_t{kwargs['t']}.npy"

#     # elif type == 'modelsinglefinallosshist':
#     #     datasetname = dataset_config['name']
#     #     dir = Path(dataset_config.get('dir', './data')) / datasetname / f"modelresults_N{kwargs['n_data']}_M{kwargs['n_features']}_n{kwargs['n_qubits']}_na{kwargs['n_ancilla_qubits']}_T{kwargs['n_timesteps']}_L{kwargs['n_backward_layers']}"
#     #     filename = f"SINGLEfinallosshist_t{kwargs['t']}.npy"

#     # elif type == 'modelcustom':
#     #     datasetname = dataset_config['name']
#     #     dir = Path(dataset_config.get('dir', './data')) / datasetname / f"modelresults_N{kwargs['n_data']}_M{kwargs['n_features']}_n{kwargs['n_qubits']}_na{kwargs['n_ancilla_qubits']}_T{kwargs['n_timesteps']}_L{kwargs['n_backward_layers']}"
#     #     filename = f"SINGLEfinalSINGLEbestlosshist_t{kwargs['t']}.npy"

#     # elif type == 'optunadict':
#     #     datasetname = dataset_config['name']
#     #     dir = Path(dataset_config.get('dir', './data')) / datasetname / "results_optuna"
#     #     filename = f"results.yaml"

#     # elif type == 'optunaparams':
#     #     datasetname = dataset_config['name']
#     #     dir = Path(dataset_config.get('dir', './data')) / datasetname / "results_optuna"
#     #     filename = f"results.yaml"

#     else:
#         raise NotImplementedError(f"Path type {type} not implemented.")

    
#     # Append a new folder at the end of the directory if specified:
#     if new_subfolder is not None:
#         parts = list(dir.parts)
#         parts.insert(-1, new_subfolder)
#         dir = Path(*parts)

#     # Append a suffix before the extension of the filename
#     if suffix is not None:
#         parts = filename.split('.')
#         filename = f"{parts[0]}_{suffix}.{parts[1]}"
    
#     dir.mkdir(parents=True, exist_ok=True)
#     return dir, filename


def get_diffusion_weights(config, device='cpu'):
    # Un diffusion schedule viene determinado por 1. El dataset type, 2. El dataset como tal (n_data, n_features y n_qubits).
    diffusion_schedule_name = config["model"]["diffusion_schedule"]["name"]
    n_timesteps = config["model"]["n_timesteps"]

    if diffusion_schedule_name == "linear":
        slope = config["model"]["diffusion_schedule"]["slope"]
        diffusion_weights = 1/(n_timesteps+1) * torch.linspace(1., slope*torch.tensor(n_timesteps+1), n_timesteps+1, device=device)

    elif diffusion_schedule_name == "square":
        slope = config["model"]["diffusion_schedule"]["slope"]
        x = torch.linspace(0., n_timesteps+1, steps=n_timesteps+1, device=device)
        diffusion_weights = slope*torch.pow(x, 2)
    

    elif diffusion_schedule_name == "custom_sq":
        slope = config["model"]["diffusion_schedule"]["slope"]
        vrescale = config["model"]["diffusion_schedule"]["vrescale"]
        hrescale = config["model"]["diffusion_schedule"]["hrescale"]

        x = torch.linspace(0., n_timesteps+1, steps=n_timesteps+1, device=device)
        diffusion_weights = slope*torch.pow(x, 2)
    else:
        raise NotImplementedError(f'Diffusion schedule {diffusion_schedule_name} not implemented.')

    return diffusion_weights


def get_path(config: Dict, type: str, new_subfolder: str = None, suffix: str = None, **kwargs):
    dataset_config = config['dataset']
    datasetname = dataset_config['name']
    sharedir = False
    
    if type == 'initialqstates.npy':
        dir = Path(dataset_config.get('dir', './data')) / datasetname / 'initialqstates'
        filename = f"initialqstates_{datasetname}_N{kwargs['n_data']}_M{kwargs['n_features']}_n{kwargs['n_qubits']}.npy"
    
    elif type == 'diffusedqstates.npy':
        dir = Path(dataset_config.get('dir', './data')) / datasetname / 'diffusedqstates'
        filename = f"diffusedqstates_{datasetname}_schedule{kwargs['diffusion_schedule_nickname']}_N{kwargs['n_data']}_M{kwargs['n_features']}_n{kwargs['n_qubits']}_T{kwargs['n_timesteps']}.npy"
    
    elif type == 'wassdistforward.npy':
        dir = Path(dataset_config.get('dir', './data')) / datasetname / 'diffusedqstates'
        filename = f"wassdistforward_{datasetname}_schedule{kwargs['diffusion_schedule_nickname']}_N{kwargs['n_data']}_M{kwargs['n_features']}_n{kwargs['n_qubits']}_T{kwargs['n_timesteps']}.npy"

    else:
        modeldir = Path(dataset_config.get('dir', './data')) / datasetname / f"modelresults_{kwargs['diffusion_schedule_nickname']}_N{kwargs['n_data']}_M{kwargs['n_features']}_n{kwargs['n_qubits']}_nza{kwargs['n_zero_ancilla_qubits']}_nha{kwargs['n_haar_ancilla_qubits']}_T{kwargs['n_timesteps']}_L{kwargs['n_backward_layers']}"
        sharedir = True

    if sharedir:
        if type == 'wassdistbackwardtrain.npy':
            dir = modeldir
            filename = f"wassdistbackwardtrain.npy"

        elif type == 'wassdistbackwardtest.npy':
            dir = Path(dataset_config.get('dir', './data')) / datasetname / f"modelresults_schedule{kwargs['diffusion_schedule']}_N{kwargs['n_data']}_M{kwargs['n_features']}_n{kwargs['n_qubits']}_na{kwargs['n_ancilla_qubits']}_T{kwargs['n_timesteps']}_L{kwargs['n_backward_layers']}"
            filename = f"wassdistbackwardtest.npy"

        elif type == 'sinkdistbackwardtrain.npy':
            dir = Path(dataset_config.get('dir', './data')) / datasetname / f"modelresults_schedule{kwargs['diffusion_schedule']}_N{kwargs['n_data']}_M{kwargs['n_features']}_n{kwargs['n_qubits']}_na{kwargs['n_ancilla_qubits']}_T{kwargs['n_timesteps']}_L{kwargs['n_backward_layers']}"
            filename = f"sinkdistbackwardtest.npy"

        elif type == 'bestparams.npy':
            dir = modeldir / "bestresults"
            filename = f"bestparams_t{kwargs['t']}.npy"
        
        elif type == 'bestlosshist.npy':
            dir = modeldir / "bestresults"
            filename = f"bestlosshist_t{kwargs['t']}.npy"

        elif type == 'finalparams.npy':
            dir = modeldir / "finalresults"
            filename = f"finalparams_t{kwargs['t']}.npy"

        elif type == 'finallosshist.npy':
            dir = modeldir / "finalresults"
            filename = f"finallosshist_t{kwargs['t']}.npy"

        elif type == 'lossplot.html':
            dir = modeldir
            filename = f"lossplot_t{kwargs['t']}.html"
            
        elif type == 'config.yaml':
            dir = modeldir
            filename = f"config.yaml"

        elif type == 'tensorboard_logs':
            dir = modeldir / "tensorboard_logs"
            filename = ""

        else:
            raise NotImplementedError(f"Path type {type} not implemented.")

    
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


def get_dataset(config, verbose=True):
    def _apply_torchvision_transforms(dataset_config):
        from torchvision import transforms
        transforms_config = dataset_config.get('transforms', None)
        if transforms_config is None:
            print("No transforms specified in config. Using default transforms.")
            transforms_list = [transforms.ToTensor()]
        else:
            # Build the composed transforms based on the config
            transforms_list = []
            if 'resize' in transforms_config:
                resize = transforms_config['resize']
                transforms_list.append(transforms.Resize(resize))
            transforms_list.append(transforms.ToTensor())
        return transforms.Compose(transforms_list)

    def _ndim_circleYGen(N_data, n_qubits, seed=None):
        # Generate a circular state in each qubit. Then tensor product them.
        np.random.seed(seed)
        phis = np.random.uniform(0, 2*np.pi, (N_data, n_qubits))
        cos = np.cos(phis) # [N_data, n_qubits]
        sin = np.sin(phis)
        components = np.stack((cos, sin), axis=-1) # [N_data, n_qubits, 2]

        res = components[:, 0, :] # [N_data, 2] we selected first qubit
        for i in range(1, n_qubits):
            res = (res[..., None] * components[:, i, None, :]).reshape(N_data, -1)

        return res.astype(np.complex64)


    dataset_config = config["dataset"]
    name = dataset_config["name"]
    parts = name.split('_')
    type = parts[0]
    if type == 'MNIST':
        from torchvision import datasets
        transform = _apply_torchvision_transforms(dataset_config)
        dir = dataset_config.get('dir', './data')
        digit_str = parts[1] # ! CURRENTLY WE ONLY SUPPORT 1 DIGIT AT ONCE (e.g. '0' or '1', but not '01').
        print(f"Digit: {digit_str}")

        dataset = datasets.MNIST(root=dir, train=True, download=True, transform=transform)
        indices = [i for i, (img, label) in enumerate(dataset) if label == int(digit_str)]
        maxsize = dataset_config.get('maxsize', len(dataset))
        indices = indices[:maxsize]
        dataset = Subset(dataset, indices)

        # Cochinada para convertir a np arrays
        batch_size = dataset_config.get('batch_size', len(dataset))
        print(f"Batch size: {batch_size}")
        dataset = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        dataset = next(iter(dataset))[0]

        dataset = dataset.reshape(dataset.shape[0], -1) # Reshape to [n_data, n_pixels]
        dataset = np.array(dataset, dtype=np.complex64)
        dataset = dataset / np.linalg.norm(dataset, axis=1, keepdims=True) # Ensure normalization

    elif type == 'CIRCLEY':
        size = dataset_config['maxsize']
        size = size if size is not None else 60000
        n_qubits = parts[1]
        dataset = _ndim_circleYGen(size, int(n_qubits), config.get('seed', None))

    else:
        raise NotImplementedError(f"Dataset type {type} not implemented.")

    if verbose:
        print(f"Loaded dataset {name} with {len(dataset)} samples.")
    return dataset


def get_diffusion_schedule_nickname(config):
    diffusion_schedule_config = config["model"]["diffusion_schedule"]
    kwargs = {k: diffusion_schedule_config[k] for k in diffusion_schedule_config if k != 'name'}
    name = diffusion_schedule_config['name']
    nickname = name
    for k, v in kwargs.items():
        nickname += f"-{k}{v}"
    return nickname


def get_n_qubits_from_data(data):
    n_pixels = data.reshape(-1).shape[0]
    n_qubits = np.int(np.ceil(np.log2(n_pixels)))
    n_qubits = 1 if n_qubits == 0 else n_qubits
    return n_qubits

def find_closest_power_of_2(x, return_power=False):
    a = 2 if x == 1 else 1<<(x-1).bit_length()
    return a, int(np.ceil(np.log2(a)))