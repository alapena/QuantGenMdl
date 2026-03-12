from src.utils import get_path, find_closest_power_of_2
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms
from pathlib import Path
import numpy as np
import yaml

def main():
    config = yaml.safe_load(open('config.yaml', 'r'))
    dataset_config = config['dataset']

    dataset = get_dataset(config)

    if dataset_config["name"] != "CIRCLEY" and dataset_config["name"] != "CIRCLEY2Q":
        # In order to move to np arrays...
        batch_size = dataset_config.get('batch_size', len(dataset))
        dataset = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        dataset = next(iter(dataset))[0]

        # Generate the quantum states corresponding to the dataset
        dataset = dataset.reshape(dataset.shape[0], -1) # Reshape to [n_data, n_pixels]
        dataset = np.array(dataset, dtype=np.complex64)
        dataset = dataset / np.linalg.norm(dataset, axis=1, keepdims=True) # Ensure normalization
        
    # Fill the rest of the states with zeroes.
    n_data = dataset.shape[0]
    n_features = dataset.shape[1]
    _, n_qubits = find_closest_power_of_2(n_features, return_power=True)
    dims_to_pad = 2**n_qubits-dataset.shape[1]
    dataset = np.pad(dataset, ((0,0), (0,dims_to_pad, 0)), 'constant', constant_values=0) if dims_to_pad > 0 else dataset

    # Save the generated quantum states
    dir, filename = get_path(config, type='initialqstates.npy', n_data=n_data, n_features=n_features, n_qubits=n_qubits)
    np.save(dir / filename, dataset)

    print(f"Dataset saved in {dir / filename}")



######################################################################
#                                                                    #
#                          HELPER FUNCTIONS                          #
#                                                                    #
######################################################################


def get_dataset(config, verbose=True):
    '''
    Docstring for get_dataset. It must return a torch.utils.data.Dataset object.
    
    :param dataset_config: Description
    :param verbose: Description
    '''
    dataset_config = config["dataset"]
    # Get the transforms from the config.
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

    transform = transforms.Compose(transforms_list)

    # Download the dataset
    dir = dataset_config.get('dir', './data')
    if dataset_config['name'] == 'MNIST':
        dataset = datasets.MNIST(root=dir, train=True, download=True, transform=transform)
        maxsize = dataset_config.get('maxsize', len(dataset))
        dataset = Subset(dataset, list(range(maxsize)))
    elif dataset_config['name'] == 'MNIST0':
        # Load the MNIST dataset and filter for digit '0' only
        dataset = datasets.MNIST(root=dir, train=True, download=True, transform=transform)
        indices = [i for i, (img, label) in enumerate(dataset) if label == 0]
        maxsize = dataset_config.get('maxsize', len(dataset))
        indices = indices[:maxsize]
        dataset = Subset(dataset, indices)

    elif dataset_config['name'] == 'MNIST1':
        # Load the MNIST dataset and filter for digit '0' only
        dataset = datasets.MNIST(root=dir, train=True, download=True, transform=transform)
        indices = [i for i, (img, label) in enumerate(dataset) if label == 1]
        maxsize = dataset_config.get('maxsize', len(dataset))
        indices = indices[:maxsize]
        dataset = Subset(dataset, indices)

    elif dataset_config['name'] == 'CIRCLEY':
        size = dataset_config['maxsize']
        size = size if size is not None else 60000
        dataset = circleYGen(size, config.get('seed', None))

    elif dataset_config['name'] == 'CIRCLEY2Q':
        size = dataset_config['maxsize']
        size = size if size is not None else 60000
        n_qubits = 2
        dataset = ndim_circleYGen(size, n_qubits, config.get('seed', None))

    else:
        raise NotImplementedError(f"Dataset {dataset_config['name']} not implemented.")

    if verbose:
        print(f"Loaded dataset {dataset_config['name']} with {len(dataset)} samples.")

    return dataset # Shape [n_data, 1, n_pixels, n_pixels]

def circleYGen(N_train, seed=None):
    '''
    generate random quantum states from RY(\phi)|0>
    assume uniform distribution
    '''
    np.random.seed(seed)
    phis = np.random.uniform(0, 2*np.pi, N_train)
    states = np.vstack((np.cos(phis), np.sin(phis))).T
    return states.astype(np.complex64)

def ndim_circleYGen(N_data, n_qubits, seed=None):
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


if __name__ == "__main__":
    main()