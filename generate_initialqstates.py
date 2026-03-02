from src.utils import get_path, find_closest_power_of_2
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms
from pathlib import Path
import numpy as np
import yaml

def main():
    config = yaml.safe_load(open('config_debug.yaml', 'r'))
    dataset_config = config['dataset']

    dataset = get_dataset(dataset_config)

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
    n_pixels = dataset.shape[1]
    _, n_qubits = find_closest_power_of_2(n_pixels, return_power=True)
    dims_to_pad = 2**n_qubits-dataset.shape[1]
    dataset = np.pad(dataset, ((0,0), (0,dims_to_pad, 0)), 'constant', constant_values=0) if dims_to_pad > 0 else dataset

    # Save the generated quantum states
    dir, filename = get_path(config, type='initialqstates', n_data=n_data, n_pixels=n_pixels, n_qubits=n_qubits)
    np.save(dir / filename, dataset)

    print(f"Dataset saved in {dir / filename}")



######################################################################
#                                                                    #
#                          HELPER FUNCTIONS                          #
#                                                                    #
######################################################################


def get_dataset(dataset_config, verbose=True):
    '''
    Docstring for get_dataset. It must return a torch.utils.data.Dataset object.
    
    :param dataset_config: Description
    :param verbose: Description
    '''
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
    else:
        raise NotImplementedError(f"Dataset {dataset_config['name']} not implemented.")

    if verbose:
        print(f"Loaded dataset {dataset_config['name']} with {len(dataset)} samples.")

    return dataset # Shape [n_data, 1, n_pixels, n_pixels]

if __name__ == "__main__":
    main()