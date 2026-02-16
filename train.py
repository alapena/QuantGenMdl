from src.QDDPM_torch_angel import DiffusionModel, QDDPM, WassDistance
from src.utils import set_device
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import time
import yaml

def main():
    config = yaml.safe_load(open('config.yaml', 'r'))

    dataset = get_dataset(config['dataset'])




######################################################################
#                                                                    #
#                          HELPER FUNCTIONS                          #
#                                                                    #
######################################################################




if __name__ == "__main__":
    main()