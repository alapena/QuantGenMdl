from src.trainers.default_trainer import QDDPMTrainer
from src.utils import set_device
import torch
import yaml

def main():
    config = yaml.safe_load(open('config.yaml', 'r'))
    print("")
    device = set_device(config.get('device', 'cpu'))
    torch.set_default_device(device)

    n_data = config['dataset']['maxsize'] # EDITABLE
    n_timesteps = config['model']['n_timesteps'] # EDITABLE
    n_qubits = int(config['dataset']['name'].split('_')[1])
    n_features = 2**n_qubits #config['dataset']['transforms']['resize']**2 # EDITABLE
    
    trainer = QDDPMTrainer(config, n_data, n_features, n_timesteps, device=device)
    trainer.train_all_timesteps()
    # trainer.train_single_timestep(t=40)


if __name__ == "__main__":
    main()
