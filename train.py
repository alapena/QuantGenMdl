from src.utils import get_path, find_closest_power_of_2, set_device
from src.QDDPM_torch_angel import DiffusionModel, QDDPM, WassDistance
from tqdm import tqdm
import numpy as np
import torch
import yaml
import time

def main():
    config = yaml.safe_load(open('config.yaml', 'r'))
    print("")
    device = set_device(config.get('device', 'cpu'))
    torch.set_default_device(device)

    n_data = config['dataset']['maxsize'] # EDITABLE
    n_pixels = config['dataset']['transforms']['resize']**2 # EDITABLE
    n_timesteps = config['model']['n_timesteps'] # EDITABLE

    _, n_qubits = find_closest_power_of_2(n_pixels, return_power=True)
    n_backward_layers = config['model']['n_backward_layers']
    n_ancilla_qubits = config['model']['n_ancilla_qubits']

    # Initialize model
    model = QDDPM(n_qubits, n_ancilla_qubits, n_timesteps, n_backward_layers, device=device).to(device)

    trainer = Trainer(model, config, n_data, n_pixels, n_timesteps, n_qubits, n_ancilla_qubits, n_backward_layers)
    trainer.train()




######################################################################
#                                                                    #
#                          HELPER FUNCTIONS                          #
#                                                                    #
######################################################################

class Trainer():
    def __init__(self, model: QDDPM, config, n_data, n_pixels, n_timesteps, n_qubits, n_ancilla_qubits, n_backward_layers):
        self.model = model
        self.config = config
        self.device = model.device

        self.n_data = n_data
        self.n_pixels = n_pixels
        self.n_timesteps = n_timesteps

        self.n_qubits = n_qubits
        self.n_ancilla_qubits = n_ancilla_qubits
        self.n_backward_layers = n_backward_layers
        self.n_epochs = self.config['training']['n_epochs']

    def train(self):
        n_data = self.n_data
        n_pixels = self.n_pixels
        n_timesteps = self.n_timesteps

        n_qubits = self.n_qubits
        n_backward_layers = self.n_backward_layers
        n_ancilla_qubits = self.n_ancilla_qubits
        learning_rate = self.config['training']['learning_rate']

        # Load diffused states
        dir, filename = get_path(self.config, type='diffusedqstates', n_data=n_data, n_pixels=n_pixels, n_qubits=n_qubits, n_timesteps=n_timesteps)
        states_diffused = np.load(dir / filename) # Must be numpy array

        self.model.set_diffusionSet(states_diffused) # This already converts the states to torch tensors in the device
        inputs_last_timestep = torch.from_numpy(states_diffused[0]).to(self.device)

        for t in range(n_timesteps, 0, -1): # From T to 1
            print(f"--- Training timestep {t} ---")
            params_tot = torch.zeros((n_timesteps, 2*(n_qubits+n_ancilla_qubits)*n_backward_layers), device=self.device)
            if t < n_timesteps:
                for tt in range(t+1, n_timesteps+1):
                    dir, filename = get_path(self.config, type='modelparams', n_data=n_data, n_pixels=n_pixels, n_qubits=n_qubits, n_timesteps=n_timesteps, t=tt)
                    params_tot[tt-1] = torch.from_numpy(np.load(dir / filename)).to(self.device)
            params, loss_hist = self.train_timestep_t(t, inputs_last_timestep, params_tot, n_data, learning_rate)

            self.save_results(params.detach().cpu(), loss_hist, t, prefix='final')

    
    def train_timestep_t(self, t, inputs_last_timestep, params_tot, n_data, lr):
        input_tplus1 = self.model.prepareInput_t(inputs_last_timestep, params_tot, t, n_data)
        states_diffused = self.model.states_diff

        # initialize parameters
        np.random.seed()
        params_t = torch.tensor(np.random.normal(size=2 * self.model.n_tot * self.model.L), device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([params_t], lr=lr)
        loss_hist = []

        t0 = time.time()
        last_save = 0 # Epoch where results were last saved
        pbar = tqdm(range(self.n_epochs))
        for epoch in pbar:
            indices = np.random.choice(states_diffused.shape[1], size=n_data, replace=False)
            true_data = states_diffused[t, indices]

            output_t = self.model.backwardOutput_t(input_tplus1, params_t)
            loss = WassDistance(output_t, true_data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_value = loss.detach().cpu()

            pbar.set_postfix({
                'ℒ (loss)': f"{loss.item():.4f}",
                '💾 (last saved)': f"{last_save}"
            })

            # Check if current step is best
            if len(loss_hist) == 0 or loss_value < min(loss_hist):
                self.save_results(params_t.detach().cpu(), loss_hist, t, verbose=False)
                last_save = epoch

            loss_hist.append(loss_value) # record the current loss
            

        return params_t, torch.stack(loss_hist)
    
    def save_results(self, params, loss_hist, t, prefix='', verbose=True):
            
            dir, filename = get_path(self.config, type='modelparams', n_data=self.n_data, n_pixels=self.n_pixels, n_qubits=self.n_qubits, n_timesteps=self.n_timesteps, t=t)
            np.save(dir / filename, params)

            if verbose:
                print(f"Saved parameters at {dir/(prefix+filename)}.")

            dir, filename = get_path(self.config, type='modellosshist', n_data=self.n_data, n_pixels=self.n_pixels, n_qubits=self.n_qubits, n_timesteps=self.n_timesteps, t=t)
            np.save(dir / filename, loss_hist)



# def training_timestep_t(model, t, inputs_T, params_tot, n_data, epochs, lr):
#     '''
#     the trianing for the backward PQC at step t
#     input_tplus1: the output from step t+1, as the role of input at step t
#     Args:
#     model: the QDDPM model
#     t: the diffusion step
#     inputs_T: the input data at step t=T
#     params_tot: collection of PQC parameters before step t
#     Ndata: number of samples in dataset
#     epochs: the number of iterations
#     '''
#     input_tplus1 = model.prepareInput_t(inputs_T, params_tot, t, n_data) # prepare input
#     states_diff = model.states_diff
#     loss_hist = [] # record of training history
#     device = model.device

#     # initialize parameters
#     np.random.seed()
#     params_t = torch.tensor(np.random.normal(size=2 * model.n_tot * model.L), device=device, requires_grad=True)
#     # set optimizer and learning rate decay
#     optimizer = torch.optim.Adam([params_t], lr=lr)
    
#     # pbar = tqdm(range(epochs))
#     t0 = time.time()
#     for step in range(epochs):
#         indices = np.random.choice(states_diff.shape[1], size=n_data, replace=False)
#         true_data = states_diff[t, indices]

#         output_t = model.backwardOutput_t(input_tplus1, params_t)
#         loss = WassDistance(output_t, true_data)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         loss_hist.append(loss) # record the current loss
        
#         print(f"Epoch {step}, loss: {loss.item():.4f}, time elapsed: {time.time() - t0:.1f}s")
#             # pbar.set_postfix({
#             #     'loss': f"{loss.item():.4f}", 
#             #     'elapsed': f"{time.time() - t0:.1f}s"
#             # })

#     return params_t, torch.stack(loss_hist)


if __name__ == "__main__":
    main()