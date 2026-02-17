from src.utils import get_path, find_closest_power_of_2, set_device
from src.QDDPM_torch_angel import DiffusionModel, QDDPM, WassDistance
from tqdm import tqdm
import numpy as np
import torch
import yaml
import time

def main():
    config = yaml.safe_load(open('config.yaml', 'r'))
    device = set_device(config.get('device', 'cpu'))

    n_data = config['dataset']['maxsize'] # EDITABLE
    n_pixels = config['dataset']['transforms']['resize']**2 # EDITABLE
    n_timesteps = config['model']['n_timesteps'] # EDITABLE

    _, n_qubits = find_closest_power_of_2(n_pixels, return_power=True)
    n_backward_layers = config['model']['n_backward_layers']
    n_ancilla_qubits = config['model']['n_ancilla_qubits']
    n_epochs = config['training']['n_epochs']
    learning_rate = config['training']['learning_rate']
    seed = config['model']['seed']

    # De momento voy a usar como input los corrupted states. Dejo esto para samplear.
    # # Sample initial random distribution
    # model_diffusion = DiffusionModel(n_qubits, n_timesteps, n_data)
    # inputs_last_timestep = model_diffusion.HaarSampleGeneration(n_data, seed=seed)

    # Load diffused states
    dir, filename = get_path(config, type='diffusedqstates', n_data=n_data, n_pixels=n_pixels, n_qubits=n_qubits, n_timesteps=n_timesteps)
    states_diffused = np.load(dir / filename) # Must be numpy array

    # Initialize model
    model = QDDPM(n_qubits, n_ancilla_qubits, n_timesteps, n_backward_layers, device=device).to(device)
    model.set_diffusionSet(states_diffused) # This already converts the states to torch tensors in the device
    inputs_last_timestep = torch.from_numpy(states_diffused[0]).to(device)

    for t in range(n_timesteps, 0, -1): # From T to 1
        print(f"--- Training timestep {t} ---")

        # params_tot = np.zeros((n_timesteps, 2*(n_qubits+n_ancilla_qubits)*n_timesteps))
        params_tot = torch.zeros((n_timesteps, 2*(n_qubits+n_ancilla_qubits)*n_backward_layers), device=device)
        if t < n_timesteps:
            for tt in range(t+1, n_timesteps+1):
                dir, filename = get_path(config, type='modelparams', n_data=n_data, n_pixels=n_pixels, n_qubits=n_qubits, n_timesteps=n_timesteps, t=tt)
                placehold = torch.from_numpy(np.load(dir / filename)).to(device)
                params_tot[tt-1] = placehold
        params, loss_hist = training_timestep_t(model, t, inputs_last_timestep, params_tot, n_data, n_epochs, learning_rate)

        dir, filename = get_path(config, type='modelparams', n_data=n_data, n_pixels=n_pixels, n_qubits=n_qubits, n_timesteps=n_timesteps, t=t)
        np.save(dir / filename, params.detach().numpy())
        dir, filename = get_path(config, type='modellosshist', n_data=n_data, n_pixels=n_pixels, n_qubits=n_qubits, n_timesteps=n_timesteps, t=t)
        np.save(dir / filename, loss_hist.detach().numpy())




######################################################################
#                                                                    #
#                          HELPER FUNCTIONS                          #
#                                                                    #
######################################################################

def training_timestep_t(model, t, inputs_T, params_tot, n_data, epochs, lr):
    '''
    the trianing for the backward PQC at step t
    input_tplus1: the output from step t+1, as the role of input at step t
    Args:
    model: the QDDPM model
    t: the diffusion step
    inputs_T: the input data at step t=T
    params_tot: collection of PQC parameters before step t
    Ndata: number of samples in dataset
    epochs: the number of iterations
    '''
    input_tplus1 = model.prepareInput_t(inputs_T, params_tot, t, n_data) # prepare input
    states_diff = model.states_diff
    loss_hist = [] # record of training history

    # initialize parameters
    np.random.seed()
    params_t = torch.tensor(np.random.normal(size=2 * model.n_tot * model.L), requires_grad=True)
    # set optimizer and learning rate decay
    optimizer = torch.optim.Adam([params_t], lr=lr)
    
    # pbar = tqdm(range(epochs))
    t0 = time.time()
    for step in range(epochs):
        indices = np.random.choice(states_diff.shape[1], size=n_data, replace=False)
        true_data = states_diff[t, indices]

        output_t = model.backwardOutput_t(input_tplus1, params_t)
        loss = WassDistance(output_t, true_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_hist.append(loss) # record the current loss
        
        print(f"Epoch {step}, loss: {loss.item():.4f}, time elapsed: {time.time() - t0:.1f}s")
            # pbar.set_postfix({
            #     'loss': f"{loss.item():.4f}", 
            #     'elapsed': f"{time.time() - t0:.1f}s"
            # })

    return params_t, torch.stack(loss_hist)


if __name__ == "__main__":
    main()