import numpy as np

# With seed:

def ndim_circleYGen_seed(N_data, n_qubits, seed=None):
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
    
def ndim_cluster0Gen_seed(n_data, n_qubits, epsilon, seed=None):
    np.random.seed(seed)
    states0 = np.zeros((n_data, 2**n_qubits), dtype=np.complex64)
    states0[:, 0] = 1.
    states1 = np.zeros((n_data, 2**n_qubits), dtype=np.complex64)
    states1[:, 1] = 1.

    rng = np.random.default_rng(seed=seed)
    re_c = rng.normal(loc=0.0, scale=1.0, size=n_data)
    im_c = rng.normal(loc=0.0, scale=1.0, size=n_data)
    c = re_c + 1j*im_c
    c = c[:, np.newaxis]
    states = states0 + epsilon*c*states1
    states /= np.linalg.norm(states, axis=1, keepdims=True)
    return states

# With generator:

def ndim_cluster0Gen_rng(n_data, n_qubits, epsilon, rng=None):
    states0 = np.zeros((n_data, 2**n_qubits), dtype=np.complex64)
    states0[:, 0] = 1.
    states1 = np.zeros((n_data, 2**n_qubits), dtype=np.complex64)
    states1[:, 1] = 1.

    re_c = rng.normal(loc=0.0, scale=1.0, size=n_data)
    im_c = rng.normal(loc=0.0, scale=1.0, size=n_data)
    c = re_c + 1j*im_c
    c = c[:, np.newaxis]
    states = states0 + epsilon*c*states1
    states /= np.linalg.norm(states, axis=1, keepdims=True)
    return states