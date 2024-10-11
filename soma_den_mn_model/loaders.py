import pickle as pkl
from soma_den_mn_model.inputs import SynInputs
from soma_den_mn_model.pool import MNPool
from typing import List, Optional
from brian2.units.allunits import second, newton, amp, volt, hertz

def save_inputs(inputs: SynInputs, path: str) -> None:
    """Save inputs to a file.

    Parameters:
        inputs (SynInputs): Synaptic inputs to a motor neuron pool.
        path (str): Path to save the inputs.
    """
    with open(path, 'wb') as f:
        pkl.dump(inputs, f) 

def load_inputs(path: str) -> SynInputs:
    """Load inputs from a file.

    Parameters:
        path (str): Path to load the inputs.

    Returns:
        SynInputs: Synaptic inputs to a motor neuron pool.
    """
    with open(path, 'rb') as f:
        inputs = pkl.load(f)

    return inputs

def save_pool_results(pool: MNPool, path: str, state_mon_vars: Optional[List[str]] = None) -> None:
    """Save motor neuron pool results to a file. 

    Parameters:
        pool_results (MNPool): Motor neuron pool results.
        path (str): Path to save the results.
        state_mon_vars (List[str], optional): Variables to save from the state monitor.
            Voltage and current variables are supported. Defaults to None.

    Note:
        All variables are saved in SI units.
    """

    spike_dict = pool.spike_mon.spike_trains()
    firings = [spikes/second for spikes in spike_dict.values()]

    outputs = {
        'fs': pool.fs/hertz,
        'nneurons': pool.N,
        'time': pool.state_mon.t/second,
        'firings': firings,
        'force': pool.force/newton,
    }

    if state_mon_vars is not None:
        for var in state_mon_vars:
            if 'v_' in var:
                outputs[var] = getattr(pool.state_mon, var)/volt
            elif 'I_' in var:
                outputs[var] = getattr(pool.state_mon, var)/amp
            else:
                raise NotImplementedError(f'{var} is not supported, only voltages and currents.')

    with open(path, 'wb') as f:
        pkl.dump(outputs, f)

def load_pool_results(path: str) -> dict:
    """Load motor neuron pool results from a file.

    Parameters:
        path (str): Path to load the results.

    Returns:
        dict: Motor neuron pool results.

    Note:
        All variables are in SI units.
    """
    with open(path, 'rb') as f:
        pool_results = pkl.load(f)

    return pool_results