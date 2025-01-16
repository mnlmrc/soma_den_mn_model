from typing import Tuple, Optional
import brian2.numpy_ as np
from brian2 import *
from scipy import signal
from scipy.fft import fft, fftfreq


class SynInputs:
    ''' Class to generate synaptic inputs to a motor neuron pool. The output
    corresponds to the dynamics of the neurotransmitter (unitless) to modulate
    the corresponding input current.
    '''

    def __init__(self,
                 ninputs: int,
                 nneurons: int,
                 duration: float,
                 fs: int,
                 mean_in: float,
                 std_in: float,
                 signal_type: str = 'noise',  # Add signal_type parameter
                 sin_freq: float = 1.0,  # Frequency of the sinusoidal input
                 sin_amplitude: float = 1.0  # Amplitude of the sinusoidal input
                 ):

        self.signal_type = signal_type  # Store signal type
        self.sin_freq = sin_freq
        self.sin_amplitude = sin_amplitude

        # Basic props
        self.ninputs = ninputs
        self.nneurons = nneurons
        self.duration = duration
        self.fs = fs

        # Percentage of inputs based on Farina and Negro (2015) - Common synaptic 
        # input to motor neurons, motor unit synchronization, and force control
        self.per_common_in = 0.37
        self.per_common_noise = 0.13
        self.per_indep_noise = 0.5

        self.bw_common_in = 5 * Hz
        self.bw_common_noise = 10 * Hz
        self.bw_indep_noise = 50 * Hz

        #  Set mean and std of input currents to ensure firings
        self.mean_in = mean_in
        self.std_in = std_in

        self._gen_all_inputs()

    def _gen_all_inputs(self) -> None:
        '''Generate all inputs including common inputs, common noise, and
        independent noise.

        This method generates the synaptic weights, common inputs, common
        noise, and independent noise for the model. It then computes the 
        power spectral density (PSD) of the inputs and scales the components 
        based on the different bandwidths. Finally, it computes the final 
        currents for each neuron based on the generated inputs.
        '''

        # Generate synaptic weights
        self.syn_weights = self._gen_syn_weights(self.nneurons, self.ninputs)

        # Generate inputs
        common_inputs = self._gen_input(self.ninputs, self.duration, self.bw_common_in / Hz, self.fs / Hz,
                                        signal_type=self.signal_type)
        common_noise = self._gen_input(self.ninputs, self.duration, self.bw_common_noise / Hz, self.fs / Hz)
        indep_noise = self._gen_input(self.nneurons, self.duration, self.bw_indep_noise / Hz, self.fs / Hz)

        # Compute PSD to get power
        psd_common_inputs, _ = self._get_psd(common_inputs, self.fs / Hz)
        psd_commmon_noise, _ = self._get_psd(common_noise, self.fs / Hz)
        psd_indep_noise, _ = self._get_psd(indep_noise, self.fs / Hz)

        power_common_inputs = np.sum(psd_common_inputs, axis=1)
        power_common_noise = np.sum(psd_commmon_noise, axis=1)
        power_indep_noise = np.sum(psd_indep_noise, axis=1)

        # Scale components due to the different bandwidths
        scaler_indep_noise = np.sqrt(power_common_inputs.mean() / power_indep_noise)
        scaler_common_noise = np.sqrt(power_common_inputs / power_common_noise)

        #  Store inputs
        self.common_inputs = common_inputs
        self.common_noise = common_noise * scaler_common_noise[:, None]
        self.indep_noise = indep_noise * scaler_indep_noise[:, None]

        # Compute final currents
        ci_term = self.per_common_in * self.syn_weights @ self.common_inputs
        cn_term = self.per_common_noise * self.syn_weights @ self.common_noise
        in_term = self.per_indep_noise * self.indep_noise

        self.input_per_neuron = self.mean_in + (ci_term + cn_term + in_term) * self.std_in

    def _gen_input(self, dim: int, duration: float, bw: float, fs: int,
                   signal_type: Optional[str] = 'noise') -> np.ndarray:
        """
        Generate input signals with specified properties.

        Parameters:
            dim (int): The dimension of the input signals.
            duration (float): The duration of the input signals in seconds.
            bw (float): The cutoff frequency for the low-pass filter (used only for noise).
            fs (int): The sampling rate of the input signals.
            signal_type (Optional[str]): Type of signal to generate ('noise' or 'sinusoidal').

        Returns:
            np.ndarray: The generated input signals with shape (dim, samples).
        """
        samples = int(duration * fs)
        time = np.arange(0, duration, 1 / fs)

        if signal_type == 'sinusoidal':
            # Generate sinusoidal inputs
            freq = self.sin_freq  # Frequency in Hz (adjust as needed)
            amplitude = self.sin_amplitude  # Amplitude of the sinusoidal input
            inputs = amplitude * np.sin(2 * np.pi * freq * time)
            inputs = np.tile(inputs, (dim, 1))  # Repeat for each dimension
        else:
            # Generate noise inputs
            inputs = np.random.normal(0, 1, (dim, samples))
            sos = signal.butter(2, bw, 'low', fs=fs, output='sos')
            inputs = signal.sosfilt(sos, inputs, axis=-1)

        return inputs

    def _get_psd(self, data: np.ndarray, fs: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the Power Spectral Density (PSD) of the given data.
        Parameters:
            data (np.ndarray): The input data array with shape (dim, samples).
            fs (int): The sampling frequency.
        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the PSD and the 
                corresponding frequency values.
        """

        samples = data.shape[1]

        # Compute PSD
        yf = fft(data, axis=1)
        xf = fftfreq(samples, 1 / fs)[:samples // 2]
        psd = 2 / samples * np.abs(yf[:, :samples // 2])

        return psd, xf

    def _gen_syn_weights(self,
                         nneurons: int,
                         ninputs: int,
                         seed: Optional[int] = None
                         ) -> np.ndarray:
        """
        Generate synaptic weights for a given number of neurons and inputs
        based on (Avrillon et al. 2023) - A graph-based approach to identify
        motor neuron synergies.

        Parameters:
            nneurons (int): The number of neurons.
            ninputs (int): The number of inputs.
            seed (Optional[int]): The seed for the random number generator to
                shuffle the weights. Defaults to None.
        Returns:
            np.ndarray: An array of synaptic weights.
        Raises:
            ValueError: If the number of inputs is not supported.
        """

        if ninputs == 0:
            return np.zeros((nneurons, ninputs + 1))

        elif ninputs == 1:
            return np.ones((nneurons, ninputs))

        elif ninputs == 2:
            # Initialise weights
            weights = np.zeros((nneurons, ninputs))

            #  Define the number neurons tuned for a single or multiple inputs
            single_input_neurons = int(0.4 * nneurons)
            mixed_input_neurons = nneurons - ninputs * single_input_neurons

            #  Assign weights
            mixed_weights = np.random.rand(mixed_input_neurons)
            for i in range(ninputs):
                idx = range(single_input_neurons * i, single_input_neurons * (i + 1))
                weights[idx, i] = 1
            weights[single_input_neurons * (i + 1):, :] = np.array([mixed_weights, 1 - mixed_weights]).T

        elif ninputs < 6:
            # Initialise weights
            weights = np.zeros((nneurons, ninputs))

            #  Define the number neurons tuned for a single input (10% of the pool)
            single_input_neurons = int(0.1 * nneurons)
            # Define the number of groups (10% of the pool) tuned for multiple inputs
            mixed_input_neurons = int(0.1 * nneurons)
            mixed_input_groups = (nneurons - single_input_neurons * ninputs) // mixed_input_neurons

            #  Assign input specific weights
            for i in range(ninputs):
                idx = range(single_input_neurons * i, single_input_neurons * (i + 1))
                weights[idx, i] = 1

            #  Assign mixed input weights (randomly across inputs in groups of 10% of the pool)
            mixed_weights = np.random.rand(mixed_input_groups, ninputs)
            mixed_weights /= np.sum(mixed_weights, axis=1)[:, None]
            mixed_weights = np.repeat(mixed_weights, mixed_input_neurons, axis=0)
            weights[single_input_neurons * (i + 1):, :] = mixed_weights

        else:
            raise ValueError("Number of inputs not supported")

        #  Permute the weight distribution across the neurons in the pool
        if seed:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()
        rng.shuffle(weights, axis=0)
        return weights
