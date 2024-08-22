from soma_den_mn_model.configs import Config
from typing import Optional, List, Dict
from scipy import signal
import brian2.numpy_ as np
from brian2 import *
from brian2.units.allunits import newton

# For MacOS
import os
os.environ['CC'] = 'gcc'
os.environ['CXX'] = 'g++'

class MNPool(Config):

    def __init__(self, N: int, fs: Optional[int] = 1e5, **kwargs):
        super().__init__(**kwargs)
        self.N = N # Number of neurons in the pool
        self.fs = fs # Sampling frequency 

        # Compute properties based on number of neurons
        self.r_soma = np.linspace(*self.r_soma_range, N)
        self.l_soma = np.linspace(*self.l_soma_range, N)
        self.r_den = np.linspace(*self.r_den_range, N)
        self.l_den = np.linspace(*self.l_den_range, N)

        self.A_soma = 2 * np.pi * self.r_soma * self.l_soma
        self.A_den = 2 * np.pi * self.r_den * self.l_den

        self.R_soma = np.linspace(*self.R_soma_range, N)
        self.R_den = np.linspace(*self.R_den_range, N)

        # Conductances
        self.g_kf_area = np.linspace(*self.g_kf_range, N)
        self.g_ks_area = np.linspace(*self.g_ks_range, N)
        self.g_ca_area = np.linspace(*self.g_ca_range, N)

        aux_soma = self.R_cyt * self.l_soma / (np.pi * self.r_soma**2) 
        aux_den = self.R_cyt * self.l_den / (np.pi * self.r_den**2) 
        self.g_coupling = 2 / (aux_soma + aux_den) 

        self.g_leak_soma = 2 * np.pi * self.r_soma * self.l_soma / self.R_soma 
        self.g_leak_den = 2 * np.pi * self.r_den * self.l_den / self.R_den 

        # Rheobase current (with 1% coefficient of variation) and input resistance
        self.I_rheo = np.linspace(*self.I_rheo_range, N) + np.random.normal(0, np.diff(self.I_rheo_range)/100, N) * nA
        self.R_n = 1 / (self.g_leak_soma + (self.g_leak_den * self.g_coupling) / (self.g_leak_den + self.g_coupling)) 

        # Force properties
        self.Apeak = np.linspace(*self.Apeak_range, N)
        self.tpeak = np.linspace(*self.tpeak_range, N)

        # Synaptic properties
        self.tau_exc = 1 / (self.alpha_exc * self.Tmax + self.beta_exc)
        self.r_exc_inf = self.alpha_exc * self.Tmax * self.tau_exc
        self.tau_inh = 1 / (self.alpha_inh * self.Tmax + self.beta_inh)
        self.r_inh_inf = self.alpha_inh * self.Tmax * self.tau_inh

        # Properties for the solver
        self.g_na = self.g_na_area * self.A_soma
        self.g_kf = self.g_kf_area * self.A_soma
        self.g_ks = self.g_ks_area * self.A_soma
        self.g_ca = self.g_ca_area * self.A_den

        self.g_leak_soma = self.g_leak_soma
        self.g_leak_den = self.g_leak_den
        self.g_coupling = self.g_coupling
        
        self.C_soma = 2 * np.pi * self.r_soma * self.l_soma * self.C_mem
        self.C_den = 2 * np.pi * self.r_den * self.l_den * self.C_mem
        
        self.v_thr = self.R_n * self.I_rheo
        self.v_thr_ca = np.linspace(*self.v_thr_ca_range, self.N) + np.random.normal(0, np.diff(self.v_thr_ca_range)/100, N) * mV
        
        self.alpha_H = np.linspace(*self.alpha_H_range, self.N)
        self.beta_H = np.linspace(*self.beta_H_range, self.N)
        self.alpha_N = np.linspace(*self.alpha_N_range, self.N)
        self.beta_N = np.linspace(*self.beta_N_range, self.N)
        self.alpha_Q = np.linspace(*self.alpha_Q_range, self.N)
        self.beta_Q = np.linspace(*self.beta_Q_range, self.N)
        self.beta_P = np.linspace(*self.beta_P_range, self.N)

        # Initial conditions
        self.m0 = np.zeros(self.N)
        self.h0 = np.ones(self.N)
        self.n0 = np.zeros(self.N)
        self.q0 = np.zeros(self.N)
        self.p0 = np.zeros(self.N)
        self.t0 = (np.zeros(self.N)) * second
        self.t0_pic = (np.zeros(self.N)) * second
        self.spike_flag = np.zeros(self.N, dtype=bool)
        self.pic_flag = np.zeros(self.N, dtype=bool)

        # Synaptic properties (needed to simulate the neurotransmitter dynamics
        # they are not needed if an external synaptic input is provided)
        # self.t_exc_aux = np.zeros(self.N) * second
        # self.t_inh_aux = np.zeros(self.N) * second
        # self.r_exc_aux = np.zeros(self.N)
        # self.r_inh_aux = np.zeros(self.N)
        # self.syn_exc_flag = np.zeros(self.N, dtype=bool)
        # self.syn_inh_flag = np.zeros(self.N, dtype=bool)

    def _get_params_solver(self) -> Dict:
        '''
        Get the neuron properties for the solver
        '''

        neuron_specific_props = {
            'gamma': np.repeat(self.gamma, self.N),

            'E_leak': np.repeat(self.E_leak, self.N),
            'E_na': np.repeat(self.E_na, self.N),
            'E_k': np.repeat(self.E_k, self.N),
            'E_ca': np.repeat(self.E_ca, self.N),
            'E_exc': np.repeat(self.E_exc, self.N),

            'g_na': self.g_na,
            'g_kf': self.g_kf,
            'g_ks': self.g_ks,
            'g_ca': self.g_ca,
            'g_leak_soma': self.g_leak_soma,
            'g_leak_den': self.g_leak_den,
            'g_coupling': self.g_coupling,
            'g_exc': np.repeat(self.g_exc, self.N),

            'C_soma': self.C_soma,
            'C_den': self.C_den,

            'v_thr': self.v_thr,
            'v_thr_ca': self.v_thr_ca,

            'alpha_M': np.repeat(self.alpha_M, self.N),
            'beta_M': np.repeat(self.beta_M, self.N),
            'alpha_H': self.alpha_H,
            'beta_H': self.beta_H,
            'alpha_N': self.alpha_N,
            'beta_N': self.beta_N,
            'alpha_Q': self.alpha_Q,
            'beta_Q': self.beta_Q,
            'alpha_P': np.repeat(self.alpha_P, self.N),
            'beta_P': self.beta_P,

            'm0': self.m0,
            'h0': self.h0,
            'n0': self.n0,
            'q0': self.q0,
            'p0': self.p0,
            't0': self.t0,
            't0_pic': self.t0_pic,
            'spike_flag': self.spike_flag,
            'pic_flag': self.pic_flag,
            'pulse_width': np.repeat(self.pulse_width, self.N),
        }

        return neuron_specific_props

    def run_sim(self,
            duration: float, 
            syn_input: Optional[TimedArray] = None,
            I_inj_soma: Optional[TimedArray] = None,
            I_inj_den: Optional[TimedArray] = None,
            vars_to_monitor: Optional[List[str]] = None,
            ) -> None:
        
        '''Run the simulation of the motor neuron pool.

        Args:
            duration (float): Duration of the simulation in seconds.
            syn_input (Optional[TimedArray]): Optional synaptic input as a 
                TimedArray with shape (samples, neurons).
            I_inj_soma (Optional[TimedArray]): Optional injected current in 
                the soma as a TimedArray with shape (samples).
            I_inj_den (Optional[TimedArray]): Optional injected current in the 
                dendrite as a TimedArray with shape (samples).
            vars_to_monitor (Optional[List[str]]): Optional list of variables to 
                monitor.

        Returns:
            None

        Notes:
            - This function runs a simulation of the motor neuron pool for the
                specified duration.
            - The synaptic input, injected currents, and variables to monitor
                can be optionally provided but their samples dimension must match
                the duration of the simulation.
            - The simulation results are stored in the object for further analysis.
                Spikes are stored in spikes_mon, the rest of the variables are stored
                in state_mon.
            - The outputs are sorted according to the neuron index, in ascending 
                order based on the motor neuron size.
        '''

        prefs.codegen.target = 'cython'
        start_scope()
        defaultclock.dt = 1/self.fs

        eq_soma = '''
            dv_soma/dt = (- g_leak_soma * (v_soma - E_leak) - g_coupling * (v_soma - v_den) - I_ion) / C_soma : volt 
            '''
        eq_den = '''
            dv_den/dt = (- g_leak_den * (v_den - E_leak) - g_coupling * (v_den - v_soma) - I_pic) / C_den : volt
            '''
         
        if I_inj_soma is not None:
            eq_soma = '''
            dv_soma/dt = (- g_leak_soma * (v_soma - E_leak) - g_coupling * (v_soma - v_den) - I_ion + I_inj_soma(t)) / C_soma : volt
            '''
        if syn_input is None and I_inj_den is not None:
            eq_den = '''
            dv_den/dt = (- g_leak_den * (v_den - E_leak) - g_coupling * (v_den - v_soma) - I_pic + I_inj_den(t)) / C_den : volt 
            '''
        elif syn_input is not None and I_inj_den is None:
            eq_den = '''
            dv_den/dt = (- g_leak_den * (v_den - E_leak) - g_coupling * (v_den - v_soma) - I_pic - I_syn) / C_den : volt 
            I_syn = syn_input(t, i) * g_exc * (v_den - E_exc) : amp (constant over dt)
            '''
        elif syn_input is not None and I_inj_den is not None:
            eq_den = '''
            dv_den/dt = (- g_leak_den * (v_den - E_leak) - g_coupling * (v_den - v_soma) - I_pic - I_syn + I_inj_den(t)) / C_den : volt
            I_syn = syn_input(t, i) * g_exc * (v_den - E_exc) : amp (constant over dt)
            '''

        eqs_currs = '''
            m = ( 1 + (m0 - 1) * exp(- alpha_M * (t - t0)) ) * int(spike_flag) + m0 * exp(- beta_M * (t - t0)) * int(spike_flag == False) : 1 (constant over dt)
            h = ( h0 * exp(- beta_H * (t - t0)) ) * int(spike_flag) + ( 1 + (h0 - 1) * exp(- alpha_H * (t - t0)) ) * int(spike_flag == False) : 1 (constant over dt)
            n = ( 1 + (n0 - 1) * exp(- alpha_N * (t - t0)) ) * int(spike_flag) + n0 * exp(- beta_N * (t - t0)) * int(spike_flag == False) : 1 (constant over dt)
            q = ( 1 + (q0 - 1) * exp(- alpha_Q * (t - t0)) ) * int(spike_flag) + q0 * exp(- beta_Q * (t - t0)) * int(spike_flag == False) : 1 (constant over dt)
            p = ( 1 + (p0 - 1) * exp(- alpha_P * (t - t0_pic)) ) * int(pic_flag) + p0 * exp(- beta_P * (t - t0_pic)) * int(pic_flag == False) : 1 (constant over dt)

            I_na = g_na * m ** 3 * h * (v_soma - E_na) : amp (constant over dt)
            I_kf = g_kf * n ** 4 * (v_soma - E_k) : amp (constant over dt)
            I_ks = g_ks * q ** 2 * (v_soma - E_k) : amp (constant over dt)
            I_ion = I_na + I_kf + I_ks : amp (constant over dt)
            I_pic = gamma * g_ca * p * (v_den - E_ca) : amp (constant over dt)
        '''

        eqs_vars = '''
            gamma : 1 (constant)

            E_leak : volt (constant)
            E_na : volt (constant)
            E_k : volt (constant)
            E_ca : volt (constant)
            E_exc : volt (constant)

            g_na: siemens (constant)
            g_kf: siemens (constant)
            g_ks: siemens (constant)
            g_ca: siemens (constant)
            g_leak_soma: siemens (constant)
            g_leak_den: siemens (constant)
            g_coupling: siemens (constant)
            g_exc: siemens (constant)
            
            C_soma: farad (constant)
            C_den: farad (constant)

            v_thr: volt (constant)
            v_thr_ca: volt (constant)
            
            alpha_M: Hz (constant)
            beta_M: Hz (constant)
            alpha_H: Hz (constant)
            beta_H: Hz (constant)
            alpha_N: Hz (constant)
            beta_N: Hz (constant)
            alpha_Q: Hz (constant)
            beta_Q: Hz (constant)
            alpha_P: Hz (constant)
            beta_P: Hz (constant)

            m0: 1 
            h0: 1 
            n0: 1
            q0: 1 
            p0: 1
            t0: second 
            t0_pic: second 
            spike_flag : boolean 
            pic_flag : boolean 
            
            pulse_width: second (constant)
        '''

        eqs = eq_soma + eq_den + eqs_currs + eqs_vars

        mn_pool = NeuronGroup(self.N, eqs, 
                threshold='v_soma >= v_thr', 
                reset = '''
                spike_flag = True
                t0 = t
                m0 = m
                h0 = h
                n0 = n
                q0 = q
                ''',
                refractory = self.refractory_period,
                method='exponential_euler',
                events = {
                    'end_pw_spike': 'spike_flag and (t - t0) > pulse_width',
                    'start_pw_pic': 'v_den >= v_thr_ca and (pic_flag == False)', 
                    'end_pw_pic':  'pic_flag and (t - t0_pic) > pulse_width', 
                },
        )

        # Add the trigger for the end of the spike pulse
        mn_pool.run_on_event(
            'end_pw_spike', 
            '''
            spike_flag = False
            t0 = t
            m0 = m
            h0 = h
            n0 = n
            q0 = q
            ''')

        # Add the trigger for the start of the PIC pulse
        mn_pool.run_on_event(
            'start_pw_pic', 
            '''
            pic_flag = True
            t0_pic = t
            p0 = p
            ''')

        # Add the trigger for the end of the PIC pulse
        mn_pool.run_on_event(
            'end_pw_pic', 
            '''
            pic_flag = False
            t0_pic = t
            p0 = p
            ''')
        
        # Set specific values
        neuron_specific_props = self._get_params_solver()
        mn_pool.set_states(neuron_specific_props)

        # Choose variables to record
        spike_mon = SpikeMonitor(mn_pool)
        if vars_to_monitor is None:
            vars_to_monitor = ['v_soma']
        state_mon = StateMonitor(mn_pool, vars_to_monitor, record=True)

        # Run the simulation
        run(duration)

        # Store outputs in the object
        self.spike_mon = spike_mon
        self.state_mon = state_mon

        # Get forces
        self.timestamps = state_mon.t
        self._get_mu_force()

        print('Simulation done!')

    def get_spike_trains(self) -> Dict:
        '''Get the spike trains of the motor neuron pool
        Returns:
            spike_trains (Dict): Dictionary with the spike trains of the motor 
                neuron pool. Keys are the neuron index and values are the 
                corresponding spike times.
        '''
        return self.spike_mon.spike_trains()     

    def _get_mu_force(self, spikes_bin: Optional[np.ndarray] = None) -> None:
        '''Compute the force of the motor neuron pool based on the spikes
        Args:
            spikes_bin (Optional[np.ndarray]): Optional input of binary spikes
                with shape (neurons, samples). If None, the function will 
                generate the binary spikes based on the spike monitor.
        Returns:
            None
        '''

        if spikes_bin is None:
            spikes_bin = self._make_spikes_bin()

            T = 1/self.fs
        neurons = spikes_bin.shape[0]
        force = np.zeros_like(spikes_bin) * newton

        for neuron in range(neurons):
            # Compute filter coefficients
            b = np.array([0, self.Apeak[neuron] * T ** 2 / self.tpeak[neuron] * exp(1 - T / self.tpeak[neuron])])
            a = np.array([1, -2 * exp(-T / self.tpeak[neuron]), exp(-2 * T / self.tpeak[neuron])])
            
            # Compute force second order damped system
            force[neuron] = signal.filtfilt(b, a, spikes_bin[neuron]) * newton

            # Clip force to maximum value
            force[neuron, force[neuron] > self.Apeak[neuron]] = self.Apeak[neuron]

        self.force = force
        
    def _make_spikes_bin(self) -> np.ndarray:
        '''Transforms the spikes from the spike monitor into binary spikes
        Returns:
            spikes_bin (np.ndarray): Binary array with the spikes as 1s and 0s
                as baselines with shape (neurons, samples).
        '''

        firings = self.get_spike_trains()
        neurons = len(firings)

        spikes_bin = np.zeros((neurons, len(self.timestamps)))
        spike_count_mon = np.zeros(neurons)
        spike_count_bin = np.zeros(neurons)
        
        for neuron in range(neurons):
            spikes_bin[neuron, np.isin(self.timestamps/ms, firings[neuron]/ms)] = 1
            spike_count_mon[neuron] = len(firings[neuron])
            spike_count_bin[neuron] = np.sum(spikes_bin[neuron])

        assert np.sum(spike_count_mon - spike_count_bin) == 0, "The spike count is not the same in the spike monitor and the binary spikes"
        
        return spikes_bin

    def plot_results(self, 
            curr_to_plot: Optional[np.ndarray] = None, 
            curr_to_plot_label: Optional[str] = None,
            ) -> plt.Axes:
        '''Plot the results of the motor neuron pool simulation.'''

        fig, axs = plt.subplots(3,1,figsize=(12,10), layout='constrained', sharex=True)
        axs = np.ravel(axs)

        # Spike trains
        # ------------
        axs[0].scatter(self.spike_mon.t/second, self.spike_mon.i, marker='|', c=plt.cm.viridis(self.spike_mon.i/self.N))
        axs[0].set(xlabel='Time (s)', ylabel='Neuron index')

        # Firing frequency
        # ----------------
        axs1 = axs[1].twinx()
        axs[1].set_prop_cycle('color',[plt.cm.viridis(i) for i in np.linspace(0, 1, self.N)])
        for neuron in range(self.N):
            dict_spikes = self.spike_mon.spike_trains()
            x = dict_spikes[neuron]/second
            isi_neuron = np.diff(dict_spikes[neuron]/second)
            axs[1].plot(x[:-1], 1/isi_neuron/Hz, label=f'Neuron {neuron}')
        axs[1].set(ylabel='Firing frequency (Hz)', xlabel='Time (s)')

        if curr_to_plot is not None:
            axs1.plot(self.state_mon.t/second, curr_to_plot/nA, color='grey', linewidth=2, label=curr_to_plot_label)
        axs1.plot(self.state_mon.t/second, np.ones(len(self.state_mon.t_)) * self.I_rheo[0]/nA, '--', color='darkgrey', linewidth=2, label='I rheo first')
        axs1.plot(self.state_mon.t/second, np.ones(len(self.state_mon.t_)) * self.I_rheo[-1]/nA, '--', color='lightgrey', linewidth=2, label='I rheo last')
        axs1.set(ylabel='Current (nA)')
        axs1.legend(loc='upper right', bbox_to_anchor=(1.2,1))

        # Force
        # -----
        axs2 = axs[2].twinx()
        axs[2].set_prop_cycle('color',[plt.cm.viridis(i) for i in np.linspace(0, 1, self.N)])
        for neuron in range(self.N):
            axs[2].plot(self.state_mon.t/second, self.force[neuron]/newton)
        axs[2].set(xlabel='Time (s)', ylabel='Force (N)')

        if curr_to_plot is not None:
            axs2.plot(self.state_mon.t/second, curr_to_plot/nA, color='grey', linewidth=2, label=curr_to_plot_label)
        axs2.plot(self.state_mon.t/second, np.ones(len(self.state_mon.t_)) * self.I_rheo[0]/nA, '--', color='darkgrey', linewidth=2, label='I rheo first')
        axs2.plot(self.state_mon.t/second, np.ones(len(self.state_mon.t_)) * self.I_rheo[-1]/nA, '--', color='lightgrey', linewidth=2, label='I rheo last')
        axs2.set(ylabel='Current (nA)')
        axs2.legend(loc='upper right', bbox_to_anchor=(1.2,1))

        return axs