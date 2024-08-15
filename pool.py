from configs import Config
from typing import Optional 
import brian2.numpy_ as np
from brian2 import *
from brian2.units.allunits import newton

class MNPool(Config):

    def __init__(self, N: int, fs: Optional[int] = 1e5, **kwargs):
        super().__init__(**kwargs)
        self.N = N # Number of neurons in the pool
        self.fs = fs * Hz # Sampling frequency 

        # Compute properties based on number of neurons
        self.r_soma = np.linspace(*self.r_soma_range, N)
        self.l_soma = np.linspace(*self.l_soma_range, N)
        self.r_den = np.linspace(*self.r_den_range, N)
        self.l_den = np.linspace(*self.l_den_range, N)

        self.A_soma = 2 * np.pi * self.r_soma * self.l_soma
        self.A_den = 2 * np.pi * self.r_den * self.l_den

        self.R_soma = np.linspace(*self.R_soma_range, N)
        self.R_den = np.linspace(*self.R_den_range, N)

        aux_soma = self.R_cyt * self.l_soma / (np.pi * self.r_soma**2) 
        aux_den = self.R_cyt * self.l_den / (np.pi * self.r_den**2) 
        self.g_coupling = 2 / (aux_soma + aux_den) 

        self.g_leak_soma = 2 * np.pi * self.r_soma * self.l_soma / self.R_soma 
        self.g_leak_den = 2 * np.pi * self.r_den * self.l_den / self.R_den 

        self.I_rheo = np.linspace(*self.I_rheo_range, N)
        self.R_n = 1 / (self.g_leak_soma + (self.g_leak_den * self.g_coupling) / (self.g_leak_den + self.g_coupling)) 

        self.g_kf_area = np.linspace(*self.g_kf_range, N)
        self.g_ks_area = np.linspace(*self.g_ks_range, N)
        self.g_ca_area = np.linspace(*self.g_ca_range, N)

        self.Apeak = np.linspace(*self.Apeak_range, N)
        self.tpeak = np.linspace(*self.tpeak_range, N)


    def get_params_solver(self):

        neuron_specific_props = {
            # Properties for the solver
            'g_na': self.g_na_area * self.A_soma,
            'g_kf': self.g_kf_area * self.A_soma,
            'g_ks': self.g_ks_area * self.A_soma ,
            'g_ca': self.g_ca_area * self.A_den,

            'g_leak_soma': self.g_leak_soma,
            'g_leak_den': self.g_leak_den,
            'g_coupling': self.g_coupling,
            
            'C_soma': 2 * np.pi * self.r_soma * self.l_soma * self.C_mem,
            'C_den': 2 * np.pi * self.r_den * self.l_den * self.C_mem,
            
            'v_thr': self.R_n * self.I_rheo,
            'v_thr_ca': np.linspace(*self.v_thr_ca_range, self.N),
            
            'alpha_H': np.linspace(*self.alpha_H_range, self.N),
            'beta_H': np.linspace(*self.beta_H_range, self.N),
            'alpha_N': np.linspace(*self.alpha_N_range, self.N),
            'beta_N': np.linspace(*self.beta_N_range, self.N),
            'alpha_Q': np.linspace(*self.alpha_Q_range, self.N),
            'beta_Q': np.linspace(*self.beta_Q_range, self.N),
            'beta_P': np.linspace(*self.beta_P_range, self.N),
            'beta_Q': np.linspace(*self.beta_Q_range, self.N),

            # Initial conditions
            'm0': np.zeros(self.N),
            'h0': np.ones(self.N),
            'n0': np.zeros(self.N),
            'q0': np.zeros(self.N),
            'p0': np.zeros(self.N),
            't0': (np.zeros(self.N)) * second,
            't0_pic': (np.zeros(self.N)) * second,
            'spike_flag': np.zeros(self.N, dtype=bool),
            'pic_flag': np.zeros(self.N, dtype=bool),
        }

        return neuron_specific_props
    
    def init_solver(self):

        neuron_specific_props = self.get_params_solver()

        defaultclock.dt = 1/self.fs

        eqs = '''

        dv_den/dt = (- g_leak_den * (v_den - E_leak) - g_coupling * (v_den - v_soma) - I_pic) / C_den : volt 
        dv_soma/dt = (- g_leak_soma * (v_soma - E_leak) - g_coupling * (v_soma - v_den) - I_ion + I_syn) / C_soma : volt 

        m = ( 1 + (m0 - 1) * exp(- alpha_M * (t - t0)) ) * int(spike_flag) + m0 * exp(- beta_M * (t - t0)) * int(spike_flag == False) : 1 (constant over dt)
        h = ( h0 * exp(- beta_H * (t - t0)) ) * int(spike_flag) + ( 1 + (h0 - 1) * exp(- alpha_H * (t - t0)) ) * int(spike_flag == False) : 1 (constant over dt)
        n = ( 1 + (n0 - 1) * exp(- alpha_N * (t - t0)) ) * int(spike_flag) + n0 * exp(- beta_N * (t - t0)) * int(spike_flag == False) : 1 (constant over dt)
        q = ( 1 + (q0 - 1) * exp(- alpha_Q * (t - t0)) ) * int(spike_flag) + q0 * exp(- beta_Q * (t - t0)) * int(spike_flag == False) : 1 (constant over dt)
        p = ( 1 + (p0 - 1) * exp(- alpha_P * (t - t0_pic)) ) * int(pic_flag) + p0 * exp(- beta_P * (t - t0_pic)) * int(pic_flag == False) : 1 (constant over dt)

        I_ion = g_na * m ** 3 * h * (v_soma - E_na) + g_kf * n ** 4 * (v_soma - E_k) + g_ks * q ** 2 * (v_soma - E_k) : amp 
        I_pic = gamma * g_ca * p * (v_den - E_ca) : amp 

        I_syn = triangular_input(t, duration) : amp 
        I_rheo : amp (constant)

        g_na: siemens
        g_kf: siemens
        g_ks: siemens
        g_ca: siemens
        g_leak_soma: siemens
        g_leak_den: siemens
        g_coupling: siemens
        C_soma: farad
        C_den: farad
        v_thr: volt
        v_thr_ca: volt
        beta_P: Hz
        beta_Q: Hz
        Apeak: newton
        tpeak: second

        m0: 1 
        h0: 1 
        n0: 1
        q0: 1 
        p0: 1
        t0: second 
        t0_pic: second 
        spike_flag : boolean 
        pic_flag : boolean 

        '''

        self.pool = NeuronGroup(self.N, eqs, 
                threshold='v_soma >= v_thr', 
                reset = '''
                spike_flag = True
                t0 = t
                m0 = m
                h0 = h
                n0 = n
                q0 = q
                ''',
                refractory = 'refractory_period',
                method='exponential_euler',
                events = {
                    'end_pw_spike': 'spike_flag and (t - t0) > pulse_width',
                    'start_pw_pic': 'v_den >= v_thr_ca and (pic_flag == False)', 
                    'end_pw_pic': 'pic_flag and (t - t0_pic) > pulse_width', 
                },
        )

        # Add the trigger for the end of pulse
        self.pool.run_on_event(
            'end_pw_spike', 
            '''
            spike_flag = False
            t0 = t
            m0 = m
            h0 = h
            n0 = n
            q0 = q
            ''')

        self.pool.run_on_event(
            'start_pw_pic', 
            '''
            pic_flag = True
            t0_pic = t
            p0 = p
            ''')

        self.pool.run_on_event(
            'end_pw_pic', 
            '''
            pic_flag = False
            t0_pic = t
            p0 = p
            ''')

        self.spike_mon = SpikeMonitor(self.pool)
        self.state_mon = StateMonitor(self.pool, ['v_den', 'v_soma', 'I_syn', 'I_pic', 'I_ion', 'm', 'n', 'h', 'q', 'p', 'spike_flag', 'pic_flag'], record=True)
        self.pool.set_states(neuron_specific_props)

    def run_solver(self):
        pass
    
    def get_spike_trains(self):
        return self.spike_mon.spike_trains()        

    def make_spikes_bin(self):
        firings = self.get_spike_trains(self)
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

    def get_mu_force(self, spikes_bin = None):

        if spikes_bin is None:
            spikes_bin = self.make_spikes_bin()

        T = 1/self.fs
        neurons = spikes_bin.shape[0]

        # Pad timestamps and spikes with two zeros
        timestamps_pad = np.concatenate([np.zeros(2), self.timestamps])
        spikes_pad = np.concatenate([np.zeros((neurons, 2)), spikes_bin], axis=1)
        self.force = np.zeros_like(spikes_pad) * newton

        # Get first spike
        first_spike, last_spike = np.nonzero(spikes_pad.sum(0))[0][[0, -1]]
        scaler = self.Apeak * T ** 2  /self.tpeak * exp(1 - T / self.tpeak)
        spike_term = scaler[:, None] * spikes_pad / second

        for i in range(first_spike, len(timestamps_pad)):
            self.force[:, i] = 2 * exp(- T / self.tpeak) * self.force[:, i - 1] - exp(- 2 * T / self.tpeak) * self.force[:, i - 2] + spike_term[:, i-1]
            if i > last_spike and self.force[:, i].sum() == 0:
                break
            
        self.force = self.force[:, 2:]

    def plot_results(self):

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

        axs1.plot(self.state_mon.t/second, self.state_mon.I_syn[neuron]/nA, color='grey', linewidth=2, label='I syn')
        axs1.plot(self.state_mon.t/second, np.ones(len(self.state_mon.t_)) * self.I_rheo[0]/nA, '--', color='lightcoral', linewidth=2, label='I rheo first')
        axs1.plot(self.state_mon.t/second, np.ones(len(self.state_mon.t_)) * self.I_rheo[-1]/nA, '--', color='coral', linewidth=2, label='I rheo last')
        axs1.set(ylabel='Current (nA)')
        axs1.legend(loc='upper right', bbox_to_anchor=(1.2,1))

        # Force
        # -----
        axs2 = axs[2].twinx()
        axs[2].set_prop_cycle('color',[plt.cm.viridis(i) for i in np.linspace(0, 1, self.N)])
        for neuron in range(self.N):
            axs[2].plot(self.state_mon.t/second, self.force[neuron]/newton)
        axs[2].set(xlabel='Time (s)', ylabel='Force (N)')

        axs2.plot(self.state_mon.t/second, self.state_mon.I_syn[neuron]/nA, color='grey', linewidth=2, label='I syn')
        axs2.plot(self.state_mon.t/second, np.ones(len(self.state_mon.t_)) * self.I_rheo[0]/nA, '--', color='lightcoral', linewidth=2, label='I rheo first')
        axs2.plot(self.state_mon.t/second, np.ones(len(self.state_mon.t_)) * self.I_rheo[-1]/nA, '--', color='coral', linewidth=2, label='I rheo last')
        axs2.set(ylabel='Current (nA)')
        axs2.legend(loc='upper right', bbox_to_anchor=(1.2,1))

        return axs