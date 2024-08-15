'''Properties for S, FR, and FF motor neurons'''

from typing import List
from dataclasses import dataclass

from brian2 import *
from brian2.units.allunits import newton

@dataclass
class Config:

    # Common properties across all motor neuron types
    refractory_period: float = 5 * ms # Refractory period

    C_mem: float = 1 * uF / cm ** 2 # Membrane capacitance
    R_cyt: float = 70 * ohm * cm # Cytoplasmic resistance
    
    E_leak: float = 0 * mV # Leak reversal potential
    E_na: float = 120 * mV # Sodium reversal potential
    E_k: float = -10 * mV # Potassium reversal potential
    E_ca: float = 140 * mV # Calcium reversal potential

    gamma: float = 0.15 # Strength of the PIC (low=0.15, medium=0.3)

    pulse_width: float = 0.6 * ms # Width of the pulse used to model the channel dynamics
    g_na_area: float = 30 * msiemens / cm ** 2 # Sodium conductance per area
    alpha_M: float = 22 / ms
    beta_M: float = 13 / ms
    alpha_P: float = 0.008 / ms 

    # Synaptic properties
    g_exc: float = 600 * nsiemens # Excitatory synaptic conductance
    g_inh: float = 500 * nsiemens # Inhibitory synaptic conductance
    E_exc: float = 70 * mV # Excitatory synaptic reversal potential
    E_inh: float = -16 * mV # Inhibitory synaptic reversal potential

@dataclass
class S_MN_Config(Config):

    # Soma and dendrites sizes (cylindrical volume)
    r_soma_range: List[float] = [38.75, 41.25] * um # Soma radius
    l_soma_range: List[float] = [77.5, 82.5] * um # Soma length
    r_den_range: List[float] = [20.75, 31.25] * um # Dendrite radius
    l_den_range: List[float] = [5.5, 6.8] * mm # Dendrite length
 
    I_rheo_range: List[float] = [3.5, 6.5] * nA # Rheobase current
    ax_cv_range: List[float] = [44., 47.] * meter / second # Axonal conduction velocity

    # Resistances
    R_soma_range: List[float] = [1.15, 1.05] * kohm * cm ** 2 # Soma resistance
    R_den_range: List[float] = [14.4, 10.7] * kohm * cm ** 2 # Dendrite resistance

    # Soma conductivity properties
    g_kf_range: List[float] = [4, 4] * msiemens / cm ** 2 # Fast potassium conductance per area
    g_ks_range: List[float] = [16, 25] * msiemens / cm ** 2 # Slow potassium conductance per area
    
    alpha_H_range: List[float] = [0.5, 0.5] / ms 
    beta_H_range: List[float] = [4, 4] / ms 
    alpha_N_range: List[float] = [1.5, 1.5] / ms 
    beta_N_range: List[float] = [0.1, 0.1] / ms 
    alpha_Q_range: List[float] = [1.5, 1.5] / ms 
    beta_Q_range: List[float] = [0.025, 0.038] / ms 
 
    # Dendrite conductivity properties
    g_ca_range: List[float] = [0.038, 0.029] * msiemens / cm ** 2 # Calcium conductance per area
    alpha_P: float = 0.008 / ms
    beta_P_range: List[float] = [0.014, 0.016] / ms
    v_thr_ca_range: List[float] = [2.5, 3.0] * mV # Calcium threshold for PIC activation

    # Force properties
    Apeak_range: List[float] = [0.103, 0.123] * newton # Twitch amplitude
    tpeak_range: List[float] = [110., 100.] * ms # Time to peak twitch

@dataclass
class FR_MN_Config(Config):

    # Soma and dendrites sizes (cylindrical volume)
    r_soma_range: List[float] = [41.25, 43.75] * um # Soma radius
    l_soma_range: List[float] = [82.5, 87.50] * um # Soma length
    r_den_range: List[float] = [31.25, 41.75] * um # Dendrite radius
    l_den_range: List[float] = [6.8, 8.1] * mm # Dendrite length

    I_rheo_range: List[float] = [6.5, 17.5] * nA # Rheobase current
    ax_cv_range: List[float] = [47., 50.] * meter / second # Axonal conduction velocity

    # Resistances
    R_soma_range: List[float] = [1.05, 0.95] * kohm * cm ** 2 # Soma resistance
    R_den_range: List[float] = [10.7, 6.95] * kohm * cm ** 2 # Dendrite resistance

    # Soma conductivity properties
    g_kf_range: List[float] = [4, 2.25] * msiemens / cm ** 2 # Fast potassium conductance per area
    g_ks_range: List[float] = [25, 19] * msiemens / cm ** 2 # Slow potassium conductance per area
 
    alpha_H_range: List[float] = [0.5, 11.25] / ms
    beta_H_range: List[float] = [4, 13] / ms
    alpha_N_range: List[float] = [1.5, 11.75] / ms
    beta_N_range: List[float] = [0.1, 11.05] / ms
    alpha_Q_range: List[float] = [1.5, 11.75] / ms
    beta_Q_range: List[float] = [0.038, 11.025] / ms

    # Dendrite conductivity properties
    g_ca_range: List[float] = [0.029, 0.016] * msiemens / cm ** 2 # Calcium conductance per area
    beta_P_range: List[float] = [0.016, 0.019] / ms 
    v_thr_ca_range: List[float] = [3.0, 8.0] * mV # Calcium threshold for PIC activation

    # Force properties
    Apeak_range: List[float] = [0.123, 0.294] * newton # Twitch amplitude
    tpeak_range: List[float] = [73.5, 55.5] * ms # Time to peak twitch

@dataclass
class FF_MN_Config(Config):

    # Soma and dendrites sizes (cylindrical volume)
    r_soma_range: List[float] = [43.75, 56.5] * um # Soma radius
    l_soma_range: List[float] = [87.50, 113.] * um # Soma length
    r_den_range: List[float] = [41.75, 46.25] * um # Dendrite radius
    l_den_range: List[float] = [8.1, 10.6] * mm # Dendrite length

    I_rheo_range: List[float] = [17.5, 22.1] * nA # Rheobase current
    ax_cv_range: List[float] = [50., 53.] * meter / second # Axonal conduction velocity

    # Resistances
    R_soma_range: List[float] = [0.95, 0.65] * kohm * cm ** 2 # Soma resistance
    R_den_range: List[float] = [6.95, 6.05] * kohm * cm ** 2 # Dendrite resistance

    # Soma conductivity properties
    g_kf_range: List[float] = [2.25, 0.5] * msiemens / cm ** 2 # Fast potassium conductance per area
    g_ks_range: List[float] = [19, 4] * msiemens / cm ** 2 # Slow potassium conductance per area
 
    alpha_H_range: List[float] = [11.25, 22.] / ms
    beta_H_range: List[float] = [13., 22.] / ms
    alpha_N_range: List[float] = [11.75, 22.] / ms
    beta_N_range: List[float] = [11.05, 22.] / ms
    alpha_Q_range: List[float] = [11.75, 22.] / ms
    beta_Q_range: List[float] = [11.025, 22.] / ms

    # Dendrite conductivity properties
    g_ca_range: List[float] = [0.016, 0.012] * msiemens / cm ** 2 # Calcium conductance per area
    beta_P_range: List[float] = [0.019, 0.020] / ms 
    v_thr_ca_range: List[float] = [8.0, 8.5] * mV # Calcium threshold for PIC activation

    # Force properties
    Apeak_range: List[float] = [0.294, 0.491] * newton # Twitch amplitude
    tpeak_range: List[float] = [82.3, 56.9] * ms # Time to peak twitch