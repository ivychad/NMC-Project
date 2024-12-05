import nengo
import numpy as np
import torch


def synaptic_current(V_m, g_syn, E_syn):
    """
    returns synaptic current I_syn
    based on:
    - synaptic conductance: g_syn           (based on neurotransmitter release and receptor binding)
    - synaptic reversal potential: E_syn    (-70 mV for inhibitory GABAergic and 0mV for excitatory glutamatergic synapses, GABA opens chloride, whereas glutamate opens other ionic gates (get source on this))
    - membrane potential: V_m
    source: [1]
    """
    return g_syn * (V_m - E_syn)

# def membrane_potential(C_m, I_syn, I_ion, I_ext):
#     """
#     returns delta V_m
#     based on:
#     - membrane capacitance: C_m
#     - synaptic current: I_syn
#     - ionic currents: I_ion                 (which is the sum of all ionic currents and the ionic leak)
#     - externally injected current: I_ext
#     source: [2]
#     """
#     return (I_syn + I_ion + I_ext) / C_m

def g_syn(tau_syn, g_syn, g_max, ):
    """
    returns g_syn
    based on:
    - synaptic time constant: tau_syn
    - synaptic peak conductance: g_max
    - dirac delta functions: delta(t - t_spike)
    source: [3]
    """
    return (-g_syn + g_max * delta(t - t_spike)) / tau_syn

# def g_syn_complex():
#     return

def V_m_complex(C_m, I_C, I_Na, I_K, I_Ca, I_Cl, I_leak, I_ext):
    """
    returns delta V_m
    based on:
    - membrane capacitance: C_m
    - synaptic current: I_syn
    - ionic currents: I_ion                 (which is the sum of all ionic currents and the ionic leak)
    - externally injected current: I_ext
    source: [2]
    """
    return (I_C + I_Na + I_K + I_Ca + I_Cl + I_leak + I_ext) / C_m

"""
Nernst equation     Important!!!!!!
Ohm's law
"""

def I_Na(V_m, g_Na, E_Na):
    return g_Na * (V_m - E_Na)

def I_K(V_m, g_K, E_K):
    return g_K * (V_m - E_K)

def I_Ca(V_m, g_Ca, E_Ca):
    return g_Ca * (V_m - E_Ca)

def I_Cl(V_m, g_Cl, E_Cl):
    return g_Cl * (V_m - E_Cl)

def I_leak(V_m, g_leak, E_leak):
    return g_leak * (V_m - E_leak)

def I_C(delta_V_m, C_m):
    # Capacitive currents, delta_V_m = dV_m/dt
    return C_m * delta_V_m 

"""
Source on which neurotransmitters we'll use:
https://www.merckmanuals.com/professional/neurologic-disorders/neurotransmission/neurotransmission#Major-Neurotransmitters-and-Receptors_v1031846

Release and Reuptake Formula (needs source):
d[N]/dt = R(t) - alpha[N]

Neurotransmitter Conductance Formula (needs source):
g_syn = g_max * f([N])
f([N]) = [N]^n / (K_d^n + [N]^n)
 
Synaptic Current Formula [3]:
I_syn = g_syn * (V - E_syn)
"""
class Neurotransmitter:
    def __init__(self, shape):
        self.R_const = None
        self.alpha = None
        self.concentration = torch.zeros(10)

    def simple_concentration(self):
        """
        [N] = concentration
        R(t) = release rate
        alpha[N] = reuptake rate
        """
        return R - alpha

    def simple_release(self):
        """
        R(t) = R_const
        """
        pass

    def spike_based_release(self):
        pass

    def calcium_based_release(self):
        pass

    def uptake(self, uptake):
        self.concentration -= uptake






class Glutamate:
    def

    def 
def Glutamate(V_m, E_glut = None):        # affects AMPA, NMDA, kainate receptors
    """
    Affects: 
    g_Na, g_K through AMPA
    g_Na, g_Ca, g_K through NMDA
    """
    return 

def Aspartate(V_m, E_asp = None):
    return

def GABA(V_m, E_GAB = None):             # affects GABA_A (ionotropic), GABA_B (metabotropic) receptors
    """
    Affects:
    g_Cl through GABA_A     (inhibitory)
    g_K through GABA_B, K+ channel activation
    """
    return

def Serotonin(V_m, E_ser = None):        # affects 5-HT1, 5-HT2, 5-HT3, 5-HT4, etc. receptors
    return

def Acetylcholine(V_m, E_ace = None):    # affects Nicotinic (ionotropic) and muscarinic (metabotropic) receptors
    return

def Dopamine(V_m, E_dop = None):         # affects D1-like (D1, D5) and D2-like (D2, D3, D4) receptors
    return

def Norepinephrine(V_m, E_nor = None):   # affects α1, α2, β1, β2, β3 adrenergic receptors
    return

def Endorphin(V_m, E_end = None):        # affects μ, δ, and κ opioid receptors
    return

def Enkephalin(V_m, E_enk = None):
    return

def Glycine(V_m, E_gly = None):          # 
    return


class BasicSynapse(nengo.synapses.Synapse):
    """
    This class implements the basic synapse model only incorporating g_syn.
    """
    def __init__(self, g_syn, E_syn):
        super().__init__()
        self.g_syn = g_syn  # Synaptic conductance
        self.E_syn = E_syn  # Synaptic reversal potential

    def I_syn(self, x):
        """
        Computes the current of the synapse.
        """
        return self.g_syn * (x - self.E_syn)

    def step(self, dt, x, output=None):
        """
        Runs one simulation step for the synapses.
        """
        # Here, 'x' represents the input current
        # Update membrane potential (voltage)
        output_current = self.I_syn(x)
        if output_current.shape != output.shape:
            output_current = np.broadcast_to(output_current, output.shape)
        return output_current

    def make_step(self, size_in, size_out, dt, rng, state):
        """
        Used during simulation compilation, creates the step function called by the other components of the simulation.
        """
        return self.step
    
class SimpleMultiparametricSynapse(nengo.synapses.Synapse):
    """
    This class implements the simple version of the synapse model using multiple neurotransmitters. G_syns requires an array with parameters for each individual neurotransmitter.
    """
    def __init__(self, g_syns: torch.Tensor, E_syn, Ws_Neuro: torch.Tensor):
        """
        G_syns should be a torch tensor with 10 variables and Ws_Neuro should be a torch tensor with weights representing the following neurotransmitters:
        Glutamate, Aspartate, GABA, Serotonin, Acetylcholine, Dopamine, Norepinephrine, Endorphin, Enkephalin, Glycine
        """
        super().__init__()
        assert type(g_syns) == torch.Tensor and type(Ws_Neuro) == torch.Tensor, "You did not pass a torch tensors for either g_syns and Ws_Neuro, are you certain this is what you want? \nCuz it won't work :)"
        self.g_syns = g_syns  # Synaptic conductance
        self.E_syn = E_syn  # Synaptic reversal potential
        self.Ws_Neuro = Ws_Neuro
        
    def g_syn(self):
        """
        Computes the total synaptic conductance.
        """
        Glutamate(V_m)
        Aspartate(V_m)
        GABA(V_m)
        Serotonin(V_m)
        Acetylcholine()
        Dopamine()
        Norepinephrine()
        Endorphin()
        Enkephalin()
        Glycine()
        return torch.dot(self.Ws_Neuro, self.g_syns)

    def I_syn(self, x):
        """
        Computes the current of the synapse.
        """
        return torch.dot(self.Ws_Neuro, self.g_syns) * (x - self.E_syn)

    def step(self, dt, x, output=None):
        """
        Runs one simulation step for the synapses.
        """
        # Here, 'x' represents the input current
        # Update membrane potential (voltage)
        output_current = self.I_syn(x)
        if output_current.shape != output.shape:
            output_current = np.broadcast_to(output_current, output.shape)
        return output_current

    def make_step(self, size_in, size_out, dt, rng, state):
        """
        Used during simulation compilation, creates the step function called by the other components of the simulation.
        """
        assert not self.Ws_Neuro.shape[0] == 10, "You haven't passed all neurotransmitters yet, make sure you do so before you compile the net, otherwise you're gonna compute nothing"
        return self.step 



""" 
Sources:
[1] Neural Computation Lecture 3
[2]
[3]
[4] https://my.clevelandclinic.org/health/articles/22513-neurotransmitters 
[5] https://dana.org/resources/neurotransmission-neurotransmitters/ 
[6] https://www.merckmanuals.com/professional/neurologic-disorders/neurotransmission/neurotransmission 
"""