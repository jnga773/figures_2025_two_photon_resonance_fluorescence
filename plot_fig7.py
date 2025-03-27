# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 21:13:37 2023

@author: Jacob
"""

import numpy as np
import matplotlib.pyplot as plt
import python_files.three_level_moments as tla

# Add figure style sheet
plt.style.use('default')
plt.style.use('./python_files/figure_style.mplstyle')

# Figure filename
filename_out = "./svgs/fig7_atom_g2_low_drive.svg"

#-----------------------------------------------------------------------------#
#                                FUNCTIONS                                    #
#-----------------------------------------------------------------------------#
def qutip_operators_3LA(Gamma_in, Omega_in, alpha_in, delta_in, xi_in, approximation=False):
    """
    Defines the atomic states, operators, and Hamiltonian in QuTiP.
    """
    from qutip import basis
    from numpy import sqrt
    
    #---------------#
    #     Basis     #
    #---------------#
    g, e, f = [basis(3, 0), basis(3, 1), basis(3, 2)]
    
    # Atomic lowering operator
    Sm = (g * e.dag()) + xi_in * (e * f.dag())
    # Atomic raising operator
    Sp = Sm.dag()
    
    #---------------------#
    #     Hamiltonian     #
    #---------------------#
    if approximation is False:
        # Hamiltonian: Full model
        H_out = -((0.5 * alpha_in) + delta_in) * (e * e.dag()) 
        H_out += (-2 * delta_in) * (f * f.dag())
        H_out += (0.5 * Omega_in) * (Sm + Sp)
        
        # Collapse operators
        c_ops_out = [sqrt(Gamma_in) * Sm]
        
    else:
        # Effective two-photon driving
        Omega_eff = xi_in * ((0.5 * Omega_in) ** 2) / ((0.5 * alpha_in) + delta_in)
        Omega_eff *= 2
        
        # Stark shifts
        dg = ((0.5 * Omega_in) ** 2) / ((0.5 * alpha_in) + delta_in)
        df = ((0.5 * xi_in * Omega_in) ** 2) / ((0.5 * alpha_in) + delta_in)
        
        # Hamiltonian: Approximate model
        H_out = dg * (g * g.dag()) + ((-2 * delta_in) + df) * (f * f.dag())
        H_out += (0.5 * Omega_eff) * ((g * f.dag()) + (f * g.dag()))
        
        # Collapse operators
        # c_ops_out = [sqrt(Gamma_in) * Sm]
        c_ops_out = [sqrt(Gamma_in) * (g * e.dag()),
                     sqrt((xi_in ** 2) * Gamma_in) * (e * f.dag())]        
    
    #----------------#
    #     Output     #
    #----------------#    
    return H_out, Sm, Sp, c_ops_out
    
def calc_atomic_g2(tau_in, Gamma_in, Omega_in, alpha_in, delta_in, xi_in, approximation=False):
    """
    Calculates the second-order correlation function.
    """
    from qutip import steadystate, expect, correlation_3op_1t
    
    #---------------#
    #     Input     #
    #---------------#
    # Hamiltonian and operators
    H, Sm, Sp, c_ops = qutip_operators_3LA(Gamma_in, Omega_in, alpha_in, delta_in, xi_in, approximation)
    
    # Calculate steady states
    rho_ss = steadystate(H, c_ops)
    # Calculate normalisation denominator
    SpSm_ss = expect(Sp * Sm, rho_ss)
    
    #--------------------------#
    #     Calculate Things     #
    #--------------------------#
    # Calculate g2
    G2 = correlation_3op_1t(H, rho_ss, tau_in, c_ops,
                            a_op=Sp, b_op=Sp * Sm, c_op=Sm)
    
    # Normalise
    g2_out = G2.real / (SpSm_ss.real ** 2)
    
    return g2_out

#-----------------------------------------------------------------------------#
#                                PARAMETERS                                   #
#-----------------------------------------------------------------------------#
# Atomic decay rate
Gamma = 1.0
# Drive detuning from two-photon resonance
delta = 0.0
# Atomic anharmonicity
alpha = -120.0
# Dipole ratio
xi = 1.0

# Time step
dt = 0.001
# Max tau
tau_max = 10.0
# Tau list
tau = np.arange(0, tau_max + dt, dt)

# List of Omega values
Omega_list = [0.1, 0.3, 0.6, 1.0]

#-----------------------------------------------------------------------------#
#                              CALCULATE THINGS                               #
#-----------------------------------------------------------------------------#
g2_list = []

# Colour list
colour_list = []
linestyle = []

for i in range(len(Omega_list)):
    # Calculate g2 from moment equations
    g2 = tla.g2_calc(tau, Gamma, Omega_list[i], alpha, delta, xi)
    
    # Append
    g2_list.append(g2)
    colour_list.append('C{}'.format(i))

# %%
#-----------------------------------------------------------------------------#
#                                  PLOT G2                                    #
#-----------------------------------------------------------------------------#
# Figure size in centimetres
figsize = np.array([7.4, 3.5])
figsize *= 1 / 2.54

# Create figure
plt.close('all')
fig = plt.figure(num='3LA g2', figsize=figsize.tolist())
ax = plt.gca()

#--------------#
#     Plot     #
#--------------#
for i in range(len(Omega_list)):
    ax.plot(tau, g2_list[i], color='C{}'.format(i), ls='solid',
            label=r'$\Omega = {} \Gamma$'.format(Omega_list[i]))
    
ax.legend(fontsize=7)

#---------------#
#     Ticks     #
#---------------#
ax.set_xticks(np.arange(0.0, 6.0, 1.0))
ax.set_xticks(np.arange(0.0, 6.0, 0.5), minor=True)

y_ticks = np.arange(0.0, 1.6e4, 0.2e4)
ax.set_yticks(y_ticks)
ax.set_yticks(y_ticks + 1e3, minor=True)

#----------------#
#     Labels     #
#----------------#
# ax.set_xlabel(r'$\Gamma \tau$')
# ax.set_ylabel(r'$g^{(2)}_{ss}(\tau)$')

#---------------------#
#     Tick Labels     #
#---------------------#
# Turn off tick labels for plot
ax.set_xticklabels([])
ax.set_yticklabels([])

#----------------#
#     Limits     #
#----------------#
ax.set_xlim(-0.1, 5.1)
# ax.set_xlim(-0, 10)
ax.set_ylim(-0.025 * 1.4e4, 1.025 * 1.4e4)

#----------------------#
#     Figure Stuff     #
#----------------------#
# fig.tight_layout()
fig.tight_layout(pad=0.2, h_pad=0.2, w_pad=0.2)
# fig.savefig(filename_out)
# fig.show()
