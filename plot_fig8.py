# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 21:13:37 2023

@author: Jacob
"""

import numpy as np
import matplotlib.pyplot as plt

# Add thesis style sheet
plt.style.use('./python_files/figure_style.mplstyle')

import python_files.three_level_moments as tla

plt.close('all')

# Figure filename
filename_out = "../svgs/fig8_atom_g2_high_drive.svg"

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

# Driving amplitude
Omega = 40.0

# Drive detuning from two-photon resonance
delta = 0.0

# Atomic anharmonicity
alpha = -120.0

# Dipole ratio
# xi = [0.5, 1.0, 1.5]
xi = [1/np.sqrt(2), 1.0, np.sqrt(2)]

# Shifted detuning
# delta = [calc_effective_detuning(Omega, alpha, xi[0]),
#          calc_effective_detuning(Omega, alpha, xi[1]), 
#          calc_effective_detuning(Omega, alpha, xi[2])]

# Time step
dt = 0.001
# Max tau
tau_max = 10.0
# Tau list
tau = np.arange(0, tau_max + dt, dt)

#-----------------------------------------------------------------------------#
#                              CALCULATE THINGS                               #
#-----------------------------------------------------------------------------#
g2_list = []

# Colour list
colour_list = []
linestyle = []

for i in range(len(xi)):
    # Calculate g2 from moment equations
    g2 = tla.g2_calc(tau, Gamma, Omega, alpha, delta, xi[i])
    
    # Append
    g2_list.append(g2)
    colour_list.append('C{}'.format(i))

# %%
#-----------------------------------------------------------------------------#
#                                  PLOT G2                                    #
#-----------------------------------------------------------------------------#
# Figure size in centimetres
figsize = np.array([7.4, 9.1])
figsize *= 1 / 2.54

plt.close('g2 (Low and High)')
fig, ax = plt.subplots(num='g2 (Low and High)', nrows=3, ncols=1, figsize=figsize,
                       sharex=True)

#--------------#
#     Plot     #
#--------------#
for i in range(len(xi)):
    # Plot spectrum: Low drive
    ax[i].plot(tau, g2_list[i], color='k', ls='solid')


#--------------------#
#     Axis Ticks     #
#--------------------#
for i in range(len(xi)):
    ax[i].set_xticks(np.arange(0.0, 12.0, 2.0))
    ax[i].set_xticks(np.arange(1.0, 12.0, 2.0), minor=True)
    
    if i == 0:
        ax[i].set_yticks(np.arange(0.75, 2.0, 0.25))
        ax[i].set_yticks(np.arange(0.875, 2.0, 0.125), minor=True)
    # elif i == 1:
    #     ax[i].set_yticks(np.arange(0.6, 1.6, 0.1))
    #     ax[i].set_yticks(np.arange(0.65, 1.65, 0.1), minor=True)
    # elif i == 2:
    else:
        ax[i].set_yticks(np.arange(0.6, 1.6, 0.2))
        ax[i].set_yticks(np.arange(0.7, 1.6, 0.2), minor=True)

#---------------------#
#     Axis Limits     #
#---------------------#
for i in range(len(xi)):
    if i == 0:
        ax[i].set_ylim(0.7, 1.8)
    # elif i == 1:
    #     ax[i].set_ylim(0.68, 1.425)
    # elif i == 2:
    else:
        ax[i].set_ylim(0.55, 1.45)
        
ax[2].set_xlim(-0.1, 10.1)

#---------------------#
#     Axis Labels     #
#---------------------#
# # Labels
# ax[2].set_xlabel(r'$\Gamma \tau$')
# for i in range(len(xi)):
#     ax[i].set_ylabel(r'$g^{(2)}_{ss}(\tau)$')

#---------------------#
#     Tick Labels     #
#---------------------#
# Turn off tick labels for plot
for i in range(len(xi)):
    ax[i].set_xticklabels([])
    ax[i].set_yticklabels([])

#-----------------------#
#     Figure Labels     #
#-----------------------#
# xpos = 9.55
# ax[0].text(x=xpos, y=1.70, s='(a)')
# ax[1].text(x=xpos, y=1.35, s='(b)')
# ax[2].text(x=xpos, y=1.35, s='(c)')

#----------------------#
#     Figure Stuff     #
#----------------------#
# fig.tight_layout()
fig.tight_layout(pad=0.2, h_pad=0.2, w_pad=0.2)
# fig.savefig(filename_out)
# fig.show()
