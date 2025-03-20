#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 11:53:53 2024

@author: jacob
"""

import numpy as np
import matplotlib.pyplot as plt
import python_files.three_level_moments as tla

# Add figure style sheet
plt.style.use('default')
plt.style.use('./python_files/figure_style.mplstyle')

# Figure filename
filename_out = "../svgs/fig9_g2_dressed_auto.svg"

#-----------------------------------------------------------------------------#
#                                FUNCTIONS                                    #
#-----------------------------------------------------------------------------#
def dressed_rates(Gamma_in, xi_in):
    """
    The two exponential rates in the derived functions.
    """
    lam_neg = -3 * Gamma_in * (xi_in ** 2)
    lam_pos = -Gamma_in * (1 - (xi_in ** 2) + (xi_in ** 4))
    
    lam_neg *= 0.5 / (1 + (xi_in ** 2))
    lam_pos *= 0.5 / (1 + (xi_in ** 2))
    
    return lam_neg, lam_pos

def calc_g2_dressed(tau_in, Gamma_in, xi_in):
    r"""
    Calculates the dressed-state correlation functions at two-photon resonance
    in the limit \Omega  --> \infty.
    """
    from numpy import exp
    
    # Calculates rates
    lam_neg, lam_pos = dressed_rates(Gamma_in, xi_in)
    
    # Central peak
    g2_0 = 1 + (0.5 * exp(lam_neg * tau_in))
    # First peak
    g2_1 = 1 - exp(lam_neg * tau_in)
    # Second peak
    g2_2 = 1 - exp(lam_neg * tau_in)
    # Third peak
    g2_3 = 1 + (0.5 * exp(lam_neg * tau_in)) - (1.5 * exp(lam_pos * tau_in))
    
    # Output
    return g2_0, g2_1, g2_2, g2_3

#-----------------------------------------------------------------------------#
#                                PARAMETERS                                   #
#-----------------------------------------------------------------------------#
# Atomic decay rate
Gamma = 1.0

# Dipole ratio
xi = 1.0

# Time step
dt = 0.001
# Max tau
tau_max = 10.0
# Tau list
tau = np.arange(0, tau_max + dt, dt)

#-----------------------------------------------------------------------------#
#                              CALCULATE THINGS                               #
#-----------------------------------------------------------------------------#
# Calculate three correlation functions
g2_0, g2_1, g2_2, g2_3 = calc_g2_dressed(tau, Gamma, xi)
    
# %%
#-----------------------------------------------------------------------------#
#                                  PLOT G2                                    #
#-----------------------------------------------------------------------------#
# Figure size in centimetres
figsize = np.array([7.4, 3.5])
figsize *= 1 / 2.54

# Create figure
plt.close('all')
fig = plt.figure(num='g2 auto dressed', figsize=figsize.tolist())
ax = plt.gca()

#--------------#
#     Plot     #
#--------------#
# Central peak
ax.plot(tau, g2_0, color='C0', ls='solid', label=r'$g^{(2)}_{0}(\tau)$')
# First peak
ax.plot(tau, g2_1, color='C1', ls='dashed', label=r'$g^{(2)}_{\pm 1}(\tau) , g^{(2)}_{\pm 2}(\tau)$')
# # Second peak
# ax.plot(tau, g2_2, color='C2', ls='dotted', label=r'$g^{(2)}_{0}(\tau)$')
# Third peak
ax.plot(tau, g2_3, color='C2', ls='dashdot', label=r'$g^{(2)}_{\pm 3}(\tau)$')

# ax.legend()

#---------------#
#     Ticks     #
#---------------#
ax.set_xticks(np.arange(0.0, 12.0, 2.0))
ax.set_xticks(np.arange(0.0, 12.0, 1.0), minor=True)

ax.set_yticks(np.arange(0.0, 1.75, 0.25))
ax.set_yticks(np.arange(0.0, 1.75, 0.125), minor=True)

#----------------#
#     Limits     #
#----------------#
ax.set_xlim(-0.1, 10.1)
ax.set_ylim(-0.05, 1.6)

#----------------#
#     Labels     #
#----------------#
# ax.set_xlabel(r'$\Gamma \tau$')
# ax.set_ylabel(r'$g^{(2)}_{\pm i}(\tau)$')

#---------------------#
#     Tick Labels     #
#---------------------#
# Turn off tick labels for plot
ax.set_xticklabels([])
ax.set_yticklabels([])

#----------------------#
#     Figure Stuff     #
#----------------------#
# fig.tight_layout()
fig.tight_layout(pad=0.2, h_pad=0.2, w_pad=0.2)
# fig.savefig(filename_out)
# fig.show()
