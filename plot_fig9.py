#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 11:53:53 2024

@author: jacob
"""

import numpy as np
import matplotlib.pyplot as plt

# Add thesis style sheet
plt.style.use('./python_files/figure_style.mplstyle')
# plt.style.use('../paper_style_pdf_tex.mplstyle')

import three_level_moments as tla

plt.close('all')

# Figure filename
filename_out = "../../images/sect4/fig9_g2_dressed_auto.pdf"
# filename_out = "../../images/svg/fig9_g2_dressed_auto.svg"

# # Use PGF backend and LaTeX settings
# import matplotlib
# matplotlib.use("pgf")
# matplotlib.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     "text.usetex": True,
#     "pgf.rcfonts": False,
# })
# filename_out = "../../images/pgf/fig9_g2_dressed_auto.pgf"

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
    """
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
plt.close('g2 auto dressed')
fig = plt.figure(num='g2 auto dressed')
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

ax.legend()

#--------------#
#     Text     #
#--------------#
# # Text
# ax.text(x=2, y=1.3, s=r'$g^{(2)}_{0}(\tau)$')
# ax.text(x=3, y=0.7, s=r'$g^{(2)}_{\pm 1}(\tau), g^{(2)}_{\pm 2}(\tau)$')
# ax.text(x=5, y=0.3, s=r'$g^{(2)}_{\pm 3}(\tau)$')

# # Arrows
# ax.arrow(x=2.175, y=1.25, dx=0, dy=-0.1, head_starts_at_zero=False,
#          head_length=0.04, head_width=0.075, edgecolor='k', facecolor='k')
# ax.arrow(x=4.4, y=0.8, dx=0, dy=0.1, head_starts_at_zero=False,
#          head_length=0.04, head_width=0.04, edgecolor='k', facecolor='k')
# ax.arrow(x=5.75, y=0.4, dx=0, dy=0.15, head_starts_at_zero=False,
#          head_length=0.04, head_width=0.075, edgecolor='k', facecolor='k')

#----------------#
#     Labels     #
#----------------#
ax.set_xlabel(r'$\Gamma \tau$')
ax.set_ylabel(r'$g^{(2)}_{\pm i}(\tau)$')

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

#----------------------#
#     Figure Stuff     #
#----------------------#
fig.tight_layout()
fig.savefig(filename_out)
# fig.show()
