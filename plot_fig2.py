#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 15:32:47 2024

@author: jnga773
"""

import numpy as np
import matplotlib.pyplot as plt

# Add thesis style sheet
plt.style.use('./python_files/figure_style.mplstyle')

import python_files.three_level_moments as tla

plt.close('all')

# Figure filename
filename_out = "./images/fig2_ss_map_constant_alpha.svg"

#-----------------------------------------------------------------------------#
#                                FUNCTIONS                                    #
#-----------------------------------------------------------------------------#
def calc_effective_detuning(Omega_in, alpha_in, xi_in):
    """
    Calculates the effective two-photon resonance detuning from the approximate
    model.
    """
    # Square root bit
    sq = ((0.5 * alpha_in) ** 2) + 2 * ((0.5 * Omega_in) ** 2) * ((xi_in ** 2) - 1)
    sq = np.sqrt(sq)
    
    # Resonance shift
    Delta_eff = (-0.25 * alpha_in) - 0.5 * sq
    
    return Delta_eff

#-----------------------------------------------------------------------------#
#                                PARAMETERS                                   #
#-----------------------------------------------------------------------------#
# Atomic decay rate
Gamma = 1.0
# Drive detuning from two-photon resonance
delta = 0.0
# Atomic anharmonicity
alpha = -120.0

# Xi values
# xi_values = [0.5, 1.0, 1.5]
xi_values = [1 / np.sqrt(2), 1.0, np.sqrt(2)]

# Number of elements to plot
N = 201

# List of detuning values
delta_list = np.linspace(-20.0, 20.0, N)
# List of Omega values
Omega_list = np.linspace(0.0, 60.0, N)

#-----------------------------------------------------------------------------#
#                              CALCULATE THINGS                               #
#-----------------------------------------------------------------------------#
# Create meshgrids
X, Y = np.meshgrid(delta_list, Omega_list)

# Create empty matrices for \rho_gg, \rho_ee, and \rho_ff data
Z_gg = np.zeros((N, N, len(xi_values)))
Z_ee = np.zeros((N, N, len(xi_values)))
Z_ff = np.zeros((N, N, len(xi_values)))

# Calculate data
for xi in range(len(xi_values)):
    for i in range(len(delta_list)):
        for j in range(len(Omega_list)):
            # Parameters
            delta = delta_list[i]
            Omega = Omega_list[j]
            
            # Calculate steady states
            rho_ss = tla.steady_states(Gamma, Omega, alpha, delta, xi_values[xi])
            
            # Save to array
            Z_gg[i, j, xi] = rho_ss[0].real
            Z_ee[i, j, xi] = rho_ss[4].real
            Z_ff[i, j, xi] = rho_ss[8].real

    # Transpose data?
    Z_gg[:, :, xi] = np.transpose(Z_gg[:, :, xi])
    Z_ee[:, :, xi] = np.transpose(Z_ee[:, :, xi])
    Z_ff[:, :, xi] = np.transpose(Z_ff[:, :, xi])
    
# Calculate shifted resonance
Delta_eff = [calc_effective_detuning(Omega_list, alpha, xi_values[0]),
             calc_effective_detuning(Omega_list, alpha, xi_values[1]),
             calc_effective_detuning(Omega_list, alpha, xi_values[2])]

# %%
#-----------------------------------------------------------------------------#
##                          Plot Steady State Maps                           ##
#-----------------------------------------------------------------------------#
# Colour map
cmap = 'jet'

# Figure size in centimetres
figsize = np.array([7.4, 11])
figsize *= 1 / 2.54

# Create figure
plt.close('Full Model')
fig, ax = plt.subplots(nrows=3, ncols=3, sharey=True, sharex=True, figsize=figsize,
                       layout='compressed', num='Full Model')

# Set constrained values for figure
fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01,
                                hspace=6.0 / 72.0, wspace=6.0 / 72.0)

# Cycle through xi values
for i in range(len(xi_values)):
    #--------------#
    #     Plot     #
    #--------------#
    # Plot \rho_{gg}
    c1 = ax[i, 0].pcolormesh(X, Y, Z_gg[:, :, i], shading='auto',
                              cmap=cmap, vmin=0.0, vmax=1.0, rasterized=True)
    
    # Plot \rho_{ee}
    c2 = ax[i, 1].pcolormesh(X, Y, Z_ee[:, :, i], shading='auto',
                              cmap=cmap, vmin=0.0, vmax=1.0, rasterized=True)
    
    # Plot \rho_{ff}
    c3 = ax[i, 2].pcolormesh(X, Y, Z_ff[:, :, i], shading='auto',
                              cmap=cmap, vmin=0.0, vmax=1.0, rasterized=True)
    
    # Plot shifted resonance
    for j in range(3):
        ax[i, j].plot(Delta_eff[i], Omega_list, ls='dotted', color='k')
    
    #--------------------#
    #     Axis ticks     #
    #--------------------#
    for j in range(3):
        ax[i, j].set_xticks([-20, 0.0, 20])
        ax[i, j].set_xticks([-10.0, 10.0], minor=True)

        ax[i, j].set_yticks(np.arange(0.0, 70.0, 15.0))
        ax[i, j].set_yticks(np.arange(7.5, 70.0, 15.0), minor=True)
    
    #---------------------#
    #     Axis Limits     #
    #---------------------#
    for j in range(3):
        ax[i, j].set_xlim(-20.0, 20.0)
        ax[i, j].set_ylim(0.0, 60.0)
    
#---------------------#
#     Axis labels     #
#---------------------#
# for j in range(3):
#     ax[2, j].set_xlabel(r'$\delta / \Gamma$')
#     ax[j, 0].set_ylabel(r'$\Omega / \Gamma$')

# ax[0, 0].set_title(r'$\rho_{gg}$')
# ax[0, 1].set_title(r'$\rho_{ee}$')
# ax[0, 2].set_title(r'$\rho_{ff}$')

#--------------------#
#     Colour bar     #
#--------------------#
cbar = plt.colorbar(c2, ax=ax)

cbar.ax.tick_params()

cbar.set_ticks(np.arange(0.0, 1.1, 0.1))
cbar.set_ticks(np.arange(0.05, 1.0, 0.1), minor=True)

#---------------------#
#     Tick Labels     #
#---------------------#
# Turn off tick labels for plot
for i in range(len(xi_values)):
    for j in range(len(xi_values)):
        ax[i, j].set_xticklabels([])
        ax[i, j].set_yticklabels([])

cbar.set_ticklabels([])

#------------------------------#
#     Add Subfigure Labels     #
#------------------------------#
# # Add text
# # x_pos = -35
# # y_pos = 60
# x_pos = -40
# y_pos = 62.5
# ax[0, 0].text(x=x_pos, y=y_pos, s='(a)')
# ax[1, 0].text(x=x_pos, y=y_pos, s='(b)')
# ax[2, 0].text(x=x_pos, y=y_pos, s='(c)')

#-----------------------#
#     Figures stuff     #
#-----------------------#
# fig.tight_layout()
fig.savefig(filename_out)
fig.show()
