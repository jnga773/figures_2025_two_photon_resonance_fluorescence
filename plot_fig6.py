#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 19:43:32 2024

@author: jacob
"""

import numpy as np
import matplotlib.pyplot as plt
from python_files.jacobs_functions import spectrum, shifted_colour_map, stretch_boundary_norm

# Add figure style sheet
plt.style.use('default')
plt.style.use('./python_files/figure_style.mplstyle')

# Xi directory
xi_directory = ["xi_1_over_root_2", "xi_1", "xi_root_2"]

# Figure filename
filename_out = './svgs/fig6{}_spectrum_scan_delta.svg'
    
#-----------------------------------------------------------------------------#
#                                  FUNCTIONS                                  #
#-----------------------------------------------------------------------------#
def read_data(xi_in):
    """
    Reads the data my g
    """

    # Read some parameters
    filename_parameters = "./fig6_data/data_files/scan_{}/g1_parameters.txt".format(xi_in[0])

    gamma, alpha, delta, xi, dt, tau_max = \
    np.genfromtxt(filename_parameters, delimiter="=", usecols=1,
                max_rows=6)

    # Time list
    tau = np.arange(0, tau_max + dt, dt)

    # Pull halfwidth values
    deltas = np.genfromtxt(filename_parameters, delimiter="=", usecols=1,
                           skip_header=8, dtype='float')
    
    # Empty list to save spectrum data to
    spec_out = []

    # Cycle through "xi_values" and read data
    for xi_directory in xi_in:
        # Data filenames
        filename_data_r = "./fig6_data/data_files/scan_{}/g1_corr_real.txt".format(xi_directory)
        filename_data_i = "./fig6_data/data_files/scan_{}/g1_corr_imag.txt".format(xi_directory)

        # Read data
        g1_scans = np.genfromtxt(filename_data_r, dtype='float') + 1j * \
                   np.genfromtxt(filename_data_i, dtype='float')

        #--------------------------------------#
        #     Calculate Fourier Transforms     #
        #--------------------------------------#
        # Create empty array (NOT MATRIX)
        spectrum_scans = np.zeros(shape=(len(tau)-1, len(deltas)))

        for i in range(len(deltas)):
            # Calculate Fourier transform
            g1 = g1_scans[:, i]
            spec, wlist = spectrum(tau, g1)

            # Append to big list
            spectrum_scans[:, i] = spec

        # Add to spec_out
        spec_out.append(spectrum_scans)

    # X-data (Make it bigger by one dimension)
    X_out = np.linspace(deltas.min(), deltas.max(), len(deltas)+1)
    # Y-data (Make it bigger by one dimension)
    Y_out = np.linspace(wlist.min(), wlist.max(), len(wlist)+1)

    #----------------#
    #     Output     #
    #----------------#
    return X_out, Y_out, spec_out

#-----------------------------------------------------------------------------#
#                                PARAMETERS                                   #
#-----------------------------------------------------------------------------#
# Read data
X, Y, spectrum_scans = read_data(xi_directory)

# Colour bar ticks
boundaries = [[0.0, 0.05, 0.14, 0.2, 0.38, 0.5, 1.0],
              [0.0, 0.1, 0.2, 0.38, 0.5, 0.85, 1.0],
              [0.0, 0.04, 0.05, 0.13, 0.16, 0.5, 0.85, 1.0]]

boundaries_minor = [[0.025, 0.095, 0.17, 0.27, 0.44, 0.75],
                    [0.05, 0.15, 0.27, 0.44, 0.675, 0.925],
                    [0.02, 0.045, 0.09, 0.145, 0.38, 0.675, 0.925]]

# %%
#-----------------------------------------------------------------------------#
##                              PLOT SPECTRUM                                ##
#-----------------------------------------------------------------------------#
# from matplotlib.cm import get_cmap
from matplotlib import colormaps
cmap = colormaps['inferno_r']

# Create figure
plt.close('all')

def plot_figure(i):
    # Subfigure labels
    subfigure_labels = ['a', 'b', 'c']
    
    # from matplotlib.cm import get_cmap
    from matplotlib import colormaps
    cmap = colormaps['inferno_r']
    
    # Figure size in centimetres
    # figsize = np.array([7.4, 3.5])
    figsize = np.array([6.5, 3.5])
    figsize *= 1 / 2.54
    
    # Create figure
    plt.close('all')
    fig = plt.figure(num='Spectrum Scans (delta {})'.format(i), figsize=figsize.tolist())
    ax = plt.gca()
    
    # Calculate the stretched boundary norm
    norm = stretch_boundary_norm(boundaries[i])
    
    #--------------#
    #     Plot     #
    #--------------#
    # Plot contourf
    contour_plot = ax.pcolormesh(X, Y, spectrum_scans[i], shading='auto',
                                 cmap=cmap, norm=norm, rasterized=True)
    
    #--------------------#
    #     Colour bar     #
    #--------------------#
    # Colorbar
    cbar = plt.colorbar(contour_plot, ax=ax)
    
    # Set label
    # cbar.set_label(r'$S_{\mathrm{inc}}(\omega)$ (a.u.)', rotation=90)
    
    # Set ticks
    cbar.set_ticks(boundaries[i])
    cbar.set_ticks(boundaries_minor[i], minor=True)
    cbar.ax.tick_params(direction='out', which='both')
    
    #--------------------#
    #     Axis ticks     #
    #--------------------#
    ax.set_xticks(np.arange(-20.0, 30.0, 10.0))
    ax.set_xticks(np.arange(-15.0, 30.0, 10.0), minor=True)

    ax.set_yticks(np.arange(-120, 160, 60))
    ax.set_yticks(np.arange(-90, 160, 60), minor=True)
    
    #---------------------#
    #     Axis Limits     #
    #---------------------#
    ax.set_ylim(-120, 120)
    ax.set_xlim(-20.0, 20.0)
    
    #---------------------#
    #     Axis labels     #
    #---------------------#
    # ax.set_xlabel(r'$\Omega / \Gamma$')
    # ax.set_ylabel(r'$\left( \omega - \omega_{d} \right) / \Gamma$')
    
    #---------------------#
    #     Tick Labels     #
    #---------------------#
    # Turn off tick labels for plot
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    cbar.set_ticklabels([])
    
    #-----------------------#
    #     Figures stuff     #
    #-----------------------#
    # fig.tight_layout()
    # fig.tight_layout(pad=0.2, h_pad=0.2, w_pad=0.2)
    fig.savefig(filename_out.format(subfigure_labels[i]))
    # fig.show()

for i in range(len(xi_directory)):
    plot_figure(i)

