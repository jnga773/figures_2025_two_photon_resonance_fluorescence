#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 19:43:32 2024

@author: jacob
"""

import numpy as np
import matplotlib.pyplot as plt

# Add thesis style sheet
plt.style.use('./python_files/figure_style.mplstyle')

plt.close('all')

# Xi directory
xi_directory = ["xi_0-5", "xi_1-0", "xi_1-5"]

# Figure filename
filename_out = './images/fig4_spectrum_scan_Omega.svg'

#-----------------------------------------------------------------------------#
#                                  FUNCTIONS                                  #
#-----------------------------------------------------------------------------#
def spectrum(tau_input, corr_input):
    from numpy.fft import fft, fftshift, fftfreq
    from numpy import where, mean, pi, array
    from numpy import max as npmax

    # Shift the arrays so they are arranged from negative to positive freq
    fft = fft(corr_input)  # , norm='ortho')
    fft = fftshift(fft)
    freq = fftfreq(tau_input.shape[0], tau_input[1]-tau_input[0])
    freq = fftshift(freq)

    # As the central peak is a delta function, we ignore the plot for w=0. To
    # do this we find the index in wlist where it is equal to 0.0, ignore it,
    # and create a new list excluding this value.
    indices = where(freq != 0.0)[0]

    # Remove zero frequency term
    spec_output = fft[indices]

    # Take only the real part
    spec_output = spec_output.real

    # take away non zero tails
    spec_output = spec_output - mean(spec_output[0])
    wlist_output = freq[indices]  # wlist is in terms of FFT frequencies
    wlist_output = array(wlist_output) * 2 * pi

    # Normalise
    spec_output = spec_output / npmax(spec_output)

    return spec_output, wlist_output

def read_data(xi_in):
    """
    Reads the data my g
    """

    # Read some parameters
    filename_parameters = "./fig4_data/data_files/scan_{}/g1_parameters.txt".format(xi_in[0])

    gamma, alpha, delta, xi, dt, tau_max = \
    np.genfromtxt(filename_parameters, delimiter="=", usecols=1,
                max_rows=6)

    # Time list
    tau = np.arange(0, tau_max + dt, dt)

    # Pull halfwidth values
    Omegas = np.genfromtxt(filename_parameters, delimiter="=", usecols=1,
                           skip_header=8, dtype='float')
    
    # Empty list to save spectrum data to
    spec_out = []

    # Cycle through "xi_values" and read data
    for xi_directory in xi_in:
        # Data filenames
        filename_data_r = "./fig4_data/data_files/scan_{}/g1_corr_real.txt".format(xi_directory)
        filename_data_i = "./fig4_data/data_files/scan_{}/g1_corr_imag.txt".format(xi_directory)

        # Read data
        g1_scans = np.genfromtxt(filename_data_r, dtype='float') + 1j * \
                   np.genfromtxt(filename_data_i, dtype='float')

        #--------------------------------------#
        #     Calculate Fourier Transforms     #
        #--------------------------------------#
        # Create empty array (NOT MATRIX)
        spectrum_scans = np.zeros(shape=(len(tau)-1, len(Omegas)))

        for i in range(len(Omegas)):
            # Calculate Fourier transform
            g1 = g1_scans[:, i]
            spec, wlist = spectrum(tau, g1)

            # Append to big list
            spectrum_scans[:, i] = spec

        # Add to spec_out
        spec_out.append(spectrum_scans)

    # X-data (Make it bigger by one dimension)
    X_out = np.linspace(Omegas.min(), Omegas.max(), len(Omegas)+1)
    # Y-data (Make it bigger by one dimension)
    Y_out = np.linspace(wlist.min(), wlist.max(), len(wlist)+1)

    #----------------#
    #     Output     #
    #----------------#
    return X_out, Y_out, spec_out

def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    from matplotlib import colors
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap

def stretch_boundary_norm(boundaries_in, ncolours_in=256):
    from matplotlib.colors import BoundaryNorm
    from numpy import linspace, interp
    
    # Number of boundary points
    N_bounds = len(boundaries_in)
    # Stretch the bounds so colourmap is continuous-esque
    stretched_bounds = interp(linspace(0, 1, ncolours_in+1), linspace(0, 1, N_bounds),
                              boundaries_in)
    # Return the norm
    norm_out = BoundaryNorm(stretched_bounds, ncolors=ncolours_in)
    
    return norm_out

#-----------------------------------------------------------------------------#
#                                PARAMETERS                                   #
#-----------------------------------------------------------------------------#
# Read data
X, Y, spectrum_scans = read_data(xi_directory)

# Colour bar ticks
# boundaries = [[0.0, 0.05, 0.14, 0.2, 0.38, 0.5, 1.0],
#               [0.0, 0.1, 0.2, 0.38, 0.5, 0.85, 1.0],
#               [0.0, 0.04, 0.05, 0.13, 0.16, 0.5, 0.85, 1.0]]

# boundaries_minor = [[0.025, 0.095, 0.17, 0.27, 0.44, 0.75],
#                     [0.5, 0.15, 0.27, 0.44, 0.675, 0.925],
#                     [0.02, 0.045, 0.09, 0.145, 0.38, 0.675, 0.925]]

boundaries = [0.0, 0.09, 0.18, 0.28, 0.37, 0.60, 0.84, 0.92, 1.0]
boundaries_minor = [0.045, 0.135, 0.23, 0.325, 0.485, 0.72, 0.88, 0.96]

# boundaries = [0.0, 0.18, 0.37, 0.84, 1.0]
# boundaries_minor = [0.09, 0.28, 0.60, 0.92]

# %%
#-----------------------------------------------------------------------------#
##                              PLOT SPECTRUM                                ##
#-----------------------------------------------------------------------------#
# Plotting index
i = 1

# from matplotlib.cm import get_cmap
from matplotlib import colormaps
cmap = colormaps['inferno_r']

# Figure size in centimetres
figsize = np.array([7.4, 3.5])
figsize *= 1 / 2.54

# Create figure
plt.close('all')
fig = plt.figure(num='Spectrum Scans (Omega)', figsize=figsize)
ax = plt.gca()

# Calculate the stretched boundary norm
norm = stretch_boundary_norm(boundaries)

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
cbar.set_ticks(boundaries)
cbar.set_ticks(boundaries_minor, minor=True)
cbar.ax.tick_params(direction='out', which='both')

#--------------------#
#     Axis ticks     #
#--------------------#
ax.set_xticks(np.arange(0.0, 60.0, 10.0))
ax.set_xticks(np.arange(5.0, 60.0, 10.0), minor=True)

ax.set_yticks(np.arange(-120, 160, 60))
ax.set_yticks(np.arange(-90, 160, 60), minor=True)

#---------------------#
#     Axis Limits     #
#---------------------#
ax.set_ylim(-120, 120)
ax.set_xlim(0.0, 50.0)

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
fig.tight_layout(pad=0.2, h_pad=0.2, w_pad=0.2)
fig.savefig(filename_out)
# fig.show()
