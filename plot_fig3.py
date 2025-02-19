# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 01:03:32 2023

@author: Jacob
"""

import numpy as np
import matplotlib.pyplot as plt

# Add thesis style sheet
plt.style.use('./python_files/figure_style.mplstyle')

plt.close('all')

# Figure filename
filename_out = "../svgs/fig3_spectrum_labelled_high_low_drive.svg"

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

def calc_3LA_spectrum(Gamma_in, Omega_in, delta_in, alpha_in, xi_in, tau_in,
                      output='spectrum', reverse=True):
    """
    Calculates the power spectrum for the driven three-level atom.
    """
    from python_files.three_level_moments import g1_calc
    
    #----------------------------------------------------#
    #     Calculate First-Order Correlation Function     #
    #----------------------------------------------------#
    # Calculate the first-order correlation function
    G1_out = g1_calc(tau_in, Gamma_in, Omega_in, alpha_in, delta_in, xi_in,
                     reverse=True)
    
    # Calculate spectrum
    spec, wlist = spectrum(tau, G1_out)
    
    if output == 'g1':
        return tau, G1_out
    elif output == 'spectrum':
        return spec, wlist
    elif output == 'both':
        return tau, G1_out, spec, wlist
    
#-----------------------------------------------------------------------------#
#                                PARAMETERS                                   #
#-----------------------------------------------------------------------------#
# Atomic decay rate
Gamma = 1.0
# Driving amplitude
Omega = [0.01, 40.0]
# Drive detuning from two-photon resonance
delta = 0.0
# Atomic anharmonicity
alpha = -120.0
# Dipole ratio
xi = 1.0

# Time step
dt = 0.001
# Max tau
tau_max = 100.0
# Tau list
tau = np.arange(0, tau_max + dt, dt)

#-----------------------------------------------------------------------------#
#                              CALCULATE THINGS                               #
#-----------------------------------------------------------------------------#
# Fourier transform
specs = []
for i in range(len(Omega)):
    spec, wlist = calc_3LA_spectrum(Gamma, Omega[i], delta, alpha, xi, tau,
                                    reverse=True)
    specs.append(spec)

# %%
#-----------------------------------------------------------------------------#
##                              PLOT SPECTRUM                                ##
#-----------------------------------------------------------------------------#
# Figure size in centimetres
figsize = np.array([9, 4.5])
figsize *= 1 / 2.54

fig, ax = plt.subplots(num='Spectrum (Low and High)', nrows=1, ncols=2,
                       figsize=figsize, sharey=True)

# # Set constrained values for figure
# fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01,
#                                 hspace=6.0 / 72.0, wspace=6.0 / 72.0)

#--------------#
#     Plot     #
#--------------#
# Plot spectrum: Low drive
ax[0].plot(wlist, specs[0], color='k', ls='solid', label='Moments')
# Plot spectrum: high drive
ax[1].plot(wlist, specs[1], color='k', ls='solid', label='Moments')

#--------------------#
#     Axis Ticks     #
#--------------------#
ax[0].set_xticks(np.arange(-120, 180, 60))
ax[0].set_xticks(np.arange(-90, 180, 60), minor=True)

ax[1].set_xticks(np.arange(-120, 180, 60))
ax[1].set_xticks(np.arange(-90, 180, 60), minor=True)

ax[0].set_yticks(np.arange(0, 1.2, 0.2))
ax[0].set_yticks(np.arange(0.1, 1.0, 0.2), minor=True)

#---------------------#
#     Axis Limits     #
#---------------------#
ax[0].set_xlim(-120, 120)
ax[1].set_xlim(-120, 120)

ax[0].set_ylim(-0.05, 1.05)
    
#---------------------#
#     Tick Labels     #
#---------------------#
# Turn off tick labels for plot
for i in range(2):
    ax[i].set_xticklabels([])
    ax[i].set_yticklabels([])

#---------------------#
#     Axis Labels     #
#---------------------#
# # Labels
# ax[0].set_ylabel(r'$S_{\mathrm{inc}}(\omega)$ (a.u.)')

# ax[0].set_xlabel(r'$\left( \omega - \omega_{d} \right) / \Gamma$')
# ax[1].set_xlabel(r'$\left( \omega - \omega_{d} \right) / \Gamma$')

#-----------------------#
#     Figure Labels     #
#-----------------------#
# ax[0].text(x=-115, y=0.975, s='(a)', fontsize=11)
# ax[1].text(x=-115, y=0.975, s='(b)', fontsize=11)

#----------------------#
#     Figure Stuff     #
#----------------------#
fig.tight_layout(pad=0.2, h_pad=0.2, w_pad=0.2)
# fig.savefig(filename_out)
# fig.show()
