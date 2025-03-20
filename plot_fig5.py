# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 01:03:32 2023

@author: Jacob
"""

import numpy as np
import matplotlib.pyplot as plt

# Add figure style sheet
plt.style.use('default')
plt.style.use('./python_files/figure_style.mplstyle')

plt.close('all')

# Figure filename
filename_out = "../svgs/fig5_spectrum_high_drive_xi.svg"

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
                     reverse=False)
    
    # Calculate spectrum
    spec, wlist = spectrum(tau, G1_out)
    
    if output == 'g1':
        return tau, G1_out
    elif output == 'spectrum':
        return spec, wlist
    elif output == 'both':
        return tau, G1_out, spec, wlist
    
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
tau_max = 100.0
# Tau list
tau = np.arange(0, tau_max + dt, dt)

#-----------------------------------------------------------------------------#
#                              CALCULATE THINGS                               #
#-----------------------------------------------------------------------------#
# Fourier transform
specs = []
for i in range(len(xi)):
    spec, wlist = calc_3LA_spectrum(Gamma, Omega, delta, alpha, xi[i], tau,
                                    reverse=True)
    specs.append(spec)

# %%
#-----------------------------------------------------------------------------#
##                              PLOT SPECTRUM                                ##
#-----------------------------------------------------------------------------#
# Figure size in centimetres
figsize = np.array([7.4, 9.1])
figsize *= 1 / 2.54

# Create figure
plt.close('all')
fig, ax = plt.subplots(num='Spectrum (Low and High)', nrows=3, ncols=1,
                       figsize=figsize.tolist(), sharex=True)

#--------------#
#     Plot     #
#--------------#
for i in range(len(xi)):
    # Plot spectrum: Low drive
    ax[i].plot(wlist, specs[i], color='k', ls='solid')


#--------------------#
#     Axis Ticks     #
#--------------------#
for i in range(len(xi)):
    ax[i].set_xticks(np.arange(-120, 180, 60))
    ax[i].set_xticks(np.arange(-120, 180, 15), minor=True)
    
    ax[i].set_yticks(np.arange(0, 1.2, 0.2))
    ax[i].set_yticks(np.arange(0.1, 1.0, 0.2), minor=True)

for i in range(len(xi)):
    ax[i].set_xticklabels([])
    ax[i].set_yticklabels([])

#---------------------#
#     Axis Limits     #
#---------------------#
for i in range(len(xi)):
    ax[i].set_xlim(-120, 120)

    ax[i].set_ylim(-0.05, 1.05)

#---------------------#
#     Axis Labels     #
#---------------------#
# # Labels
# ax[2].set_xlabel(r'$\left( \omega - \omega_{d} \right) / \Gamma$')

# for i in range(len(xi)):
#     ax[i].set_ylabel(r'$S_{\mathrm{inc}}(\omega)$ (a.u.)')

#---------------------#
#     Tick Labels     #
#---------------------#
# Turn off tick labels for plot
for i in range(len(xi)):
    ax[i].set_xticklabels([])
    ax[i].set_yticklabels([])

#----------------------#
#     Figure Stuff     #
#----------------------#
# fig.tight_layout()
fig.tight_layout(pad=0.2, h_pad=0.2, w_pad=0.2)
# fig.savefig(filename_out)
# fig.show()
