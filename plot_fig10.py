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
filename_out = "./svgs/fig10_g2_dressed_cross.svg"

#-----------------------------------------------------------------------------#
#                                FUNCTIONS                                    #
#-----------------------------------------------------------------------------#
def _peak_label_to_operator(label_str_in):
    """
    Covert the peak label ('-3', '-2', '-1', '0', '+1', '+2', '+3') into
    the correct operator.
    """
    # List of possible input strings
    possible_inputs = ['-3', '-2', '-1', '0', '+1', '+2', '+3']
    
    # Check if input is not in possible_inputs
    if label_str_in not in possible_inputs:
        from sys import exit
        exit("Wuh woh: state_in != {'-'3, '-2', '-1', '0', '+1', '+2', '+3'}")
    
    # Dictionary for correct operators
    operator_dict = {'-3': 'ul_m',
                     '-2': 'um_m',
                     '-1': 'ml_p',
                      '0': 'sz',
                     '+1': 'ml_m',
                     '+2': 'um_p',
                     '+3': 'ul_p'}
    
    return operator_dict[label_str_in]

def qutip_operators(Gamma_in, Omega_in, alpha_in, delta_in, xi_in):
    """
    Sets up the QuTiP operators.
    """
    from python_files.dressed_state_functions import three_level_eig, _Sigma_matrix_elements
    from qutip import basis, lindblad_dissipator, liouvillian
    from numpy import sqrt
    
    # Calculate eigenvalues for Hamiltonaian
    wm, wu, wl = three_level_eig(Omega_in, alpha_in, delta_in, xi_in, 'vals')
    
    # Get matrix elements of Sigma_{-}
    a1, a2, a3, a4, a5, a6, a7, a8, a9 = \
        _Sigma_matrix_elements(Omega_in, alpha_in, delta_in, xi_in)
    
    # Dressed state: |m>, |u>, |l>
    m, u, l = (basis(3, 0), basis(3, 1), basis(3, 2))

    # |Lowering operators
    um_m = m * u.dag()
    ml_m = l * m.dag()
    ul_m = l * u.dag()
    
    # Raising operators
    um_p = um_m.dag()
    ml_p = ml_m.dag()
    ul_p = ul_m.dag()
    
    mm = m * m.dag()
    uu = u * u.dag()
    ll = l * l.dag()
    
    sz = uu - ll
    
    # Sort into dictionary for quick and easy access
    operator_dict = {'um_m': um_m, 'um_p': um_p,
                     'ml_m': ml_m, 'ml_p': ml_p,
                     'ul_m': ul_m, 'ul_p': ul_p,
                     'sz': sz}

    #---------------------#
    #     Hamiltonian     #
    #---------------------#
    # Hamiltonian
    H_A = (wm * mm) + (wu * uu) + (wl * ll)
    
    # Collapse operators
    c_ops = [sqrt(Gamma_in * (a1 ** 2)) * mm,
             sqrt(Gamma_in * (a5 ** 2)) * uu,
             sqrt(Gamma_in * (a9 ** 2)) * ll,
             #-----------------------#
             sqrt(Gamma_in * (a2 ** 2)) * um_m,
             sqrt(Gamma_in * (a4 ** 2)) * um_p,
             #-----------------------#
             sqrt(Gamma_in * (a3 ** 2)) * ml_p,
             sqrt(Gamma_in * (a7 ** 2)) * ml_m,
             #-----------------------#
             sqrt(Gamma_in * (a6 ** 2)) * ul_p,
             sqrt(Gamma_in * (a8 ** 2)) * ul_m]
    
    # Turn into Louivillian
    L_out = liouvillian(H_A, c_ops)
    # # Append other terms
    L_out = L_out + (Gamma_in * a1 * a5) * (lindblad_dissipator(mm, uu) + lindblad_dissipator(uu, mm))
    L_out = L_out + (Gamma_in * a1 * a9) * (lindblad_dissipator(mm, ll) + lindblad_dissipator(ll, mm))
    L_out = L_out + (Gamma_in * a5 * a9) * (lindblad_dissipator(uu, ll) + lindblad_dissipator(ll, uu))
    
    #----------------#
    #     Output     #
    #----------------#
    # return H_A, c_ops, operator_dict
    return L_out, operator_dict

def calc_g2_qutip(tau_in, Gamma_in, Omega_in, alpha_in, delta_in, xi_in,
                  a_op_str, b_op_str):
    """
    Uses QuTiP to calculate the second-order correlation function.
    """
    from qutip import expect, steadystate, mesolve
    
    # Get QuTiP operators
    L, operator_dict = qutip_operators(Gamma_in, Omega_in, alpha_in, delta_in, xi_in)
    
    # # Get operators
    a_op_qutip = operator_dict[a_op_str]
    b_op_qutip = operator_dict[b_op_str]    
    
    # Calculate steady state density operator
    rho_ss = steadystate(L)
    
    # Calculate steady state moments
    a_ss = expect(a_op_qutip.dag() * a_op_qutip, rho_ss)
    b_ss = expect(b_op_qutip.dag() * b_op_qutip, rho_ss)
    
    # Initial state
    rho0 = a_op_qutip * rho_ss * a_op_qutip.dag()
    
    # Calculate second-order correlation function
    result = mesolve(L, rho0, tau_in, e_ops=b_op_qutip.dag() * b_op_qutip)
    G2 = result.expect[0]
    
    # Normalise
    G2 *= (1 / (a_ss * b_ss))
    
    return G2.real

def calc_g2_cross_dressed(tau_in, Gamma_in, Omega_in, alpha_in, delta_in, xi_in,
                          peak_1_in, peak_2_in):
    """
    Calculates the two-time cross correlation function for two different
    transitions, using the dressed state correlation functions.
    """
    # Convert peak string into operator string
    a_op_str = _peak_label_to_operator(peak_1_in)
    b_op_str = _peak_label_to_operator(peak_2_in)
    
    # Calculate the positive times
    dressed_pos_out = calc_g2_qutip(tau_in, Gamma_in, Omega_in, alpha_in, delta_in, xi_in,
                                    a_op_str, b_op_str)
    
    # Flip operators and calculate negative times
    dressed_neg_out = calc_g2_qutip(tau_in, Gamma_in, Omega_in, alpha_in, delta_in, xi_in,
                                    b_op_str, a_op_str)
    
    return dressed_neg_out, dressed_pos_out

def dressed_rates(Gamma_in, xi_in):
    """
    The two exponential rates in the derived functions.
    """
    lam_neg = -3 * Gamma_in * (xi_in ** 2)
    lam_pos = -Gamma_in * (1 - (xi_in ** 2) + (xi_in ** 4))
    
    lam_neg *= 0.5 / (1 + (xi_in ** 2))
    lam_pos *= 0.5 / (1 + (xi_in ** 2))
    
    return lam_neg, lam_pos

#-----------------------------------------------------------------------------#
#                                PARAMETERS                                   #
#-----------------------------------------------------------------------------#
# Atomic decay
Gamma = 1.0
# Driving amplitude
# Omega = 40
Omega = 1e6
# Anharmonicity
alpha = -120.0
# Drive detuning from two-photon resonance
delta = 0.0
# Dipole moment ratio
# xi = 1 / np.sqrt(2)
xi = 1.0 
# xi = np.sqrt(2)

# Time step
dt = 0.001
# Max time
tau_max = 30.0
# List of times
tau = np.arange(0, tau_max + dt, dt)

#-----------------------------------------------------------------------------#
#                  CALCULATE DRESSED STATE MOMENT EVOLUTION                   #
#-----------------------------------------------------------------------------#
def plot_label(peak1_in, peak2_in):
    return rf'$g^{{(2)}}(\tilde{{\omega}}_{{{peak1_in}}}, 0; \tilde{{\omega}}_{{{peak2_in}}}, \tau)$'

#----------------#
#     Peak 1     #
#----------------#
# Set operators
peak1 = '-3'
peak2 = '+3'

# Calculate dressed state correlation functions
g2_dressed_neg_1, g2_dressed_pos_1 = \
    calc_g2_cross_dressed(tau, Gamma, Omega, alpha, delta, xi,
                          peak1, peak2)

# Plot label
label1 = plot_label(peak1, peak2)
    
#----------------#
#     Peak 2     #
#----------------#
# Set operators
peak1 = '-1'
peak2 = '+1'

# Calculate dressed state correlation functions
g2_dressed_neg_2, g2_dressed_pos_2 = \
    calc_g2_cross_dressed(tau, Gamma, Omega, alpha, delta, xi,
                          peak1, peak2)

# Plot label
label2 = plot_label(peak1, peak2)

#----------------#
#     Peak 3     #
#----------------#
# Set operators
peak1 = '-2'
peak2 = '+1'

# Calculate dressed state correlation functions
g2_dressed_neg_3, g2_dressed_pos_3 = \
    calc_g2_cross_dressed(tau, Gamma, Omega, alpha, delta, xi,
                          peak1, peak2)

# Plot label
label3 = plot_label(peak1, peak2)

#------------------#
#     Analytic     #
#------------------#
# # Exponential rates
# lam_neg, lam_pos = dressed_rates(Gamma, xi)

# # Analytic expressions
# g2_anal = 1 + (0.5 * np.exp(lam_neg * tau)) + (1.5 * np.exp(lam_pos * tau))
# g2_anal = 1 + (2 * np.exp(lam_neg * tau))

# %%
#-----------------------------------------------------------------------------#
#                           PLOT CROSS-CORRELATION                            #
#-----------------------------------------------------------------------------#
# Figure size in centimetres
figsize = np.array([7.4, 3.5])
figsize *= 1 / 2.54

# Create figure
plt.close('all')
fig = plt.figure(num='G2 Cross', figsize=figsize.tolist())
ax = plt.gca()

# Add dashed line at \tau = 0
ax.axvline(x=0.0, color='k', alpha=0.25, ls='solid')

#--------------------------------#
#     Plot: 'Positive' Times     #
#--------------------------------#
# Plot: Dressed state correlation function
ax.plot(tau, g2_dressed_pos_1, color='C0', ls='solid', label=label1)
ax.plot(tau, g2_dressed_pos_2, color='C1', ls='dashed', label=label2)
ax.plot(tau, g2_dressed_pos_3, color='C2', ls='dotted', label=label3)

#--------------------------------#
#     Plot: 'Negative' Times     #
#--------------------------------#
# Plot: Dressed state correlation function
ax.plot(np.flip(-tau), np.flip(g2_dressed_neg_1), color='C0', ls='solid')
ax.plot(np.flip(-tau), np.flip(g2_dressed_neg_2), color='C1', ls='dashed')
ax.plot(np.flip(-tau), np.flip(g2_dressed_neg_3), color='C2', ls='dotted')

#------------------------------------#
#     Plot: Analytic Expressions     #
#------------------------------------#
# ax.plot(tau, g2_anal, color='k', ls='dotted', label='Analytic')
# ax.plot(np.flip(-tau), np.flip(g2_anal), color='k', ls='dotted')

#----------------#
#     Legend     #
#----------------#
# ax.legend(fontsize=5)

#--------------------#
#     Axis Ticks     #
#--------------------#
# X-Axis
ax.set_xticks(np.arange(-10.0, 12.0, 2.0))
ax.set_xticks(np.arange(-10.0, 12.0, 1.0), minor=True)

ax.set_yticks(np.arange(0.0, 3.5, 0.5))
ax.set_yticks(np.arange(0.0, 3.5, 0.25), minor=True)

#---------------------#
#     Axis Limits     #
#---------------------#
# ax.set_xlim(-5.05, 5.05)
ax.set_xlim(-10, 10)
ax.set_ylim(-0.05, 3.05)

#---------------------#
#     Axis Labels     #
#---------------------#
# ax.set_xlabel(r'$\Gamma \tau$')
# ax.set_ylabel(r'$g^{{(2)}}(\tilde{\omega}_{i}, 0; \tilde{\omega}_{j}, \tau)$')

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
fig.show()
