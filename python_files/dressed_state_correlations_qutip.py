#==============================================================================#
#     THREE-LEVEL LADDER-TYPE ATOM: DRESSED STATE MOMENT EQUATIONS (QuTiP)     #
#==============================================================================#
# This Python module [dressed_state_functions_qutip.py] contain functions/routines
# to calculate the dressed-states and dressed-state frequencies of the driven
# three-level ladder-type ato using the Quantum Toolbox in Python (QuTiP).
#
# This module contains the following functions:
# - _qutip_operators
#       Sets up the operators, Hamiltonian, and Lindblad decay terms for the
#       dressed-state picture master equation.
# - calc_g2

#------------------------------------------------------------------------------#
#                           QUTIP OPERATOR FUNCTIONS                           #
#------------------------------------------------------------------------------#
def _qutip_operators(Gamma_in, Omega_in, alpha_in, delta_in, xi_in):
    """
    Sets up the QuTiP operators.
    """
    from python_files.dressed_state_correlations import three_level_eig, _Sigma_matrix_elements
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

#------------------------------------------------------------------------------#
def calc_g2_dressed_state(tau_in, Gamma_in, Omega_in, alpha_in, delta_in, xi_in,
                          a_op_str_in, b_op_str_in):
    """
    Calculates the approximate dressed state second-order correlation function
    based on input parameters.

    Parameters
    ----------
    tau_in : float, array
        Array of tau times for correlation function.
    Gamma_in : float
        Atomic decay rate.
    Omega_in : float
        Driving amplitude.
    alpha_in : float
        Anharmonicity of atom.
    delta_in : float
        Driven detuning from two-photon resonance.
    xi_in : float
        Dipole moment ratio.
    a_op_str_in : str
        Operator string identifier of the first transition.
    b_op_str_in : str (Default: None)
        Operator string identifier of the second transition (if None, is equal to w0a_in).

    Returns
    -------
    corr_out : complex array
        Normalised second-order correlation function
    """
    from qutip import expect, steadystate, mesolve
    from numpy import exp

    # If no second-filter given, set second frequency to be the same as first
    if b_op_str_in is None:
        b_op_str_in = a_op_str_in
    
    # Get QuTiP operators
    L, operator_dict = _qutip_operators(Gamma_in, Omega_in, alpha_in, delta_in, xi_in)
    
    # Get operators
    a_op = operator_dict[a_op_str_in]
    b_op = operator_dict[b_op_str_in]
    
    # Calculate steady state density operator
    rho_ss = steadystate(L)
    
    # Calculate steady state moments
    ata_ss = expect(a_op.dag() * a_op, rho_ss)
    btb_ss = expect(b_op.dag() * b_op, rho_ss)
    
    # Initial state
    rho0 = a_op * rho_ss * a_op.dag()
    
    #------------------------------------#
    #     Calculate: Long-Time Limit     #
    #------------------------------------#
    # Calculate second-order correlation function
    result = mesolve(L, rho0, tau_in, e_ops=b_op.dag() * b_op)
    G2 = result.expect[0]
    
    # Normalise
    g2_out = G2 / (ata_ss * btb_ss)

    #----------------#
    #     Output     #
    #----------------#    
    return g2_out.real
