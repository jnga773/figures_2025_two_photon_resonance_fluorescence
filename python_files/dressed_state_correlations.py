#==============================================================================#
#         THREE-LEVEL LADDER-TYPE ATOM: DRESSED STATE MOMENT EQUATIONS         #
#==============================================================================#
# This Python module [dressed_state_functions.py] contain functions/routines
# to calculate the dressed-states and dressed-state frequencies of the driven
# three-level ladder-type atom.
#
# This module contains the following functions:
# - _state_str_to_index
#     Converts the input state string value {'m', 'u', 'l'} into an index
#     to map to the correct moment equation.
#
# - hamiltonian_3LA
#     Sets the Hamiltonian of the model.
#
# - three_level_eig
#      Calculates the dressed-state eigenvalues and corresponding eigenvectors
#      of the Hamiltonian.
#
# - three_level_transition_frequencies
#     Calculates the dressed-state transition freqencies from the eigenvalues.
#
# - _which_transition
#     Checks the input frequency against the calculated eigenvalue differences.
#     If there is a match, returns the corresponding dressed-state transition
#     order.
#
# - _Sigma_matrix_elements
#      Calculates and defines the matrix elements (a_i) from the dressed-state
#      representation of the total atomic lowering operator.
#
# - _off_diagonal_Gamma_values
#       Calculates the decay rates for the off-diagonal dressed-state moment
#       equations from the semi-analytical expression.
#
# - _get_Gamma_ij
#       From the initial and final dressed states, get the corresponding
#       Gamma decay rate from off_diagonal_Gamma_values.
#
# - _diagonal_moments_evolution_matrix
#       Sets up the evolution matrix and non-homogeneous vector for the
#       diagonal dressed-state moment equations.
#
# - steady_state_diagonal_moments
#       Calculates the steady state expectation values of the diagonal moment
#       equations. The off-diagonal moments all have zero steady state values.
#
# - calc_off_diagonal_moments
#       Calculates the temporal evolution of the off-diagonal moment equations
#       from the (easily derived) analytical expression.
#
# - calc_diagonal_moments
#       Calculates the temporal evolution of the diagonal moment equations
#       using Runge-Kutta 4th order.
#
# - calc_g1_dressed_state
#       Calculates the temporal evolution of the first-order correlation function
#       using Runge-Kutta 4th order.
#
# - calc_g2_dressed_state
#       Calculates the generalised cross-correlation second-order correlation
#       function for two transition operators: \sigma_{a} and \sigma_{b}.

#------------------------------------------------------------------------------#
def _str_to_operator(str_in):
    """
    Converts the input string into a matrix representation of the operator
    """
    from numpy import array, matmul

    # Basis vectors for dressed state
    m = array([[1], [0], [0]])
    u = array([[0], [1], [0]])
    l = array([[0], [0], [1]])

    # Define operators

    # |m><m|
    mm = matmul(m, m.T)
    # |u><u|
    uu = matmul(u, u.T)
    # |l><l|
    ll = matmul(l, l.T)

    # \sigma_{z}^{ul}
    sz = uu - ll

    # \sigma^{um}_{-} = |m><u|
    um_m = matmul(m, u.T)
    # \sigma^{um}_{+} = |u><m|
    um_p = matmul(u, m.T)

    # \sigma^{ul}_{-} = |l><u|
    ul_m = matmul(l, u.T)
    # \sigma^{ul}_{+} = |u><l|
    ul_p = matmul(u, l.T)

    # \sigma^{ml}_{-} = |l><m|
    ml_m = matmul(l, m.T)
    # \sigma^{ml}_{+} = |m><l|
    ml_p = matmul(m, l.T)

    # Turn into a dictionary
    operator_dict = {'um_m': um_m, 'um_p': um_p,
                     'ml_m': ml_m, 'ml_p': ml_p,
                     'ul_m': ul_m, 'ul_p': ul_p,
                     'mm': mm, 'uu': uu, 'll': ll,
                     'sz': sz}
    
    return operator_dict[str_in]

#------------------------------------------------------------------------------#
def _state_str_to_index(state_in):
    """
    Converts the input state string {'m', 'u', 'l'} into an index {0, 1, 2}.
    The dressed states are, in the defined order: |m>, |u> and |l>.

    Parameters
    ----------
    state_in : string
        Input state to convert

    Returns
    -------
    index_out : integer
        Index for array
    """

    #---------------#
    #     Input     #
    #---------------#
    # Dictionary for string to index
    state_dic = {'m': 0, 'u': 1, 'l': 2}

    # Check if state_in /= 'm', 'u', or 'l'
    if state_in not in ['m', 'u', 'l']:
        from sys import exit
        exit("Wuh woh: state_in != 'm', 'u', or 'l'!")

    #-----------------#
    #      Output     #
    #-----------------#
    # Find index
    index_out = state_dic[state_in]

    return index_out

#------------------------------------------------------------------------------#
def hamiltonian_3LA(Omega_in, alpha_in, delta_in, xi_in):
    """
    Defines the Hamiltonian in the bare-state basis (|g>, |e>, |f>) as a 
    3x3 matrix.

    Parameters
    ----------
    Omega_in : float
        Driving amplitude.
    alpha_in : float
        Anharmonicity of atom.
    delta_in : float
        Driven detuning from two-photon resonance.
    xi_in : float
        Dipole moment ratio.

    Returns
    -------
    H_out: matrix
        3x3 matrix representation of the Hamiltonian.
    """
    from numpy import matrix

    #---------------------------#
    #     Define the Matrix     #
    #---------------------------#
    # Set up the rows of the matrix
    r1 = [0, 0.5 * Omega_in, 0]
    r2 = [0.5 * Omega_in, -((0.5 * alpha_in) + delta_in), 0.5 * xi_in * Omega_in]
    r3 = [0, 0.5 * xi_in * Omega_in, -2.0 * delta_in]

    #----------------#
    #     Output     #
    #----------------#
    # Make the matrix
    H_out = matrix([r1, r2, r3], dtype='float')

    return H_out

#------------------------------------------------------------------------------#
def three_level_eig(Omega_in, alpha_in, delta_in, xi_in, vals_or_vecs='vals'):
    """
    Calculates the dressed/eigenvalues of the Hamiltonian in matrix form.

    Parameters
    ----------
    Omega_in : float
        Driving amplitude.
    alpha_in : float
        Anharmonicity of atom.
    delta_in : float
        Driven detuning from two-photon resonance.
    xi_in : float
        Dipole moment ratio.
    vals_or_vecs : string, optional
        Choose output to be eigenvalues ('vals'), eigenvectors ('vecs') or
        both ('both'). The default is 'vals'.

    Returns
    -------
    eigvals_out : array
        Array of eigenvalues in order [wm, wu, wl].
    eigvecs_out : matrix
        S matrix from diagonalisation of H; containing eigenvectors in order
        [vm, vl, vu].
    """
    from numpy.linalg import eig
    
    #---------------#
    #     Input     #
    #---------------#
    # Generate the matrix
    H = hamiltonian_3LA(Omega_in, alpha_in, delta_in, xi_in)

    #--------------------------#
    #     Calculate Things     #
    #--------------------------#
    # Calculate eigenvalues
    eigvals_out, eigvecs_out = eig(H)

    # Get the indicies that would sort them from big to small
    sorted_indices = eigvals_out.argsort()[::-1]
    # Return the sorted eigenvalues and eigenvectors
    eigvals_out = eigvals_out[sorted_indices]
    eigvecs_out = eigvecs_out[:, sorted_indices]

    # This sorting will be in the order [\omega_{+}, \omega_{0}, \omega_{-}].
    # To match it with the analytic calculations I've done, let's change it to
    # the the order of [\omega_{0}, \omega_{+}, \omega_{-}]
    sorted_indices = [1, 0, 2]
    eigvals_out = eigvals_out[sorted_indices]
    eigvecs_out = eigvecs_out[:, sorted_indices]

    # Check if final element of each eigenvector is negative. If so, multiply
    # that eigenvector by -1.
    for i in range(3):
        if eigvecs_out[-1, i] < 0:
            eigvecs_out[:, i] *= -1

    #----------------#
    #     Output     #
    #----------------#
    # Return the output depending on vals_or_vecs
    if vals_or_vecs == 'vals':
        return eigvals_out
    if vals_or_vecs == 'vecs':
        return eigvecs_out
    if vals_or_vecs == 'both':
        return eigvals_out, eigvecs_out

#------------------------------------------------------------------------------#
def three_level_transition_frequencies(Omega_in, alpha_in, delta_in, xi_in):
    """
    Calculates the transition frequencies from the eigenvalues of the
    Hamiltonian in matrix form.
    """
    
    # Calculate eigenvalues
    wm, wu, wl = three_level_eig(Omega_in, alpha_in, delta_in, xi_in)

    # Transitions
    w0 = 0.0
    w1 = abs(wm - wl)
    w2 = abs(wu - wm)
    w3 = abs(wu - wl)
    
    return w0, w1, w2, w3

#------------------------------------------------------------------------------#
def _which_transition(w_in, Omega_in, alpha_in, delta_in, xi_in, str_or_op='str'):
    """
    Checks where w0 is an compares to dressed-state transition
    frequency to see which dressed-state transition has occured.

    Parameters
    ----------
    w_in : float
        Central resonance frequency of filter.
    Omega_in : float
        Driving amplitude.
    alpha_in : float
        Anharmonicity of atom.
    delta_in : float
        Driven detuning from two-photon resonance.
    xi_in : float
        Dipole moment ratio.
    str_or_op : string
        Either output the array of state label strings, or the operator.

    Returns
    -------
    transition_out = list, string
        String identifier for the operator (e.g. 'um_m'), or the matrix
        of the operator.
    """

    # Calculate three level transition frequencies
    w0, w1, w2, w3 = three_level_transition_frequencies(Omega_in, alpha_in, delta_in, xi_in)

    # For each transition, numerically order transition frequency
    w_trans = [-w3, -w2,
               -w1, w0, w1,
               w2, w3]
    
    # Transition operators
    labels_op_str = ['ul_p', 'um_p',
                     'ml_p', 'sz', 'ml_m',
                     'um_m', 'ul_m']

    # Cycle through w_trans and compare with w0_in to find matching transition
    w_check = False
    for index, w in enumerate(w_trans):
        if round(w_in, 2) == round(w, 2):
            # Set check to True so no fail
            w_check = True
            
            # Grab the transition labels
            if str_or_op == 'str':
                transition_out = labels_op_str[index]
            elif str_or_op == 'op':
                transition_out = _str_to_operator(labels_op_str[index])

    #----------------#
    #     Output     #
    #----------------#
    # If w0_in does not match any transition frequency, exit the function
    # with an error.
    if w_check is False:
        from sys import exit
        exit("Wuh woh! w0_in does not match any eigen-transition!")
    else:
        return transition_out

#------------------------------------------------------------------------------#
#                       MATRIX ELEMENTS AND COEFFICIENTS                       #
#------------------------------------------------------------------------------#
def _Sigma_matrix_elements(Omega_in, alpha_in, delta_in, xi_in):
    r"""
    Calculates the matrix elements (a1 - a9) of the \Sigma_{-} operator in the
    dressed state basis.

    Parameters
    ----------
    Omega_in : float
        Driving amplitude.
    alpha_in : float
        Anharmonicity of atom.
    delta_in : float
        Driven detuning from two-photon resonance.
    xi_in : float
        Dipole moment ratio.

    Returns
    -------
    a_out : array_like
        Array of the matrix elements.
    """
    from numpy.linalg import inv
    from numpy import reshape, asarray

    #--------------------------------#
    #     Calculate Eigenvectors     #
    #--------------------------------#
    # Calculate eigenvectors
    S = three_level_eig(Omega_in, alpha_in, delta_in, xi_in, "vecs")

    # Invert eigvec matrix
    S_inv = inv(S)

    # Generate base-states in dressed-state basis
    g_dressed = S_inv[:, 0]
    e_dressed = S_inv[:, 1]
    f_dressed = S_inv[:, 2]

    # Generate atomic lowering operators
    # |g><e|
    sigma_ge = g_dressed * e_dressed.T
    # |e><f|
    sigma_ef = e_dressed * f_dressed.T

    # Total lowering operator: \Sigma_{-} = |g><e| + \xi |e><f|
    Sm = sigma_ge + (xi_in * sigma_ef)

    #----------------#
    #     Output     #
    #----------------#
    a_out = Sm.flatten('C')
    a_out = reshape(asarray(a_out.T), -1)

    # Return
    return a_out

#------------------------------------------------------------------------------#
def _off_diagonal_Gamma_values(Gamma_in, Omega_in, alpha_in, delta_in, xi_in):
    """
    Calculate the different Gamma values for each of the off-diagonal dressed-state
    operators.

    Parameters
    ----------
    Omega_in : float
        Driving amplitude.
    alpha_in : float
        Anharmonicity of atom.
    delta_in : float
        Driven detuning from two-photon resonance.
    xi_in : float
        Dipole moment ratio.

    Returns
    -------
    Gamma_um, Gamma_ml, Gamma_ul : floats
        Decay rates for each off-diagonal dressed-state transition.
    """
    # Matrix elements
    a_elements = _Sigma_matrix_elements(Omega_in, alpha_in, delta_in, xi_in)
    a1, a2, a3, a4, a5, a6, a7, a8, a9 = a_elements

    # Gamma: |u> <-> |m>
    Gamma_um = Gamma_in * ((a2 ** 2) + (a4 ** 2) + (a7 ** 2) + (a8 ** 2) + ((a1 - a5) ** 2))
    # Gamma: |m> <-> |l>
    Gamma_ml = Gamma_in * ((a3 ** 2) + (a4 ** 2) + (a6 ** 2) + (a7 ** 2) + ((a1 - a9) ** 2))
    # Gamma: |u> <-> |l>
    Gamma_ul = Gamma_in * ((a2 ** 2) + (a3 ** 2) + (a6 ** 2) + (a8 ** 2) + ((a5 - a9) ** 2))

    return Gamma_um, Gamma_ml, Gamma_ul

#------------------------------------------------------------------------------#
def _get_Gamma_ij(Gamma_in, Omega_in, alpha_in, delta_in, xi_in,
                  state_init, state_final):
    """
    Calculate the correct Gamma_ij value from the initial and final states.

    Parameters
    ----------
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
    state_init : string ['m', 'u', 'l']
        Initial state.
    state_final : string ['m', 'u', 'l']
        Final state.
    
    Returns
    -------
    Gamma_out : float
        Corret Gamma rate for off-diagonal moment evolution.
    """
    from numpy import matrix

    #---------------#
    #     Input     #
    #---------------#
    # Get indices
    i_init = _state_str_to_index(state_init)
    i_final = _state_str_to_index(state_final)

    # Calculate dressed state Gamma rates (no matrix)
    Gamma_um, Gamma_ml, Gamma_ul = \
      _off_diagonal_Gamma_values(Gamma_in, Omega_in, alpha_in, delta_in, xi_in)

    #---------------------------#
    #     Sort Gamma Values     #
    #---------------------------#
    # Sort into matrix
    mat = matrix([[0.0, Gamma_um, Gamma_ml],
                  [Gamma_um, 0.0, Gamma_ul],
                  [Gamma_ml, Gamma_ul, 0.0]])
    
    # Grab gamma_value
    Gamma_out = mat[i_init, i_final]

    #----------------#
    #     Output     #
    #----------------#
    # Print warning if Gamma is 0.0
    if Gamma_out == 0.0:
        print("Gamma_ij = 0.0!")
    
    return Gamma_out

#------------------------------------------------------------------------------#
def _diagonal_moments_evolution_matrix(Gamma_in, Omega_in, alpha_in, delta_in,
                                       xi_in, matrix_dim=2):
    r"""
    Calculates the evolution matrix for the diagonal moments (\sigma_{ij} = |i><j|),
        < \sigma_{mm} >, < \sigma_{uu} >, and < \sigma_{ll} > ,
    with either the 3x3 matrix or the 2x2 matrix and non-homogeneous vector.

    Parameters
    ----------
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
    matrix_dim : integer (default = 2)
        3x3 matrix or 2x2 matrix and B Vector

    Returns
    -------
    M_out : matrix
        Evolution matrix of the operators
    B_out : matrix
        If matrix_dim == 2, outputs the non-homogeneous B vector too.
    """
    from numpy import zeros
    
    #---------------#
    #     Input     #
    #---------------#
    # Sigma matrix elements
    a_elements = _Sigma_matrix_elements(Omega_in, alpha_in, delta_in, xi_in)
    a1, a2, a3, a4, a5, a6, a7, a8, a9 = a_elements
    
    #--------------------------------#
    #     Define Matrix Elements     #
    #--------------------------------#
    # Define matrix
    M_out = zeros(shape=(matrix_dim, matrix_dim), dtype='complex')
    # Define homogenous vector
    B_out = zeros(shape=(matrix_dim, 1), dtype='complex')

    if matrix_dim == 3:
        # d/dt |m><m|
        M_out[0, 0] = -Gamma_in * ((a4 ** 2) + (a7 ** 2))
        M_out[0, 1] = Gamma_in * (a2 ** 2)
        M_out[0, 2] = Gamma_in * (a3 ** 2)

        # d/dt |u><u|
        M_out[1, 0] = Gamma_in * (a4 ** 2)
        M_out[1, 1] = -Gamma_in * ((a2 ** 2) + (a8 ** 2))
        M_out[1, 2] = Gamma_in * (a6 ** 2)

        # d/dt |l><l|
        M_out[2, 0] = Gamma_in * (a7 ** 2)
        M_out[2, 1] = Gamma_in * (a8 ** 2)
        M_out[2, 2] = -Gamma_in * ((a3 ** 2) + (a6 ** 2))
        
    elif matrix_dim == 2:
        # d/dt |m><m|
        M_out[0, 0] = -Gamma_in * ((a4 ** 2) + (a7 ** 2) + (a3 ** 2))
        M_out[0, 1] = Gamma_in * ((a2 ** 2) - (a3 ** 2))

        # d/dt |u><u|
        M_out[1, 0] = Gamma_in * ((a4 ** 2) - (a6 ** 2))
        M_out[1, 1] = -Gamma_in * ((a2 ** 2) + (a8 ** 2) + (a6 ** 2))
    
        # B vector thing
        B_out[0, 0] = Gamma_in * (a3 ** 2)
        B_out[1, 0] = Gamma_in * (a6 ** 2)
        
    #----------------#
    #     Output     #
    #----------------#
    return M_out, B_out

#------------------------------------------------------------------------------#
#                    TEMPORAL EVOLUTION OF MOMENT EQUATIONS                    #
#------------------------------------------------------------------------------#
def steady_state_diagonal_moments(Gamma_in, Omega_in, alpha_in, delta_in, xi_in,
                                  vec_mat='vec'):
    r"""
    Calculates the steady states of the diagonal moments (\sigma_{ij} = |i><j|),
        < \sigma_{mm} >, < \sigma_{uu} >, and < \sigma_{ll} > .

    Parameters
    ----------
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
    vec_mat : str
        Output the steady states as a vector or a matrix

    Returns
    -------
    mm_out, uu_ou, ll_out : float
        Steady state values of the operators \sigma_{mm}, \sigma_{uu} and
        \sigma_{ll}.
    """
    from numpy import matmul, array
    from numpy.linalg import inv
    
    #=--------------------------------#
    #     Calculate Steady States     #
    #---------------------------------#
    # 2x2 evolution matrix and non-homogenous vector
    M, B = \
      _diagonal_moments_evolution_matrix(Gamma_in, Omega_in, alpha_in, delta_in,
                                         xi_in, matrix_dim=2)
    
    # Invert the matrix
    M_inv = inv(M)

    # Multiply by the non-homogeneous vector
    ss_out = -1.0 * matmul(M_inv, B)
    
    # Grab individual solutions
    mm_out = ss_out[0, 0].real
    uu_out = ss_out[1, 0].real
    ll_out = 1.0 - (mm_out + uu_out)

    # Steady states vector
    if vec_mat == 'vec':
        steady_states_out = array([[mm_out],
                                   [uu_out],
                                   [ll_out]], dtype='float')
    elif vec_mat == 'mat':
        steady_states_out = array([[mm_out, 0, 0],
                                   [0, uu_out, 0],
                                   [0, 0, ll_out]], dtype='float')
    
    return steady_states_out

#------------------------------------------------------------------------------#
def calc_diagonal_moments(t_in, Gamma_in, Omega_in, alpha_in, delta_in, xi_in,
                          initial_state, output_state='all'):
    r"""
    Calculates the time evolution of the off-diagonal moments:
        \sigma_{ij} = |i > < i | .

    Parameters
    ----------
    t_in: float, array
        Array of t times for evolution.
    Gamma_in: float
        Atomic decay rate.
    Omega_in: float
        Driving amplitude.
    alpha_in: float
        Anharmonicity of atom.
    delta_in: float
        Driven detuning from two-photon resonance.
    xi_in: float
        Dipole moment ratio.
    initial_state : array
        Initial state is an input array (eg [0.3, 0.3, 0.4]).
    output_state : string (default = 'all')
        Which operator we want to output ['all', 'm', 'u', 'l']
    
    Returns
    -------
    moment_out : complex, array
        Time evolution of operator < |i><i| (t) >
    """
    from numpy import zeros, matmul

    #---------------#
    #     Input     #
    #---------------#
    # Time step
    dt = t_in[1] - t_in[0]
    
    # Calculate Matrix and non-homogeneous vector
    M, B = _diagonal_moments_evolution_matrix(Gamma_in, Omega_in, alpha_in, delta_in, xi_in, matrix_dim=3)

    #------------------------#
    #     Initialise RK4     #
    #------------------------#
    # State vector for [\sigma_{mm}, \sigma_{uu}, \sigma_{ll}]
    X = zeros(shape=(3, 1), dtype='complex')

    # Input initial state as numerical values
    X[:, 0] = initial_state[:, 0]

    # Runge-Kutta vectors
    k1 = zeros(shape=(3, 1), dtype='complex')
    k2 = zeros(shape=(3, 1), dtype='complex')
    k3 = zeros(shape=(3, 1), dtype='complex')
    k4 = zeros(shape=(3, 1), dtype='complex')

    # data arrays
    sigma_mm = zeros(shape=len(t_in), dtype='complex')
    sigma_uu = zeros(shape=len(t_in), dtype='complex')
    sigma_ll = zeros(shape=len(t_in), dtype='complex')

    #----------------------------#
    #     Calculate with RK4     #
    #----------------------------#
    # Calculate X with RK4
    for step in range(len(t_in)):
        # Update data
        sigma_mm[step] = X[0, 0]
        sigma_uu[step] = X[1, 0]
        sigma_ll[step] = X[2, 0]

        # Calculate Runge-Kutta Vectors
        k1 = dt * matmul(M, X)
        k2 = dt * matmul(M, X + 0.5 * k1)
        k3 = dt * matmul(M, X + 0.5 * k2) 
        k4 = dt * matmul(M, X + k3)

        # Update X vector
        X += (1/6) * (k1 + 2 * (k2 + k3) + k4)

    # Create list of all moments
    all_moments = [sigma_mm.real,
                   sigma_uu.real,
                   sigma_ll.real]
    
    #----------------#
    #     Output     #
    #----------------#
    if output_state == "all":
        return all_moments
    else:
        i_out = _state_str_to_index(output_state)
        return all_moments[i_out]

#------------------------------------------------------------------------------#
#                        TWO-TIME CORRELATION FUNCTIONS                        #
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
    from numpy import matmul, diag, array

    # If no second-filter given, set second frequency to be the same as first
    if b_op_str_in is None:
        b_op_str_in = a_op_str_in
    
    #---------------------------------------#
    #     Find Dressed State Transition     #
    #---------------------------------------#
    # From the central resonance frequency w0_in, check which transition
    # has occured first.
    a_op = _str_to_operator(a_op_str_in)
    # Now check what the second transition was from w0b
    b_op = _str_to_operator(b_op_str_in)

    # Multiply operators
    ata = matmul(a_op.T, a_op)
    btb = matmul(b_op.T, b_op)
    
    # Calculate steady states of the diagonal moments
    steady_state = steady_state_diagonal_moments(Gamma_in, Omega_in, alpha_in, delta_in, xi_in, 'mat')
    
    # < \sigma_{1}^{\dagger} \sigma_{1} >_ss = < |i_1><i_1| >
    ata_ss = sum(diag(matmul(ata, steady_state)))
    # < \sigma_{2}^{\dagger} \sigma_{2} >_ss = < |i_2><i_2| >
    btb_ss = sum(diag(matmul(btb, steady_state)))
    
    #----------------------------------------#
    #     Calculate Correlation Function     #
    #----------------------------------------#
    # Initial state is given by: < \sigma_{a}^{\dagger} |m><m| \sigma_{a} >
    #                            < \sigma_{a}^{\dagger} |u><u| \sigma_{a} >
    #                            < \sigma_{a}^{\dagger} |l><l| \sigma_{a} >
    X_init = diag(matmul(matmul(a_op, steady_state), a_op.T))
    X_init = array([[X_init[0]],
                    [X_init[1]],
                    [X_init[2]]], dtype='float')

    # Calculate the thing
    corr_calc = calc_diagonal_moments(tau_in, Gamma_in, Omega_in, alpha_in, delta_in, xi_in,
                                      initial_state=X_init, output_state='all')
    
    # We want < \sigma_{b}^{\dagger} \sigma_{b}(\tau) > components only
    G2 = matmul(diag(btb), corr_calc)

    # Normalise
    g2_out = G2 / (ata_ss * btb_ss)

    #----------------#
    #     Output     #
    #----------------#    
    return g2_out.real
    