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
# - _which_transition
#     Checks the input frequency against the calculated eigenvalue differences.
#     If there is a match, returns the corresponding dressed-state transition
#     order.
#
# - hamiltonian_3LA
#     Sets the Hamiltonian of the model.
#
# - three_level_eig
#      Calculates the dressed-state eigenvalues and corresponding eigenvectors
#      of the Hamiltonian.
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
def _which_transition(w_in, Omega_in, alpha_in, delta_in, xi_in):
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

    Returns
    -------
    transition_out = list, string
        Two element list where first is the initial state and
        the second element is the final state ['m', 'u', 'l']
    """

    # Calculate eigenvalues
    wm, wu, wl = three_level_eig(Omega_in, alpha_in, delta_in, xi_in)

    # For each transition, numerically order transition frequency
    w_trans = [-(wu - wl), -(wu - wm),
               -(wm - wl), 0.0, wm - wl,
               wu - wm, wu - wl]

    # Transition labels
    labels_trans = [['l', 'u'], ['m', 'u'],
                    ['l', 'm'], ['u', 'u'], ['m', 'l'],
                    ['u', 'm'], ['u', 'l']]

    # Cycle through w_trans and compare with w0_in to find matching transition
    w_check = False
    for index, w in enumerate(w_trans):
        if round(w_in, 2) == round(w, 2):
            # Set check to True so no fail
            w_check = True
            # Grab the transition labels
            transition_out = labels_trans[index]

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
#                       MATRIX ELEMENTS AND COEFFICIENTS                       #
#------------------------------------------------------------------------------#
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
#------------------------------------------------------------------------------#
def steady_state_diagonal_moments(Gamma_in, Omega_in, alpha_in, delta_in, xi_in):
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

    Returns
    -------
    mm_out, uu_ou, ll_out : float
        Steady state values of the operators \sigma_{mm}, \sigma_{uu} and
        \sigma_{ll}.
    """
    from numpy import matmul
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
    
    return mm_out, uu_out, ll_out

#------------------------------------------------------------------------------#
def calc_off_diagonal_moments(t_in, Gamma_in, Omega_in, alpha_in, delta_in, xi_in,
                              state_init, state_final, initial_in=1.0):
    r"""
    Calculates the time evolution of the off-diagonal moments:
        \sigma_{ij} = |i><j|, i /= j.
    
    Parameters
    ----------
    t_in : float, array
        Array of t times for evolution.
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
    initial_in : float (default = 1.0)
        Initial condition of moment.
    
    Returns
    -------
    moment_out : complex array
        Time evolution of operator < |i><j|(t) >
    """
    from numpy import exp

    #---------------#
    #     Input     #
    #---------------#
    # Calculate the dressed-state frequencies
    w0, wp, wm = three_level_eig(Omega_in, alpha_in, delta_in, xi_in)
    w_dressed = [w0, wp, wm]

    # Grab indices
    i_init = _state_str_to_index(state_init)
    i_final = _state_str_to_index(state_final)

    # Grab necessary dressed-state frequencies
    w_init = w_dressed[i_init]
    w_final = w_dressed[i_final]

    #------------------------------------#
    #     Calculate Moment Evolution     #
    #------------------------------------#
    # Calculate dressed state Gamma rates
    Gamma_ij = _get_Gamma_ij(Gamma_in, Omega_in, alpha_in, delta_in, xi_in,
                             state_init, state_final)

    # Set initial condition
    moment_out = initial_in
    # Calculate moment evolution
    moment_out *= exp(-((0.5 * Gamma_ij) + (1j * (w_init - w_final))) * t_in)

    #----------------#
    #     Output     #
    #----------------#
    return moment_out

#------------------------------------------------------------------------------#
def calc_diagonal_moments(t_in, Gamma_in, Omega_in, alpha_in, delta_in, xi_in,
                          initial_state, B_multiplier=1.0,
                          output_state='all', matrix_dim=2):
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
    initial_state : string or array
        Initial state is either one of the dressed states ['m', 'u', 'l'],
        or an input array (eg [0.3, 0.3, 0.4]).
        this will be the steady state value of the initial two-time state.
    B_multiplier : float (default = 1)
        Scalar multiplier of the non-homogeneous vector B. If calculating
        two-time correlation functions, this is the steady state of the
        product of the outer two operators <X^{\dagger}(0) Y^{\dagger}Y(tau) X(0)>
    output_state : string (default = 'all')
        Which operator we want to output ['all', 'm', 'u', 'l']
    matrix_dim : integer (default = 2)
        3x3 matrix or 2x2 matrix and B Vector
    
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
    M, B = _diagonal_moments_evolution_matrix(Gamma_in, Omega_in, alpha_in, delta_in, xi_in,
                                              matrix_dim)

    # If calculating two-time correlation functions, and using the non-zero
    # B vector, multiply by the steady state value of the X(0) operator
    B *= B_multiplier

    #------------------------#
    #     Initialise RK4     #
    #------------------------#
    # State vector for [\sigma_{mm}, \sigma_{uu}, \sigma_{ll}]
    X = zeros(shape=(matrix_dim, 1), dtype='complex')

    # Check if initial_state is a string
    if initial_state.all() in ['m', 'u', 'l']:
      if matrix_dim == 2 and initial_state == 'l':
          # Starting in the lower state which, in the 2D case, doesn't exist.
          # Initial condition is then 0
          X[0, 0] = 0.0
          X[1, 0] = 0.0

      else:
        # Get index of initial state and output state
        i_init = _state_str_to_index(initial_state)
        # Set initial condition
        X[i_init] = 1.0

    else:
        # Input initial state as numerical values
        X[:, 0] = initial_state[0:matrix_dim]

    # Runge-Kutta vectors
    k1 = zeros(shape=(matrix_dim, 1), dtype='complex')
    k2 = zeros(shape=(matrix_dim, 1), dtype='complex')
    k3 = zeros(shape=(matrix_dim, 1), dtype='complex')
    k4 = zeros(shape=(matrix_dim, 1), dtype='complex')

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

        if matrix_dim == 3:
            sigma_ll[step] = X[2, 0]
        elif matrix_dim == 2:
            sigma_ll[step] = B_multiplier - (X[0, 0] + X[1, 0])

        # Calculate Runge-Kutta Vectors
        k1 = dt * (matmul(M, X) + B)
        k2 = dt * (matmul(M, X + 0.5 * k1) + B)
        k3 = dt * (matmul(M, X + 0.5 * k2) + B)
        k4 = dt * (matmul(M, X + k3) + B)

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
def calc_g1_dressed_state(tau_in, Gamma_in, Omega_in, alpha_in, delta_in, xi_in,
                          w0_in):
    """
    Calculates the approximate dressed state first-order correlation function
    based on input parameters.

    Parameters
    ----------
    tau_in : float, array
        Array of tau times for correlation function.
    w0_in : float
        Central frequency of filter.
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

    Returns
    -------
    corr_out : complex array
        Normalised first-order correlation function.
    """

    #---------------------------------------#
    #     Find Dressed State Transition     #
    #---------------------------------------#
    # From the central resonance frequency w0_in, check which transition
    # has occured.
    transition = _which_transition(w0_in, Omega_in, alpha_in, delta_in,
                                   xi_in)
    
    # Transition is from |i> -> |j> = |j><i|
    initial_state = transition[0]
    final_state   = transition[1]
    
    #----------------------------------------#
    #     Calculate Correlation Function     #
    #----------------------------------------#
    # Calculate steady states of diagonal moments
    steady_state = steady_state_diagonal_moments(Gamma_in, Omega_in, alpha_in, delta_in, xi_in)

    # Initial condition is the steady state of the final state
    # |j><j| operator
    ss_norm = steady_state[_state_str_to_index(final_state)]
    
    if transition[0] != transition[1]:
        # Off-diagonal transition
        corr_out = calc_off_diagonal_moments(tau_in, Gamma_in, Omega_in,
                                             alpha_in, delta_in, xi_in,
                                             initial_state, final_state)
    else:
        # Central peak
        corr_out = calc_diagonal_moments(tau_in, Gamma_in, Omega_in,
                                         alpha_in, delta_in, xi_in,
                                         initial_state, final_state)
    
    #----------------#
    #     Output     #
    #----------------#
    return corr_out

#------------------------------------------------------------------------------#
def calc_g2_dressed_ops(tau_in, Gamma_in, Omega_in, alpha_in, delta_in,
                        xi_in, a_op_str, b_op_str):
    r"""
    Calculates the approximate dressed state second-order correlation function
    based on input parameters:
        G^{(2)}(\tau) = < a^{\dagger}(0) b^{\dagger} b(\tau) a(0) >.

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
    a_op_str : string
        String indentifier of the outer operator.
    b_op_str : float (Default None)
        Frequency of second transition (if None, is equal to w0a_in).

    Returns
    -------
    corr_out : complex array
        Normalised second-order correlation function
    """
    from numpy import zeros, matmul, diagonal
    
    #---------------------------------------#
    #     Find Dressed State Transition     #
    #---------------------------------------#
    # Get the operators from the input strings
    a_op = _str_to_operator(a_op_str)
    b_op = _str_to_operator(b_op_str)
    
    # Calculate steady states of the diagonal moments
    steady_state = steady_state_diagonal_moments(Gamma_in, Omega_in, alpha_in, delta_in, xi_in)


    # < \sigma_{1}^{\dagger} \sigma_{1} >_ss = < |i_1><i_1| >
    ata_ss = sum(matmul(matmul(a_op.T, a_op), steady_state))
    # < \sigma_{2}^{\dagger} \sigma_{2} >_ss = < |i_2><i_2| >
    btb_ss = sum(matmul(matmul(b_op.T, b_op), steady_state))
    
    #----------------------------------------#
    #     Calculate Correlation Function     #
    #----------------------------------------#
    # Initial state is given by: < |i1><f1| |m><m| |f1><i1| >
    #                            < |i1><f1| |u><u| |f1><i1| >
    #                            < |i1><f1| |l><l| |f1><i1| >
    # That is, the steady state value of |i1><i1| in the position of
    # |f1>
    X_init = matmul(matmul(a_op.T, a_op), steady_state)

    # Calculate the thing
    corr_calc = calc_diagonal_moments(tau_in, Gamma_in, Omega_in, alpha_in, delta_in, xi_in,
                                      initial_state=X_init,
                                      output_state='all', matrix_dim=3)

    # corr_out = calc_diagonal_moments(tau_in, Gamma_in, Omega_in, alpha_in, delta_in, xi_in,
    #                                  initial_state=X_init, B_multiplier=ss_norm1,
    #                                  output_state='all', matrix_dim=2)
    
    bbt = matmul(b_op, b_op.T)
    
    # Get the expectation of <b^{\dagger} b>
    corr_out = (corr_calc[0] * bbt[0, 0]) + \
               (corr_calc[1] * bbt[1, 1]) + \
               (corr_calc[2] * bbt[2, 2])
               
    #----------------#
    #     Output     #
    #----------------#
    # Normalise
    corr_out = corr_out.real / (ata_ss * btb_ss)
    
    return corr_out

#------------------------------------------------------------------------------#
def calc_g2_dressed_state(tau_in, Gamma_in, Omega_in, alpha_in, delta_in,
                          xi_in, w0a_in, w0b_in=None):
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
    w0a_in : float
        Frequency of first transition.
    w0b_in : float (Default None)
        Frequency of second transition (if None, is equal to w0a_in).

    Returns
    -------
    corr_out : complex array
        Normalised second-order correlation function
    """
    from numpy import zeros

    # If no second-filter given, set second frequency to be the same as first
    if w0b_in is None:
        w0b_in = w0a_in
    
    #---------------------------------------#
    #     Find Dressed State Transition     #
    #---------------------------------------#
    # From the central resonance frequency w0_in, check which transition
    # has occured first.
    transition1 = _which_transition(w0a_in, Omega_in, alpha_in, delta_in,
                                    xi_in)
    # Now check what the second transition was from w0b
    transition2 = _which_transition(w0b_in, Omega_in, alpha_in, delta_in,
                                    xi_in)
    
    # Initial states
    i_1 = transition1[0]
    i_2 = transition2[0]
    
    # Final states
    f_1 = transition1[1]
    f_2 = transition2[1]
    
    # Calculate steady states of the diagonal moments
    steady_state = steady_state_diagonal_moments(Gamma_in, Omega_in, alpha_in, delta_in, xi_in)

    # Transition is from |i_12> -> |f_12> = |f_12><f_12|, so steady state value is
    # |f_12><f_12| operator

    # < \sigma_{1}^{\dagger} \sigma_{1} >_ss = < |i_1><i_1| >
    ss_norm1 = steady_state[_state_str_to_index(i_1)]
    # < \sigma_{2}^{\dagger} \sigma_{2} >_ss = < |i_2><i_2| >
    ss_norm2 = steady_state[_state_str_to_index(i_2)]
    
    #----------------------------------------#
    #     Calculate Correlation Function     #
    #----------------------------------------#
    # Initial state is given by: < |i1><f1| |m><m| |f1><i1| >
    #                            < |i1><f1| |u><u| |f1><i1| >
    #                            < |i1><f1| |l><l| |f1><i1| >
    # That is, the steady state value of |i1><i1| in the position of
    # |f1>
    X_init = zeros(3)
    X_init[_state_str_to_index(f_1)] = ss_norm1

    # Calculate the thing
    corr_out = calc_diagonal_moments(tau_in, Gamma_in, Omega_in, alpha_in, delta_in, xi_in,
                                     initial_state=X_init,
                                     output_state=i_2, matrix_dim=3)
    # corr_out = calc_diagonal_moments(tau_in, Gamma_in, Omega_in, alpha_in, delta_in, xi_in,
    #                                  initial_state=X_init, B_multiplier=ss_norm1,
    #                                  output_state=i_2, matrix_dim=2)
    
    #----------------#
    #     Output     #
    #----------------#
    # Normalise
    corr_out = corr_out.real / (ss_norm1 * ss_norm2)
    
    return corr_out
    