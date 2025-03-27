#==============================================================================#
#          THREE-LEVEL LADDER-TYPE ATOM: BARE STATE MOMENT EQUATIONS           #
#==============================================================================#
# This Python module [three_level_moments.py] contain functions/routines to
# calculate the time evolution of the atomic operator moments for a three-level
# ladder-type atom.
#
# This module contains the following functions:
# - _ij
#   Converts the input state string into an index.
#
# - _matrix
#   Calculates the 9x9 matrix that governs the evolution of the atomic operator
#   moments.
#
# - three_level_eig
#   Calculates the dressed/eigenvalues of the Hamiltonian in matrix form.
#
# - print_transition_frequencies
#   Prints or outputs the frequencies for the dressed-state transitions.
#
# - steady_states
#   Calculates the steady states (if xi != 0) for the three-level atomic
#   operator moments using the inverse matrix method.
#
# - evolve_moments
#   Using Runge-Kutta 4th order, solves for the time evolution of the three-
#   level atomic operator moments.
#
# - atomic_population
#   Using Runge-Kutta 4th order, solves for the time evolution of the three-
#   level atomic operator moments.
#
# - g1_calc
#   Using Runge-Kutta 4th order, solves for the time evolution of the first-
#   order correlation function <\Sigma_{+}(0) \Sigma_{-}(\tau) >
#
# - g2_calc
#   Using Runge-Kutta 4th order, solves for the time evolution of the second-
#   order correlation function:
#       < \Sigma_{+}(0) \Sigma_{+} \Sigma_{-}(\tau) \Sigma_{-}(0) >

#------------------------------------------------------------------------------#
def _ij(state_in):
    r"""
    Converts the input state string into an index:
        {'gg', 'ge', 'gf',         {0, 1, 2,
         'eg', 'ee', 'ef',    ===>  3, 4, 5,
         'fg', 'fe', 'ff'}          6, 7, 8}

    Parameters
    ----------
    state_in : string
        Input state to convert

    Returns
    -------
    index_out : integer
        Index for array
    """
    # Dictionary for string to index
    state_dic = {'gg': 0, 'ge': 1, 'gf': 2,
                 'eg': 3, 'ee': 4, 'ef': 5,
                 'fg': 6, 'fe': 7, 'ff': 8}
    # Check if state_in /= '0', '+', or '-'
    if state_in not in ['gg', 'ge', 'gf', 'eg', 'ee', 'ef', 'fg', 'fe', 'ff']:
        from sys import exit
        exit("pee pee poo poo: state_in != 'ij' where i,j = {g,e,f} !")
    # Find index
    index_out = state_dic[state_in]

    return index_out

#------------------------------------------------------------------------------#
def _matrix(Gamma_in, Omega_in, alpha_in, delta_in, xi_in):
    r"""
    Calculates the 9x9 matrix:
        d/dt <\rho(t)> = M <\sigma(t)>,
    for <\rho(t)> = (|g><g|, |g><e|, |g><f|,
                     |e><g|, |e><e|, |e><f|,
                     |f><g|, |f><e|, |f><f|).
    
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
    
    Output
    ------
    matrix_out : complex, matrix
        9x9 matrix that governs the evolution of the atomic operator moments.
    """
    from numpy import matrix
    # Diagonal decay rate
    G_ge = -((0.5 * Gamma_in) + 1j * ((0.5 * alpha_in) + delta_in))
    G_eg = -((0.5 * Gamma_in) - 1j * ((0.5 * alpha_in) + delta_in))
    
    G_ef = -((0.5 * Gamma_in * (1 + (xi_in ** 2)) - 1j * ((0.5 * alpha_in) - delta_in)))
    G_fe = -((0.5 * Gamma_in * (1 + (xi_in ** 2)) + 1j * ((0.5 * alpha_in) - delta_in)))

    G_gf = -((0.5 * Gamma_in * (xi_in ** 2)) + 2j * delta_in)
    G_fg = -((0.5 * Gamma_in * (xi_in ** 2)) - 2j * delta_in)
    
    # d/dt \rho_gg
    gg = [0, 0.5j * Omega_in, 0,
          -0.5j * Omega_in, Gamma_in, 0,
          0, 0, 0]
    # d/dt \rho_ge
    ge = [0.5j * Omega_in, G_ge, 0.5j * xi_in * Omega_in,
          0, -0.5j * Omega_in, Gamma_in * xi_in,
          0, 0, 0]
    # d/dt \rho_gf
    gf = [0, 0.5j * xi_in * Omega_in, G_gf,
          0, 0, -0.5j * Omega_in,
          0, 0, 0]

    # d/dt \rho_eg
    eg = [-0.5j * Omega_in, 0, 0,
          G_eg, 0.5j * Omega_in, 0,
          -0.5j * xi_in * Omega_in, Gamma_in * xi_in, 0]
    # d/dt \rho_ee
    ee = [0, -0.5j * Omega_in, 0,
          0.5j * Omega_in, -Gamma_in, 0.5j * xi_in * Omega_in,
          0, -0.5j * xi_in * Omega_in, Gamma_in * (xi_in ** 2)]
    # d/dt \rho_ef
    ef = [0, 0, -0.5j * Omega_in,
          0, 0.5j * xi_in * Omega_in, G_ef,
          0, 0, -0.5j * xi_in * Omega_in]

    # d/dt \rho_fg
    fg = [0, 0, 0,
          0, -0.5j * xi_in * Omega_in, 0,
          G_fg, 0.5j * Omega_in, 0]
    fg = [0, 0, 0,
          -0.5j * xi_in * Omega_in, 0, 0,
          G_fg, 0.5j * Omega_in, 0]
    # d/dt \rho_fe
    fe = [0, 0, 0,
          0, -0.5j * xi_in * Omega_in, 0,
          0.5j * Omega_in, G_fe, 0.5j * xi_in * Omega_in]
    # d/dt \rho_ff
    ff = [0, 0, 0,
          0, 0, -0.5j * xi_in * Omega_in,
          0, 0.5j * xi_in * Omega_in, -Gamma_in * (xi_in ** 2)]
    
    # Combine rows into a matrix
    matrix_out = matrix([gg, ge, gf,
                         eg, ee, ef,
                         fg, fe, ff], dtype='complex')
    
    return matrix_out

#------------------------------------------------------------------------------#
def three_level_eig(Omega_in, alpha_in, delta_in, xi_in, vals_or_vecs='vals'):
    r"""
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
        Array of eigenvalues in order [w0, wp, wm].
    eigvecs_out : matrix
        S matrix from diagonalisation of H; containing eigenvectors in order
        [v0, vp, vm].
    """
    from numpy.linalg import eig
    from numpy import matrix
    # Set up matrix
    H = matrix([[0, 0.5 * Omega_in, 0],
                [0.5 * Omega_in, -(0.5 * alpha_in) - delta_in, 0.5 * xi_in * Omega_in],
                [0, 0.5 * xi_in * Omega_in, -2 * delta_in]])
    # Calculate eigenvalues
    # eigvals_out = eigvals(H)
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

    # Return the output depending on vals_or_vecs
    if vals_or_vecs == 'vals':
        return eigvals_out
    if vals_or_vecs == 'vecs':
        return eigvecs_out
    if vals_or_vecs == 'both':
        return eigvals_out, eigvecs_out
    
#------------------------------------------------------------------------------#
def print_transition_frequencies(Omega_in, alpha_in, delta_in, xi_in,
                                 output=False):
    r"""
    Prints or outputs the frequencies for the dressed-state transitions.
    
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
    output : boolean (default = False)
        If True, outputs the frequencies as a list
        
    Returns
    -------
    transition_frequencies : array like
        List of transition frequencies in numerical order
    """
    # Calculate eigen-frequencies
    w0, wp, wm = three_level_eig(Omega_in, alpha_in, delta_in, xi_in,
                                 vals_or_vecs='vals')
    # Print as a pretty thingy
    print(r"|\omega_+ - \omega_-| = {}".format(abs(wp - wm)))
    print(r"|\omega_0 - \omega_+| = {}".format(abs(w0 - wp)))
    print(r"|\omega_0 - \omega_-| = {}".format(abs(w0 - wm)))
    
    if output:
        from numpy import array
        transition_frequencies = [wm - wp,
                                  w0 - wp,
                                  wm - w0,
                                  0.0,
                                  w0 - wm,
                                  wp - w0,
                                  wp - wm]
        transition_frequencies = array(transition_frequencies)
        
        return transition_frequencies

#------------------------------------------------------------------------------#
def steady_states(Gamma_in, Omega_in, alpha_in, delta_in, xi_in):
    r"""
    Calculates the steady states (if xi != 0) for the three-level atomic
    operator moments using the inverse matrix method.
    
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
    
    Output
    ------
    steady_states_out : complex, array
        Steady state values for the operator moments
    """
    # Calculate matrix and B vector
    M = _matrix(Gamma_in, Omega_in, alpha_in, delta_in, xi_in)

    # Find eigenvector corresponding to zero eigenvalue.
    from numpy.linalg import eig
    from numpy import abs, where, squeeze, asarray

    # Calculate eigenvalues
    eigval, eigvec = eig(M)

    # Find where the zero eigenvalue is
    zero = (where(min(abs(eigval)) == abs(eigval)))[0][0]

    # Grab the zero eigenvalue eigenvector
    steady_states_out = eigvec[:, zero]

    # Normalise by sum of |g><g| and |e><e|
    steady_states_out *= 1 / (steady_states_out[_ij('gg')] + 
                              steady_states_out[_ij('ee')] + 
                              steady_states_out[_ij('ff')])
    
    # Squeeze into an array
    steady_states_out = squeeze(asarray(steady_states_out))
        
    return steady_states_out

#------------------------------------------------------------------------------#
def evolve_moments(t_in, Gamma_in, Omega_in, alpha_in, delta_in, xi_in,
                   initial_in=[1, 0, 0, 0, 0, 0, 0, 0, 0], output_in='all'):
    r"""
    Using Runge-Kutta 4th order, solves for the time evolution of the three-
    level atomic operator moments.
    
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
    t_in : float, array
        Array of times to caluclate over.
    initial_in : float, array
        Initial condition for the moments (default to ground state |g><g| = 1)
    output_in : str (Default: 'all')
        Determines which matrix element is returned.
    
    Output
    ------
    moments_out : complex, array
        Time evolution of the operator moments.
    """
    from numpy import zeros, array, matmul
    
    # Time step
    dt = t_in[1] - t_in[0]
    
    # Calculate matrix and B vector
    M = _matrix(Gamma_in, Omega_in, alpha_in, delta_in, xi_in)
    
    # Set initial condition
    X = zeros(shape=(9, 1), dtype='complex')
    X[:, 0] = initial_in

    # Runge-Kutta vectors
    k1 = zeros(shape=(9, 1), dtype='complex')
    k2 = zeros(shape=(9, 1), dtype='complex')
    k3 = zeros(shape=(9, 1), dtype='complex')
    k4 = zeros(shape=(9, 1), dtype='complex')

    # data arrays
    gg = zeros(shape=len(t_in), dtype='complex')
    ge = zeros(shape=len(t_in), dtype='complex')
    eg = zeros(shape=len(t_in), dtype='complex')
    ee = zeros(shape=len(t_in), dtype='complex')
    ef = zeros(shape=len(t_in), dtype='complex')
    fe = zeros(shape=len(t_in), dtype='complex')
    fg = zeros(shape=len(t_in), dtype='complex')
    gf = zeros(shape=len(t_in), dtype='complex')
    ff = zeros(shape=len(t_in), dtype='complex')    

    # Calculate X with RK4
    for step in range(len(t_in)):
        # Update data
        gg[step] = X[_ij('gg'), 0]
        ge[step] = X[_ij('ge'), 0]
        gf[step] = X[_ij('gf'), 0]
        
        eg[step] = X[_ij('eg'), 0]
        ee[step] = X[_ij('ee'), 0]
        ef[step] = X[_ij('ef'), 0]
        
        fg[step] = X[_ij('fg'), 0]
        fe[step] = X[_ij('fe'), 0]
        ff[step] = X[_ij('ff'), 0]

        # Calculate Runge-Kutta Vectors
        k1 = dt * matmul(M, X)
        k2 = dt * matmul(M, X + 0.5 * k1)
        k3 = dt * matmul(M, X + 0.5 * k2)
        k4 = dt * matmul(M, X + k3)

        # Update X vector
        X += (1/6) * (k1 + 2 * (k2 + k3) + k4)

    # Create list of all moments
    moments_out = array([gg, ge, gf,
                         eg, ee, ef,
                         fg, fe, ff])
    if output_in == 'all':
        return moments_out
    else:
        return moments_out[_ij(output_in)]

#------------------------------------------------------------------------------#
def atomic_population(t_in, Gamma_in, Omega_in, alpha_in, delta_in, xi_in,
                      initial_in=[1, 0, 0, 0, 0, 0, 0, 0, 0]):
    r"""
    Using Runge-Kutta 4th order, solves for the time evolution of the three-
    level atomic operator moments.
    
    Parameters
    ----------
    t_in : float, array
        Array of times to caluclate over.
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
    initial_in : float, array
        Initial condition for the moments (default to ground state |g><g| = 1)
    
    Output
    ------
    moments_out : real, array
        Time evolution of the operator moments |g><g|, |e><e|, and |f><f|.
    """
    # Calculate moments
    moments = evolve_moments(t_in, Gamma_in, Omega_in, alpha_in, delta_in,
                             xi_in, initial_in)

    # Return population moments
    moments_out = []
    moments_out.append(moments[_ij('gg')].real)
    moments_out.append(moments[_ij('ee')].real)
    moments_out.append(moments[_ij('ff')].real)
    
    return moments_out

#------------------------------------------------------------------------------#
def g1_calc(tau_in, Gamma_in, Omega_in, alpha_in, delta_in, xi_in,
            reverse=False):
    r"""
    Using Runge-Kutta 4th order, solves for the time evolution of the first-
    order correlation function <\Sigma_{+}(0) \Sigma_{-}(\tau) >
    
    Parameters
    ----------
    tau_in : float, array
        Array of times to caluclate over.
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
    reverse : boolean (default = False)
        If True, calculate <\Sigma_{+}(\tau) \Sigma_{-}(0)>
    Output
    ------
    g1_out : complex, array
        Normalised first-order correlation function.
    """
    # Get the steady states of the operator moments
    ss = steady_states(Gamma_in, Omega_in, alpha_in, delta_in, xi_in)
    
    # Calculate steady state for \Sigma_{+} \Sigma_{-} = |e><e| + \xi^{2} |f><f|
    Spm_ss = ss[_ij('ee')].real + (xi_in ** 2) * ss[_ij('ff')].real

    # Initialise initial condition array
    initial_condition = [0, 0, 0,
                         0, 0, 0,
                         0, 0, 0]
    
    # Initial condition is <\Sigma_{+} \Sigma_{-}>_{ss}
    if reverse is False:
        # Calculate < \Sigma_{+}(\tau) \Sigma_{-}(0) >
        # Propagate \Sigma_{-} \rho_{ss} as the initial state
        # |g><e| \rho
        initial_condition[_ij('gg')] = ss[_ij('eg')]
        initial_condition[_ij('ge')] = ss[_ij('ee')]
        initial_condition[_ij('gf')] = ss[_ij('ef')]
        # \xi |e><f| \rho
        initial_condition[_ij('eg')] = xi_in * ss[_ij('fg')]
        initial_condition[_ij('ee')] = xi_in * ss[_ij('fe')]
        initial_condition[_ij('ef')] = xi_in * ss[_ij('ff')]
    else:
        # Calculate < \Sigma_{+}(0) \Sigma_{-}(\tau) >
        # Propagate \rho_{ss} \Sigma_{+} as the initial state
        # |g><e| \rho
        initial_condition[_ij('gg')] = ss[_ij('ge')]
        initial_condition[_ij('eg')] = ss[_ij('ee')]
        initial_condition[_ij('fg')] = ss[_ij('fe')]
        # \xi |e><f| \rho
        initial_condition[_ij('ge')] = xi_in * ss[_ij('gf')]
        initial_condition[_ij('ee')] = xi_in * ss[_ij('ef')]
        initial_condition[_ij('fe')] = xi_in * ss[_ij('ff')]
    
    moments = evolve_moments(tau_in, Gamma_in, Omega_in, alpha_in, delta_in,
                             xi_in, initial_in=initial_condition)
    
    # Grab the relevant moments
    if reverse is False:
        # Solve for < \Sigma_{+}(\tau) >
        moment_1 = moments[_ij('ge')]
        moment_2 = moments[_ij('ef')]
    else:
        # Solve for < \Sigma_{-}(\tau) >
        moment_1 = moments[_ij('eg')]
        moment_2 = moments[_ij('fe')]
    # Add them together
    g1_out = moment_1 + xi_in * moment_2

    # Normalise
    g1_out *= (1 / Spm_ss)
    
    return g1_out

#------------------------------------------------------------------------------#
def g2_calc(tau_in, Gamma_in, Omega_in, alpha_in, delta_in, xi_in):
    r"""
    Using Runge-Kutta 4th order, solves for the time evolution of the second-
    order correlation function:
        < \Sigma_{+}(0) \Sigma_{+} \Sigma_{-}(\tau) \Sigma_{-}(0) >
    
    Parameters
    ----------
    tau_in : float, array
        Array of times to caluclate over.
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
    reverse : boolean (default = False)
        If True, calculate <\Sigma_{+}(\tau) \Sigma_{-}(0)>
    Output
    ------
    g2_out : real, array
        Normalised first-order correlation function.
    """
    # Get the steady states of the operator moments
    ss = steady_states(Gamma_in, Omega_in, alpha_in, delta_in, xi_in)
    
    # Calculate steady state for \Sigma_{+} \Sigma_{-} = |e><e| + \xi^{2} |f><f|
    Spm_ss = ss[_ij('ee')].real + (xi_in ** 2) * ss[_ij('ff')].real

    # Initialise initial condition array
    initial_condition = [0, 0, 0,
                         0, 0, 0,
                         0, 0, 0]
    
    # Initial condition < \Sigma_{+} \Sigma_{+} \Sigma_{-} \Sigma_{-} >_ss
    # becomes < \Sigma_{+} |e><e| \Sigma_{-} >_ss
    #               + \xi^{2} < \Sigma_{+} |f><f| \Sigma_{-} >_ss.
    
    # So initial conditions are: \Sigma_{-} \rho_{ss} \Sigma_{+}
    initial_condition[_ij('gg')] = ss[_ij('ee')]
    initial_condition[_ij('ge')] = xi_in * ss[_ij('ef')]
    initial_condition[_ij('eg')] = xi_in * ss[_ij('fe')]
    initial_condition[_ij('ee')] = (xi_in ** 2) * ss[_ij('ff')]
    
    # Evolve moments    
    moments = evolve_moments(tau_in, Gamma_in, Omega_in, alpha_in, delta_in,
                             xi_in, initial_in=initial_condition)
    
    # Add them together
    g2_out = moments[_ij('ee')] + (xi_in ** 2) * moments[_ij('ff')]

    # Normalise
    g2_out *= (1 / (Spm_ss ** 2))
    g2_out = g2_out.real
    
    return g2_out
