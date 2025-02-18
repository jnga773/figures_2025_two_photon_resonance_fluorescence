! This module contains the subroutines used in any of the single-filter
! programs [just_steady_states.f90, g1_RK4.f90, g2_RK4.f90].

! This module contains the following subroutines:
! - SquareMatrixInverse: Calculates the inverse of an input matrix.
!
! - SquareMatrixZeroEigenvalue: Calculates the eigenvector of a matrix
!                               corresponding to the zero-valued eigenvalue.
!
! - SteadyStateMoments: Calculates the steady states of the various operator
!                       moment equations for the atom-filter coupled system.
!
! - MatrixInverseSS: Uses SquareMatrixInverse to return the steady state array
!                    without having to type out the multiplication.
!
! - G1_InitialConditions: Calculates the initial conditions for the first-
!                         order correlation function.
!
! - G1_CalculateRK4: Calculates the time evolution of the first-order
!                    correlation function using Runge-Kutta 4th Order.
!
! - G2_InitialConditions: Calculates the initial conditions for the second-
!                         order correlation function.
!
! - G2_InitialValues : Calculates only the initial value of the second-
!                      order correlation function.
!
! - G2_CalculateRK4: Calculates the time evolution of the second-order
!                    correlation function using Runge-Kutta 4th Order.

! This file must be added to the compilation command when compiling any of the 
! single-filter programs. Eg, with Intel oneAPI or GFORTRAN:
!     (IFORT): ifort -qmkl ./[FILENAME].f90 ./MODULE_atom.f90
!  (GFORTRAN): gfortran ./[FILENAME].f90 ./MODULE_atom.f90
!                -I/path/to/LAPACK -L/path/to/LAPACK -llapack -lblas

MODULE ATOM_SUBROUTINES

CONTAINS

!==============================================================================!
!                          LAPACK MATRIX SUBROUTINES                           !
!==============================================================================!
! Subroutine to calculate the inverse of a matrix USING LAPACK LIBRARY
SUBROUTINE SquareMatrixInverse(N_in, MatrixInv_out)
  ! Import the MKL Library LAPACK95, from which the eigenvalue/eigenvector and
  ! matrix inversion subroutines come from.
  ! MUST INCLUDE THE -mkl OR /Qmkl FLAG AS A COMPILER OPTION IF USING INTEL.
  ! Otherwise you'll have to link it yourself and I don't know how to do that :)

  USE LAPACK95

  ! The subroutines used from LAPACK are:
  ! - zGETRF - Calculates LU-factorisation of a complexmatrix so it can be
  !            inverted by...,
  ! - zGETRI - Calculates the inverse of a complex matrix.

  IMPLICIT NONE

  !-------------------------!
  !     INPUT ARGUMENTS     !
  !-------------------------!
  ! Dimension of matrix (N_in x N_in)
  INTEGER, INTENT(IN)                                   :: N_in

  !--------------------------!
  !     OUTPUT ARGUMENTS     !
  !--------------------------!
  ! Inverted matrix to be output
  COMPLEX(KIND=8), DIMENSION(N_in, N_in), INTENT(INOUT) :: MatrixInv_out

  !--------------------------!
  !     SUBROUTINE STUFF     !
  !--------------------------!
  ! Work space dimension
  INTEGER, PARAMETER                                    :: LWMAX = 300
  INTEGER                                               :: LWORK
  ! Work space array
  COMPLEX(KIND=8), DIMENSION(LWMAX)                     :: WORK
  REAL(KIND=8), DIMENSION(2*N_in)                       :: RWORK
  ! LU-factorisation array
  INTEGER, DIMENSION(N_in)                              :: IPIV
  ! Info IO
  INTEGER                                               :: INFO

  ! Perform LU-factorization of matrix
  CALL zGETRF(N_in, N_in, MatrixInv_out, N_in, IPIV, INFO)
  IF (INFO .NE. 0) THEN
    PRINT*, "zGETRF M failed :( INFO = ", INFO
    STOP
  END IF

  ! Query optimal work space
  ! LWORK = -1
  ! CALL zGETRI(N_in, MatrixInv_out, N_in, IPIV, WORK, LWORK, INFO)
  ! ! Set optimal work space and run again
  ! LWORK = MIN(LWMAX, INT(WORK(1)))

  ! Set LWORK to N_in
  LWORK = N_in

  CALL zGETRI(N_in, MatrixInv_out, N_in, IPIV, WORK, LWORK, INFO)

  ! End of subroutine
END SUBROUTINE SquareMatrixInverse

! Subroutine to calculate the inverse of a matrix USING LAPACK LIBRARY
SUBROUTINE SquareMatrixZeroEigenvalue(N_in, Matrix_in, SS_out)
  ! Import the MKL Library LAPACK95, from which the eigenvalue/eigenvector and
  ! matrix inversion subroutines come from.
  ! MUST INCLUDE THE -mkl OR /Qmkl FLAG AS A COMPILER OPTION IF USING INTEL.
  ! Otherwise you'll have to link it yourself and I don't know how to do that :)

  USE LAPACK95

  ! The subroutines used from LAPACK are:
  ! - zGEEV  - Calculates the eigenvalues and eigevectors of a complex matrix.

  IMPLICIT NONE

  !-------------------------!
  !     INPUT ARGUMENTS     !
  !-------------------------!
  ! Dimension of matrix (N_in x N_in)
  INTEGER, INTENT(IN)                                :: N_in
  ! Matrix to calculate eigenvalues/vectors from
  COMPLEX(KIND=8), DIMENSION(N_in, N_in), INTENT(IN) :: Matrix_in

  !--------------------------!
  !     OUTPUT ARGUMENTS     !
  !--------------------------!
  ! Steady state vector out
  COMPLEX(KIND=8), DIMENSION(N_in), INTENT(OUT)      :: SS_out

  !---------------------!
  !     OTHER STUFF     !
  !---------------------!
  INTEGER                                            :: j, x

  !--------------------------!
  !     SUBROUTINE STUFF     !
  !--------------------------!
  ! Work space dimension
  INTEGER, PARAMETER                                 :: LWMAX = 300
  INTEGER                                            :: LWORK
  ! Work space array
  COMPLEX(KIND=8), DIMENSION(LWMAX)                  :: WORK
  REAL(KIND=8), DIMENSION(2*N_in)                    :: RWORK
  ! Eigenvalues of matrix M
  COMPLEX(KIND=8), DIMENSION(N_in)                   :: eigval
  ! S, and S^{-1} matrix for diagonalising eigenvectors
  COMPLEX(KIND=8), DIMENSION(N_in, N_in)             :: S, Sinv
  ! Info IO
  INTEGER                                            :: INFO

  ! Calculate eigenvalues and eigenvectors (Optimal LWORK = 264)
  LWORK = 264
  CALL zGEEV('N', 'V', N_in, Matrix_in, N_in, eigval, S, N_in, S, N_in, WORK, LWORK, RWORK, INFO)
  ! Check convergence
  IF (INFO .GT. 0) THEN
     PRINT*, "zGEEV failed ON eigenvalues/vectors of Mat_OG"
     STOP
  END IF

  SS_out = 0.0d0
  ! Cycle through eigenvalues and, for the eigenvalue that = 0, use that
  ! eigenvector as the steady state
  DO x = 1, N_in
    IF (ABS(REAL(eigval(x))) .LT. 1D-10 .AND. ABS(REAL(eigval(x))) .LT. 1D-10) THEN
      ! Save steady state eigenvector
      DO j = 1, N_in
        SS_out(j) = S(j, x)
      END DO
    END IF
  END DO

  ! End of subroutine
END SUBROUTINE SquareMatrixZeroEigenvalue

!==============================================================================!
!                            STEADY STATE SUBROUTINE                           !
!==============================================================================!
! Subroutine to calculate steady state coupled moments using SquareMatrixInverse
SUBROUTINE MatrixInverseSS(N_in, Matrix_in, Bvec_in, SS_out)
  ! Calculates the steady state of a system of coupled equations,
  !   d/dt < A > = M < A > + B,
  ! by multiplying the non-homogeneous vector with the inverse matrix:
  !   < A >_{ss} = -M^{-1} B.

  ! Parameters
  ! ----------
  ! N_in : integer
  !   The dimension of Matrix_in (N_in x N_in)
  ! Matrix_in : complex matrix, dimension(N_in, N_in)
  !   The matrix which we will invert.
  ! Bvec_in : complex array, dimension(N_in)
  !   The non-homogeneous vector for the system.

  ! Output
  ! ------
  ! SS_out : complex array, dimension(N_in)
  !   Output eigenvector for zero eigenvalue

  !============================================================================!
  !                   DEFINING AND DECLARING VARIABLES/ARRAYS                  !
  !============================================================================!

  IMPLICIT NONE

  !-------------------------!
  !     INPUT ARGUMENTS     !
  !-------------------------!
  ! Dimension of matrix (N_in x N_in)
  INTEGER, INTENT(IN)                                   :: N_in
  ! Input evolution matrix
  COMPLEX(KIND=8), DIMENSION(N_in, N_in), INTENT(IN)    :: Matrix_in
  ! Non-homogeneous vecotr
  COMPLEX(KIND=8), DIMENSION(N_in), INTENT(IN)          :: Bvec_in

  !--------------------------!
  !     OUTPUT ARGUMENTS     !
  !--------------------------!
  ! Inverted matrix to be output
  COMPLEX(KIND=8), DIMENSION(N_in), INTENT(INOUT)       :: SS_out

  !--------------------------!
  !     SUBROUTINE STUFF     !
  !--------------------------!
  ! Input evolution matrix
  COMPLEX(KIND=8), DIMENSION(N_in, N_in)                :: MatrixInverse

  !============================================================================!
  !                         END OF VARIABLE DELCARATION                        !
  !============================================================================!
  ! Set the matrix to be inverted
  MatrixInverse = Matrix_in

  ! Invert the matrix
  CALL SquareMatrixInverse(N_IN, MatrixInverse)

  ! Calculate steady states
  SS_out = 0.0d0
  SS_out = -MATMUL(MatrixInverse, Bvec_in)

END SUBROUTINE MatrixInverseSS

! Subroutine to calculate the steady states for the atom-filter operator moments
SUBROUTINE SteadyStateMoments(Gamma_in, Omega_in, alpha_in, delta_in, xi_in, &
                            & sigma_out)

  !============================================================================!
  !                   DEFINING AND DECLARING VARIABLES/ARRAYS                  !
  !============================================================================!

  IMPLICIT NONE

  !---------------!
  !     INPUT     !
  !---------------!
  ! Atomic decay rate
  REAL(KIND=8), INTENT(IN)                 :: Gamma_in
  ! Driving amplitude
  REAL(KIND=8), INTENT(IN)                 :: Omega_in
  ! Atomic anharmonicity
  REAL(KIND=8), INTENT(IN)                 :: alpha_in
  ! Drive detuning from two-photon resonance
  REAL(KIND=8), INTENT(IN)                 :: delta_in
  ! Dipole moment ratio
  REAL(KIND=8), INTENT(IN)                 :: xi_in

  !------------------------------------!
  !     MOMENT EQUATION ARRAY STUFF    !
  !------------------------------------!
  ! Dimension of M matrix
  INTEGER, PARAMETER                       :: N_mat = 8
  ! M matrix (filled as transpose)
  COMPLEX(KIND=8), DIMENSION(N_mat, N_mat) :: Mat_OG
  ! Non-homogeneous vector
  COMPLEX(KIND=8), DIMENSION(N_mat)        :: B_OG

  ! Integer indices for sigma operators
  INTEGER, PARAMETER                       :: gg = 1, ge = 2, eg = 3
  INTEGER, PARAMETER                       :: ee = 4, ef = 5, fe = 6
  INTEGER, PARAMETER                       :: gf = 7, fg = 8

  !----------------!
  !     OUTPUT     !
  !----------------!
  ! Steady state arrays
  ! First-order moments: Atomic equations (< \sigma >)
  COMPLEX(KIND=8), DIMENSION(N_mat), INTENT(OUT) :: sigma_out

  !----------------------------!
  !     OTHER USEFUL STUFF     !
  !----------------------------!
  ! Imaginary i
  COMPLEX(KIND=8), PARAMETER               :: i = CMPLX(0.0d0, 1.0d0, 8)
  ! Complex temporary values
  COMPLEX(KIND=8)                          :: moment_out

  !============================================================================!
  !               DEFINING ANALYTIC MATRICES/EIGENVALUES/VECTORS               !
  !============================================================================!
  !------------------------!
  !     BLOCH MATRIX M     !
  !------------------------!
  Mat_OG = 0.0d0
  ! Row 1: d/dt |g><g|
  Mat_OG(1, 2) = -i * 0.5d0 * Omega_in
  Mat_OG(1, 3) = i * 0.5d0 * Omega_in
  Mat_OG(1, 4) = Gamma_in
  ! Row 2: d/dt |g><e|
  Mat_OG(2, 1) = -i * 0.5d0 * Omega_in
  Mat_OG(2, 2) = -(0.5d0 * Gamma_in - i * ((0.5d0 * alpha_in) + delta_in))
  Mat_OG(2, 4) = i * 0.5d0 * Omega_in
  Mat_OG(2, 5) = Gamma_in * xi_in
  Mat_OG(2, 7) = -i * xi_in * 0.5d0 * Omega_in
  ! Row 3: d/dt |e><g|
  Mat_OG(3, 1) = i * 0.5d0 * Omega_in
  Mat_OG(3, 3) = -(0.5d0 * Gamma_in + i * ((0.5d0 * alpha_in) + delta_in))
  Mat_OG(3, 4) = -i * 0.5d0 * Omega_in
  Mat_OG(3, 6) = Gamma_in * xi_in
  Mat_OG(3, 8) = i * xi_in * 0.5d0 * Omega_in
  ! Row 4: d/dt |e><e|
  Mat_OG(4, 1) = -Gamma_in * (xi_in ** 2)
  Mat_OG(4, 2) = i * 0.5d0 * Omega_in
  Mat_OG(4, 3) = -i * 0.5d0 * Omega_in
  Mat_OG(4, 4) = -Gamma_in * (1.0d0 + (xi_in ** 2))
  Mat_OG(4, 5) = -i * xi_in * 0.5d0 * Omega_in
  Mat_OG(4, 6) = i * xi_in * 0.5d0 * Omega_in
  ! Row 5: d/dt |e><f|
  Mat_OG(5, 1) = -i * xi_in * 0.5d0 * Omega_in
  Mat_OG(5, 4) = -i * xi_in * Omega_in
  Mat_OG(5, 5) = -(0.5d0 * Gamma_in * (1.0d0 + (xi_in ** 2)) + i * ((0.5d0 * alpha_in) - delta_in))
  Mat_OG(5, 7) = i * 0.5d0 * Omega_in
  ! Row 6: d/dt |f><e|
  Mat_OG(6, 1) = i * xi_in * 0.5d0 * Omega_in
  Mat_OG(6, 4) = i * xi_in * Omega_in
  Mat_OG(6, 6) = -(0.5d0 * Gamma_in * (1.0d0 + (xi_in ** 2)) - i * ((0.5d0 * alpha_in) - delta_in))
  Mat_OG(6, 8) = -i * 0.5d0 * Omega_in
  ! Row 7: d/dt |g><f|
  Mat_OG(7, 2) = -i * xi_in * 0.5d0 * Omega_in
  Mat_OG(7, 5) = i * 0.5d0 * Omega_in
  Mat_OG(7, 7) = -(0.5d0 * Gamma_in * (xi_in ** 2) - 2.0d0 * i * delta_in)
  ! Row 8: d/dt |g><f|
  Mat_OG(8, 3) = i * xi_in * 0.5d0 * Omega_in
  Mat_OG(8, 6) = -i * 0.5d0 * Omega_in
  Mat_OG(8, 8) = -(0.5d0 * Gamma_in * (xi_in ** 2) + 2.0d0 * i * delta_in)

  !--------------------------------!
  !     NON-HOMOGENEOUS VECTOR     !
  !--------------------------------!
  B_OG = 0.0d0
  B_OG(4) = Gamma_in * (xi_in ** 2)
  B_OG(5) = i * xi_in * 0.5d0 * Omega_in
  B_OG(6) = -i * xi_in * 0.5d0 * Omega_in

  !============================================================================!
  !                       CALCULATE STEADY-STATE MOMENTS                       !
  !============================================================================!
  !---------------------------!
  !     FIRST-ORDER: ATOM     !
  !---------------------------!
  IF (xi_in .NE. 0.0d0) THEN
    ! Calculate steady states
    CALL MatrixInverseSS(N_mat, Mat_OG, B_OG, sigma_out)

  ELSE IF (xi_in .EQ. 0.0d0) THEN
    !------------------------------------------------!
    !     CALCULATE EIGENVALUES AND EIGENVECTORS     !
    !------------------------------------------------!
    ! Calculate steady state from eigenvectors
    CALL SquareMatrixZeroEigenvalue(N_mat, Mat_OG, sigma_out)
    ! Normalise sigma_out so |g><g| + |e><e| = 1
    sigma_out = sigma_out / (REAL(sigma_out(1)) + REAL(sigma_out(4)))

  END IF

END SUBROUTINE SteadyStateMoments

!==============================================================================!
!                          G1 CORRELATION SUBROUTINES                          !
!==============================================================================!
! Subroutine to calculate the initial conditions for the auto-correlations
SUBROUTINE G1_InitialConditions(Gamma_in, Omega_in, alpha_in, delta_in, xi_in, &
                              & sigma_out, B_OG_out, sigmapm_ss_out)

  !============================================================================!
  !                   DEFINING AND DECLARING VARIABLES/ARRAYS                  !
  !============================================================================!

  IMPLICIT NONE

  !---------------!
  !     INPUT     !
  !---------------!
  ! Atomic decay rate
  REAL(KIND=8), INTENT(IN)                   :: Gamma_in
  ! Driving amplitude
  REAL(KIND=8), INTENT(IN)                   :: Omega_in
  ! Atomic anharmonicity
  REAL(KIND=8), INTENT(IN)                   :: alpha_in
  ! Drive detuning from two-photon resonance
  REAL(KIND=8), INTENT(IN)                   :: delta_in
  ! Dipole moment ratio
  REAL(KIND=8), INTENT(IN)                   :: xi_in

  !------------------------------------!
  !     MOMENT EQUATION ARRAY STUFF    !
  !------------------------------------!
  ! Dimension of M matrix
  INTEGER, PARAMETER                         :: N_mat = 8
  ! M matrix (filled as transpose)
  COMPLEX(KIND=8), DIMENSION(N_mat, N_mat)   :: Mat, Mat_OG

  ! Integer indices for sigma operators
  INTEGER, PARAMETER                         :: gg = 1, ge = 2, eg = 3
  INTEGER, PARAMETER                         :: ee = 4, ef = 5, fe = 6
  INTEGER, PARAMETER                         :: gf = 7, fg = 8

  ! Steady state arrays
  ! First-order moments: Atomic equations (< \sigma >)
  COMPLEX(KIND=8), DIMENSION(N_mat)              :: sigma_ss

  !----------------!
  !     OUTPUT     !
  !----------------!
  ! First-order moments: Atomic equations (< \sigma >)
  COMPLEX(KIND=8), DIMENSION(N_mat), INTENT(OUT) :: sigma_out
  ! Steady state photon number
  REAL(KIND=8), INTENT(OUT)                      :: sigmapm_ss_out
  ! Non-homogeneous vector
  COMPLEX(KIND=8), DIMENSION(N_mat), INTENT(OUT) :: B_OG_out

  !----------------------------!
  !     OTHER USEFUL STUFF     !
  !----------------------------!
  ! Steady state (< \Sigma_{-} >_{ss})
  COMPLEX(KIND=8)                       :: sigmam_ss
  ! Imaginary i
  COMPLEX(KIND=8), PARAMETER            :: i = CMPLX(0.0d0, 1.0d0, 8)

  !============================================================================!
  !                       CALCULATE STEADY-STATE MOMENTS                       !
  !============================================================================!
  CALL SteadyStateMoments(Gamma_in, Omega_in, alpha_in, delta_in, xi_in, &
                        & sigma_ss)

  !============================================================================!
  !        CALCULATE FIRST-ORDER CORRELATION FUNCTION INITIAL CONDITIONS       !
  !============================================================================!
  ! Set initial conditions and non-homogeneous vector
  ! < \Sigma_{+} (\tau = 0) \Sigma_{-} (0) > = < \Sigma_{+} \Sigma_{-} >_ss

  ! Steady state value < \Sigma_{+} \Sigma_{-} >
  sigmapm_ss_out = REAL(sigma_ss(ee) + (xi_in ** 2) * (1 - sigma_ss(gg) - sigma_ss(ee)))
  ! < \Sigma_{-} >_{ss}
  sigmam_ss = sigma_ss(ge) + xi_in * sigma_ss(ef)

  ! Initial conditions
  sigma_out = 0.0d0
  ! |g><e| \rho
  sigma_out(gg) = sigma_ss(ge)
  sigma_out(ge) = sigma_ss(ee)
  sigma_out(gf) = sigma_ss(ef)
  ! \xi |e><f| \rho
  sigma_out(eg) = xi_in * sigma_ss(fg)
  sigma_out(ee) = xi_in * sigma_ss(fe)
  sigma_out(ef) = xi_in * (1 - sigma_ss(gg) - sigma_ss(ee))
  
  ! non homogeneous vector
  B_OG_out = 0.0d0
  B_OG_out(4) = B_OG_out(4) + Gamma_in * (xi_in ** 2) * sigmam_ss
  B_OG_out(5) = B_OG_out(5) + i * xi_in * 0.5d0 * Omega_in * sigmam_ss
  B_OG_out(6) = B_OG_out(6) - i * xi_in * 0.5d0 * Omega_in * sigmam_ss

END SUBROUTINE G1_InitialConditions

! Subroutine to calculate the time evolution of the g2 correlation
SUBROUTINE G1_CalculateRK4(Gamma_in, Omega_in, alpha_in, delta_in, xi_in, &
                         & dt_in, tau_steps_in, &
                         & g1_array_out, WRITE_DATA_IN, filename_data_in)

  !============================================================================!
  !                   DEFINING AND DECLARING VARIABLES/ARRAYS                  !
  !============================================================================!

  IMPLICIT NONE

  !---------------!
  !     INPUT     !
  !---------------!
  ! Atomic decay rate
  REAL(KIND=8), INTENT(IN)                  :: Gamma_in
  ! Driving amplitude
  REAL(KIND=8), INTENT(IN)                  :: Omega_in
  ! Atomic anharmonicity
  REAL(KIND=8), INTENT(IN)                  :: alpha_in
  ! Drive detuning from two-photon resonance
  REAL(KIND=8), INTENT(IN)                  :: delta_in
  ! Dipole moment ratio
  REAL(KIND=8), INTENT(IN)                  :: xi_in

  ! Time stuff
  ! Time step
  REAL(KIND=8), INTENT(IN)                  :: dt_in
  ! Maxi_inmum number of steps to integrate for
  INTEGER, INTENT(IN)                       :: tau_steps_in

  ! Data stuff
  ! Boolean for writing data
  LOGICAL, INTENT(IN)                       :: WRITE_DATA_IN
  ! Filename for writing data to
  CHARACTER(LEN=*), INTENT(IN)              :: filename_data_in

  !----------------!
  !     OUTPUT     !
  !----------------!
  ! Data array
  COMPLEX(KIND=8), DIMENSION(:), ALLOCATABLE, INTENT(OUT) :: g1_array_out

  !------------------------------------!
  !     MOMENT EQUATION ARRAY STUFF    !
  !------------------------------------!
  ! Dimension of M matrix
  INTEGER, PARAMETER                        :: N_mat = 8
  ! M matrix (filled as transpose)
  COMPLEX(KIND=8), DIMENSION(N_mat, N_mat)  :: Mat_OG, Mat
  ! Non-homogeneous vector
  COMPLEX(KIND=8), DIMENSION(N_mat)         :: B_OG, B_vec

  ! Integer indices for sigma operators
  INTEGER, PARAMETER                        :: gg = 1, ge = 2, eg = 3
  INTEGER, PARAMETER                        :: ee = 4, ef = 5, fe = 6
  INTEGER, PARAMETER                        :: gf = 7, fg = 8

  ! Time integration arrays
  ! First-order moments: Atomic equations (< \sigma >)
  COMPLEX(KIND=8), DIMENSION(N_mat)         :: sigma
  COMPLEX(KIND=8), DIMENSION(N_mat)         :: k1_sigma, k2_sigma, k3_sigma, k4_sigma

  !----------------------------!
  !     OTHER USEFUL STUFF     !
  !----------------------------!
  ! Steady state value of < \Sigma_{+} \Sigma_{-} >
  REAL(KIND=8)                              :: sigmapm_ss
  ! Time step integer
  INTEGER                                   :: t
  ! Sample rate for state populations
  INTEGER                                   :: sample_rate
  ! Imaginary i
  COMPLEX(KIND=8), PARAMETER                :: i = CMPLX(0.0d0, 1.0d0, 8)
  ! 1 / 6
  REAL(KIND=8), PARAMETER                   :: xis = 1.0d0 / 6.0d0
  ! Complex data
  COMPLEX(KIND=8)                           :: moment_out

  !============================================================================!
  !               DEFINING ANALYTIC MATRICES/EIGENVALUES/VECTORS               !
  !============================================================================!
  !------------------------!
  !     BLOCH MATRIX M     !
  !------------------------!
  Mat_OG = 0.0d0
  ! Row 1: d/dt |g><g|
  Mat_OG(1, 2) = -i * 0.5d0 * Omega_in
  Mat_OG(1, 3) = i * 0.5d0 * Omega_in
  Mat_OG(1, 4) = Gamma_in
  ! Row 2: d/dt |g><e|
  Mat_OG(2, 1) = -i * 0.5d0 * Omega_in
  Mat_OG(2, 2) = -(0.5d0 * Gamma_in - i * ((0.5d0 * alpha_in) + delta_in))
  Mat_OG(2, 4) = i * 0.5d0 * Omega_in
  Mat_OG(2, 5) = Gamma_in * xi_in
  Mat_OG(2, 7) = -i * xi_in * 0.5d0 * Omega_in
  ! Row 3: d/dt |e><g|
  Mat_OG(3, 1) = i * 0.5d0 * Omega_in
  Mat_OG(3, 3) = -(0.5d0 * Gamma_in + i * ((0.5d0 * alpha_in) + delta_in))
  Mat_OG(3, 4) = -i * 0.5d0 * Omega_in
  Mat_OG(3, 6) = Gamma_in * xi_in
  Mat_OG(3, 8) = i * xi_in * 0.5d0 * Omega_in
  ! Row 4: d/dt |e><e|
  Mat_OG(4, 1) = -Gamma_in * (xi_in ** 2)
  Mat_OG(4, 2) = i * 0.5d0 * Omega_in
  Mat_OG(4, 3) = -i * 0.5d0 * Omega_in
  Mat_OG(4, 4) = -Gamma_in * (1.0d0 + (xi_in ** 2))
  Mat_OG(4, 5) = -i * xi_in * 0.5d0 * Omega_in
  Mat_OG(4, 6) = i * xi_in * 0.5d0 * Omega_in
  ! Row 5: d/dt |e><f|
  Mat_OG(5, 1) = -i * xi_in * 0.5d0 * Omega_in
  Mat_OG(5, 4) = -i * xi_in * Omega_in
  Mat_OG(5, 5) = -(0.5d0 * Gamma_in * (1.0d0 + (xi_in ** 2)) + i * ((0.5d0 * alpha_in) - delta_in))
  Mat_OG(5, 7) = i * 0.5d0 * Omega_in
  ! Row 6: d/dt |f><e|
  Mat_OG(6, 1) = i * xi_in * 0.5d0 * Omega_in
  Mat_OG(6, 4) = i * xi_in * Omega_in
  Mat_OG(6, 6) = -(0.5d0 * Gamma_in * (1.0d0 + (xi_in ** 2)) - i * ((0.5d0 * alpha_in) - delta_in))
  Mat_OG(6, 8) = -i * 0.5d0 * Omega_in
  ! Row 7: d/dt |g><f|
  Mat_OG(7, 2) = -i * xi_in * 0.5d0 * Omega_in
  Mat_OG(7, 5) = i * 0.5d0 * Omega_in
  Mat_OG(7, 7) = -(0.5d0 * Gamma_in * (xi_in ** 2) - 2.0d0 * i * delta_in)
  ! Row 8: d/dt |g><f|
  Mat_OG(8, 3) = i * xi_in * 0.5d0 * Omega_in
  Mat_OG(8, 6) = -i * 0.5d0 * Omega_in
  Mat_OG(8, 8) = -(0.5d0 * Gamma_in * (xi_in ** 2) + 2.0d0 * i * delta_in)

  !--------------------------------!
  !     NON-HOMOGENEOUS VECTOR     !
  !--------------------------------!
  B_OG = 0.0d0
  B_OG(4) = Gamma_in * (xi_in ** 2)
  B_OG(5) = i * xi_in * 0.5d0 * Omega_in
  B_OG(6) = -i * xi_in * 0.5d0 * Omega_in

  !------------------------------------------!
  !     INITALISE OPERATOR MOMENT ARRAYS     !
  !------------------------------------------!
  ! Data
  ALLOCATE(g1_array_out(0:tau_steps_in)); g1_array_out = 0.0d0

  !============================================================================!
  !                        CALCULATE INITIAL CONDITIONS                        !
  !============================================================================!
  CALL G1_InitialConditions(Gamma_in, Omega_in, alpha_in, delta_in, xi_in, &
                          & sigma, B_OG, sigmapm_ss)

  !============================================================================!
  !                 CALCULATE SECOND-ORDER CORRELATION FUNCTION                !
  !============================================================================!
  ! Calculate the sample rate for writing data to the file
  IF (tau_steps_in > 100000) THEN
    sample_rate = NINT(DBLE(tau_steps_in) / 1d5)
  ELSE
    sample_rate = 1
  END IF

  ! If WRITE_DATA_IN is TRUE, open file to write data to
  IF (WRITE_DATA_IN .EQV. .TRUE.) THEN
    ! Open file to write time and data to
    OPEN(UNIT=4, FILE=filename_data_in, STATUS='REPLACE', ACTION='WRITE', RECL=4000)
  END IF

  ! Cycle through time steps
  DO t = 0, tau_steps_in
    !============================================================================!
    !                          CALCULATE AND WRITE DATA                          !
    !============================================================================!
    !-------------------------------!
    !     CALCULATE DATA STATES     !
    !-------------------------------!
    ! Grab correlation value
    moment_out = 0.0d0
    moment_out = sigma(ge) + xi_in * sigma(ef)

    ! Normalise correlation by steady-state photon number
    IF (sigmapm_ss .NE. 0.0) THEN
      moment_out = moment_out / (sigmapm_ss)
    END IF

    !-----------------------!
    !     WRITE TO FILE     !
    !-----------------------!
    g1_array_out(t) = MOMENT_OUT
    ! Second-order correlation function
    ! If WRITE_DATA_IN is TRUE, write data to file
    IF (WRITE_DATA_IN .EQV. .TRUE.) THEN
      ! If t_max is really big, only take a sample of results to write to file
      ! so file size isn't huge-mongous.
      IF (MOD(t, sample_rate) == 0) THEN
        WRITE(4, *) dt_in * DBLE(t), REAL(moment_out), AIMAG(moment_out)
      END IF
    END IF

    !============================================================================!
    !                  CALCULATE USING FOURTH-ORDER RUNGE-KUTTA                  !
    !============================================================================!
    !----------------------------------------!
    !     INITIALISE RUNGE-KUTTA VECTORS     !
    !----------------------------------------!
    k1_sigma = 0.0d0; k2_sigma = 0.0d0; k3_sigma = 0.0d0; k4_sigma = 0.0d0

    !---------------------------!
    !     FIRST-ORDER: ATOM     !
    !---------------------------!
    ! Calculate Runge-Kutta vectors
    k1_sigma = dt_in * (MATMUL(Mat_OG, sigma) + B_OG)
    k2_sigma = dt_in * (MATMUL(Mat_OG, (sigma + 0.5d0 * k1_sigma)) + B_OG)
    k3_sigma = dt_in * (MATMUL(Mat_OG, (sigma + 0.5d0 * k2_sigma)) + B_OG)
    k4_sigma = dt_in * (MATMUL(Mat_OG, (sigma + k3_sigma)) + B_OG)

    !============================================================================!
    !                   UPDATE ARRAYS FROM RUNGE-KUTTA ARRAYS                    !
    !============================================================================!
    ! First-order
    sigma = sigma + xis * (k1_sigma + 2.0d0 * (k2_sigma + k3_sigma) + k4_sigma)

    ! Close t loop
  END DO

  ! If WRITE_DATA_IN is TRUE, close the file
  IF (WRITE_DATA_IN .EQV. .TRUE.) THEN
    ! Close file
    CLOSE(4)
  END IF

END SUBROUTINE G1_CalculateRK4

END MODULE ATOM_SUBROUTINES