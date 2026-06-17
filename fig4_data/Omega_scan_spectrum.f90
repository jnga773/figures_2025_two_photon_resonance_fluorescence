!==============================================================================!
!                                                                              !
!  Scans the driving amplitude (Rabi frequency) Omega over a specified range   !
!  and, for each value, integrates the moment equations of a driven            !
!  three-level atom (via 4th-order Runge-Kutta, using the routines in          !
!  moment_equations_3LA_functions.f90) to compute the first-order              !
!  correlation function g1(tau). The resulting g1(tau) curves are assembled    !
!  into a matrix (rows = time steps tau, columns = Omega values) -- the data   !
!  needed to build a resonance fluorescence spectrum as a function of          !
!  driving strength.                                                           !
!                                                                              !
!  Input:                                                                      !
!    A NAMELIST file containing the ATOM, SCANPARAMS, and TIME groups          !
!    (system parameters, Omega scan range, and integration time step/          !
!    duration respectively). Its path, along with the output data              !
!    directory, can be set via command-line flags:                             !
!      --data-dir=<path>    (default: ./data_files/)                           !
!      --name-list=<path>   (default: ./ParamList.nml)                         !
!                                                                              !
!  Output (written to the data directory):                                     !
!    g1_parameters.txt   -- system/scan/time parameters used for the run       !
!    g1_corr_real.txt    -- real part of g1(tau), one column per Omega         !
!    g1_corr_imag.txt    -- imaginary part of g1(tau), one column per Omega    !
!                                                                              !
!==============================================================================!
 
 PROGRAM OMEGA_SCAN_SPECTRUM

! Import subroutines from the module file
USE MOMENT_EQUATIONS_3LA_FUNCTIONS

!==============================================================================!
!                    DEFINING AND DECLARING VARIABLES/ARRAYS                   !
!==============================================================================!

IMPLICIT NONE

!---------------------------------!
!     SYSTEM PARAMETERS STUFF     !
!---------------------------------!
! Atomic decay rate
REAL(dp)                                  :: Gamma
! Driving amplitude
REAL(dp)                                  :: Omega
! Atomic anharmonicity
REAL(dp)                                  :: alpha
! Drive detuning from two-photon resonance
REAL(dp)                                  :: delta
! Dipole moment ratio
REAL(dp)                                  :: xi, xi_squared

!-------------------------------!
!     TIME PARAMETERS STUFF     !
!-------------------------------!
! Time step
REAL(dp)                                  :: dt
! Maximum time to integrate for
REAL(dp)                                  :: tau1_max, tau2_max
! Maximum number of steps to integrate for
INTEGER                                   :: tau_steps
! Runtime variables
REAL(dp)                                  :: start_time, end_time

!-------------------------------!
!     SCAN PARAMETERS STUFF     !
!-------------------------------!
! Starting scan value
REAL(dp)                                  :: scan_start
! Final scan value
REAL(dp)                                  :: scan_end
! Scan step size
REAL(dp)                                  :: scan_step
! Number of scan steps
INTEGER                                   :: number_of_scans

!--------------------!
!     DATA STUFF     !
!--------------------!
! Array of Omega values to scan over
REAL(dp), DIMENSION(:), ALLOCATABLE       :: Omega_array
! Scan variable
REAL(dp)                                  :: Omega_scan
! Run counter
INTEGER                                   :: run_counter

! Correlation data
COMPLEX(dp), DIMENSION(:), ALLOCATABLE    :: g1_array
! Matrix of correlation values
COMPLEX(dp), DIMENSION(:, :), ALLOCATABLE :: corr_matrix
! Index integer
INTEGER                                   :: idx

!------------------------!
!     FILENAME STUFF     !
!------------------------!
! Parameter Name List
CHARACTER(:), ALLOCATABLE                 :: filename_NameList
! Data subdirectory name
CHARACTER(:), ALLOCATABLE                 :: data_directory
! Filename of parameters
CHARACTER(LEN=99)                         :: filename_parameters = 'g1_parameters.txt'
! Filename for first-order correlation
CHARACTER(LEN=99)                         :: filename_g1_real = 'g1_corr_real.txt'
CHARACTER(LEN=99)                         :: filename_g1_imag = 'g1_corr_imag.txt'

! Read status integer
INTEGER                                   :: ISTAT
! NameList file unit integer
INTEGER                                   :: IUNIT_NML = 420
! Line to be read from file
CHARACTER(LEN=512)                        :: LINE

!==============================================================================!
!                      DEFINE PARAMETERS IN NAMELIST FILE                      !
!==============================================================================!
! Namelist parameters
NAMELIST /ATOM/ Gamma, Omega, alpha, delta, xi_squared
NAMELIST /SCANPARAMS/ scan_start, scan_end, scan_step
NAMELIST /TIME/ dt, tau1_max, tau2_max

!==============================================================================!
!                             INITIALISE RUN TIME                              !
!==============================================================================!
! Call start time from CPU_TIME
CALL CPU_TIME(start_time)

!==============================================================================!
!               PARSE INPUT FOR DATA DIRECTORY AND NAMELIST FILE               !
!==============================================================================!
! Parse input arguments for data_directory and filename_NameList
CALL PARSE_INPUT_ARGUMENTS(data_directory, filename_NameList)

! Print data directory and NameList file
WRITE(*, *) 'Reading parameters from: ' // TRIM(filename_NameList)
WRITE(*, *) ' Save data to directory: ' // TRIM(data_directory)

!==============================================================================!
!                      READ PARAMETERS FROM NAMELIST FILE                      !
!==============================================================================!
! Read the parameters from the NAMELIST file
OPEN(IUNIT_NML, FILE=filename_NameList, STATUS="OLD", DELIM="QUOTE")

! Read the 'ATOM' name list
READ(IUNIT_NML, NML=ATOM, IOSTAT=ISTAT)
IF (ISTAT .NE. 0) THEN
  BACKSPACE(IUNIT_NML)
  READ(IUNIT_NML, FMT='(A)') LINE
  CLOSE(IUNIT_NML)
  PRINT *, "Invalid line in ATOM namelist: " // TRIM(line)
  CALL EXIT(1)
END IF

! Read the 'SCANPARAMS' name list
READ(IUNIT_NML, NML=SCANPARAMS, IOSTAT=ISTAT)
IF (ISTAT .NE. 0) THEN
  BACKSPACE(IUNIT_NML)
  READ(IUNIT_NML, FMT='(A)') LINE
  CLOSE(IUNIT_NML)
  PRINT *, "Invalid line in SCANPARAMS namelist: " // TRIM(line)
  CALL EXIT(1)
END IF

! Read the 'TIME' name list
READ(IUNIT_NML, NML=TIME, IOSTAT=ISTAT)
IF (ISTAT .NE. 0) THEN
  BACKSPACE(IUNIT_NML)
  READ(IUNIT_NML, FMT='(A)') LINE
  CLOSE(IUNIT_NML)
  PRINT *, "Invalid line in TIME namelist: " // TRIM(line)
  CALL EXIT(1)
END IF

! Close file
CLOSE(IUNIT_NML)

!---------------------------!
!     Format Parameters     !
!---------------------------!
! Number of time-steps
tau_steps = NINT(tau1_max / dt)

! Number of scan steps
number_of_scans = NINT((scan_end - scan_start) / scan_step)

! Set xi
xi = SQRT(xi_squared)

!==============================================================================!
!                         ALLOCATING ARRAYS AND STUFF                          !
!==============================================================================!
! Set halfwidth array
ALLOCATE(Omega_array(0:number_of_scans))
Omega_array = 0.0_dp
! Set values
DO idx = 0, number_of_scans
  Omega_array(idx) = scan_start + DBLE(idx) * scan_step
END DO

! Allocate data matrix
ALLOCATE(corr_matrix(0:tau_steps, 0:number_of_scans))
corr_matrix = 0.0_dp

!==============================================================================!
!                          CREATE DATA SUBDIRECTORIES                          !
!==============================================================================!
! ! Create folder for data files
! IF (xi .EQ. SQRT(0.5_dp)) THEN
!   data_directory = TRIM(data_directory) // 'scan_xi_1_over_root_2/'
! ELSE IF (xi .EQ. 1.0_dp) THEN
!   data_directory = TRIM(data_directory) //' scan_xi_1/'
! ELSE IF (xi .EQ. SQRT(2.0_dp)) THEN
!   data_directory = TRIM(data_directory) // 'scan_xi_root_2/'
! END IF

! Create data directory
CALL EXECUTE_COMMAND_LINE('mkdir -p ' // TRIM(data_directory))

! Set filenames
filename_parameters = TRIM(data_directory) // TRIM(filename_parameters)
filename_g1_real    = TRIM(data_directory) // TRIM(filename_g1_real)
filename_g1_imag    = TRIM(data_directory) // TRIM(filename_g1_imag)

!==============================================================================!
!                           WRITE PARAMETERS TO FILE                           !
!==============================================================================!
! Open file to write time to
OPEN(UNIT=1, FILE=filename_parameters, STATUS='REPLACE', ACTION='WRITE')

! Write parameters
WRITE(1,"(A15,F25.15)") "Gamma =", Gamma
! WRITE(1,"(A15,F25.15)") "Omega =", Omega
WRITE(1,"(A15,F25.15)") "alpha =", alpha
WRITE(1,"(A15,F25.15)") "delta =", delta
WRITE(1,"(A15,F25.15)") "xi =", xi

WRITE(1,"(A15,F25.15)") "dt =", dt
! WRITE(1,"(A15,F25.15)") "Max t =", t_max
WRITE(1,"(A15,F25.15)") "Max tau1 =", tau1_max
! WRITE(1,"(A15,F25.15)") "Max tau2 =", tau2_max

WRITE(1, *) " "
WRITE(1, *) "Omega Scan Values"
DO idx = 0, number_of_scans
  WRITE(1, "(A15, F25.15)") "Omega =", Omega_array(idx)
END DO

! Close file
CLOSE(1)

!===============================================================================!
!                  CALCULATE SECOND-ORDER CORRELATION FUNCTION                  !
!===============================================================================!
! Reset run_counter
run_counter = 0

! Set OMP clauses
!$OMP PARALLEL DO PRIVATE(idx, Omega_scan, g1_array) SHARED(run_counter)

! Cycle through halfwidth values
DO idx = 0, number_of_scans
  ! Grab Omega value
  Omega_scan = Omega_array(idx)

  ! Calculate g1
  CALL G1_CalculateRK4(Gamma, Omega_scan, alpha, delta, xi, &
                       dt, tau_steps, &
                       g1_array, .FALSE., "NONE")

  ! Save data to matrix
  corr_matrix(:, idx) = g1_array

  ! Deallocate the data array
  DEALLOCATE(g1_array)

  ! Print completion
  WRITE(*, "(I4, A3, I4, A15)") run_counter+1, " / ", number_of_scans+1, " scans complete"
  run_counter = run_counter + 1

  ! Close DO loop
END DO

!==============================================================================!
!                              WRITE DATA TO FILE                              !
!==============================================================================!
! Open file to write data to
OPEN(UNIT=2, FILE=filename_g1_real, STATUS='REPLACE', ACTION='WRITE', RECL=32000)
OPEN(UNIT=3, FILE=filename_g1_imag, STATUS='REPLACE', ACTION='WRITE', RECL=32000)

! Unformatted binary files
! OPEN(UNIT=2, FILE="./data_files/scan/g1_corr_real.dat", STATUS='REPLACE', FORM='UNFORMATTED')
! OPEN(UNIT=3, FILE="./data_files/scan/g1_corr_imag.dat", STATUS='REPLACE', FORM='UNFORMATTED')

! Scan through time steps
DO idx = 0, tau_steps
  ! Write to file by g1 point.
  ! Each row is a data point, each column is a scan
  WRITE(2, *) REAL(corr_matrix(idx, :))
  WRITE(3, *) AIMAG(corr_matrix(idx, :))

  ! WRITE(2) REAL(corr_matrix(idx, :))
  ! WRITE(3) AIMAG(corr_matrix(idx, :))
END DO

! Close files
CLOSE(2)
CLOSE(3)

!==============================================================================!
!                                END OF PROGRAM                                !
!==============================================================================!
! Call end time from CPU_TIME
CALL CPU_TIME(end_time)
PRINT*, "Runtime: ", end_time - start_time, "seconds"

!==============================================================================!
!                                 SUBROUTINES                                  !
!==============================================================================!
CONTAINS

SUBROUTINE PARSE_INPUT_ARGUMENTS(data_dir, filename_NameList)
  ! Parses command-line arguments to set the data output directory and
  ! parameter namelist filepath. Accepts flags with or without the '--'
  ! prefix (e.g. --data-dir=./out/ or data-dir=./out/). Applies
  ! normalisation: ensures data_dir ends with '/' and that both paths
  ! are prepended with './' if they are not absolute or parent-relative.
  ! Falls back to './data_files/' and './NameList.nml' if the
  ! corresponding flag is not supplied.

  !============================================================================!
  !                      DEFINING AND DECLARING VARIABLES                      !
  !============================================================================!
  !----------------!
  !     OUTPUT     !
  !----------------!
  ! Data directory to save (or read) data from
  CHARACTER(:), ALLOCATABLE, INTENT(OUT) :: data_dir
  ! Filename of NameList to read parameters from
  CHARACTER(:), ALLOCATABLE, INTENT(OUT) :: filename_NameList

  !----------------------!
  !     OTHER THINGS     !
  !----------------------!
  ! Number of arguments
  INTEGER                                 :: N_args, idx
  ! Input argument
  CHARACTER(LEN=256)                      :: arg_char
  ! Logical check for parameter input
  LOGICAL                                 :: found_data_dir, found_NameList

  !============================================================================!
  !                          CALCULATE FUNCTION OUTPUT                         !
  !============================================================================!
  !--------------------!
  !     Read Input     !
  !--------------------!
  found_data_dir   = .FALSE.
  found_NameList = .FALSE.
  N_args = COMMAND_ARGUMENT_COUNT()

  DO idx = 1, N_args
    CALL GET_COMMAND_ARGUMENT(idx, arg_char)

    IF (INDEX(arg_char, '--data-dir=') == 1) THEN
      data_dir = TRIM(arg_char(12:))
      found_data_dir = .TRUE.

    ELSE IF (INDEX(arg_char, 'data-dir=') == 1) THEN
      data_dir = TRIM(arg_char(10:))
      found_data_dir = .TRUE.

    ELSE IF (INDEX(arg_char, '--name-list=') == 1) THEN
      filename_NameList = TRIM(arg_char(13:))
      found_NameList = .TRUE.

    ELSE IF (INDEX(arg_char, 'name-list=') == 1) THEN
      filename_NameList = TRIM(arg_char(11:))
      found_NameList = .TRUE.

    END IF
  END DO

  !------------------------!
  !     Default Option     !
  !------------------------!
  IF (.NOT. found_data_dir) THEN
    data_dir = './data_files/'
  END IF

  IF (.NOT. found_NameList) THEN
    filename_NameList = './ParamList.nml'
  END IF

  !------------------------------!
  !     Normalise 'data_dir'     !
  !------------------------------!
  ! Append trailing '/' if missing
  IF (data_dir(LEN(data_dir):LEN(data_dir)) .NE. '/') THEN
    data_dir = data_dir // '/'
  END IF

  ! Prepend './' if not absolute and not parent-relative
  IF (data_dir(1:1) .NE. '/' .AND. data_dir(1:3) .NE. '../') THEN
    IF (data_dir(1:2) .NE. './') THEN
      data_dir = './' // data_dir
    END IF
  END IF

  !--------------------------------!
  !     Normalise 'param_list'     !
  !--------------------------------!
  ! Prepend './' if not absolute and not parent-relative
  IF (filename_NameList(1:1) .NE. '/' .AND. filename_NameList(1:3) .NE. '../') THEN
    IF (filename_NameList(1:2) .NE. './') THEN
      filename_NameList = './' // filename_NameList
    END IF
  END IF

END SUBROUTINE PARSE_INPUT_ARGUMENTS

!==============================================================================!
!                                END OF PROGRAM                                !
!==============================================================================!
END PROGRAM OMEGA_SCAN_SPECTRUM
