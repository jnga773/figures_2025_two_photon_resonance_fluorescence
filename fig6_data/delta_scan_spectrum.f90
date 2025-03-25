PROGRAM THREE_LEVEL_ATOM_G1

! Import subroutines from the module file
USE ATOM_SUBROUTINES

!==============================================================================!
!                    DEFINING AND DECLARING VARIABLES/ARRAYS                   !
!==============================================================================!

IMPLICIT NONE

!---------------------------------!
!     SYSTEM PARAMETERS STUFF     !
!---------------------------------!
! Atomic decay rate
REAL(KIND=8)                                           :: Gamma
! Driving amplitude
REAL(KIND=8)                                           :: Omega
! Atomic anharmonicity
REAL(KIND=8)                                           :: alpha
! Drive detuning from two-photon resonance
REAL(KIND=8)                                           :: delta
! Dipole moment ratio
REAL(KIND=8)                                           :: xi, xi_squared

! Time stuff
! Time step
REAL(KIND=8)                                           :: dt
! Maximum time to integrate for
REAL(KIND=8)                                           :: tau1_max, tau2_max
! Maximum number of steps to integrate for
INTEGER                                                :: tau_steps
! Runtime variables
REAL(KIND=8)                                           :: start_time, end_time

! Scan parameters
! Starting scan value
REAL(KIND=8)                                           :: scan_start
! Final scan value
REAL(KIND=8)                                           :: scan_end
! Scan step size
REAL(KIND=8)                                           :: scan_step
! Number of scan steps
INTEGER                                                :: number_of_scans

! Scan stuff
! Array of delta values to scan over
REAL(KIND=8), DIMENSION(:), ALLOCATABLE                :: delta_array
! Scan variable
REAL(KIND=8)                                           :: delta_scan
! Run counter
INTEGER                                                :: run_counter


!----------------------------!
!     OTHER USEFUL STUFF     !
!----------------------------!
! Correlation data
COMPLEX(KIND=8), DIMENSION(:), ALLOCATABLE             :: g1_array
! Matrix of correlation values
COMPLEX(KIND=8), DIMENSION(:, :), ALLOCATABLE          :: corr_matrix
! Index integer
INTEGER                                                :: index

!------------------------!
!     FILENAME STUFF     !
!------------------------!
! Paramert Name List
CHARACTER(LEN=15), PARAMETER                           :: filename_ParamList = "./ParamList.nml"
! Data subdirectory name
CHARACTER(LEN=99)                                      :: data_directory
! Filename of parameters
CHARACTER(LEN=99)                                      :: filename_parameters
! Filename for first-order correlation
CHARACTER(LEN=99)                                      :: filename_g1_real
CHARACTER(LEN=99)                                      :: filename_g1_imag

!==============================================================================!
!                 NAMELIST AND PARAMETERS TO BE READ FROM FILE                 !
!==============================================================================!
! NameList things
! Status and unit integers
INTEGER            :: ISTAT, IUNIT
! Line to be read from file
CHARACTER(LEN=512) :: LINE
! Namelist parameters
NAMELIST /ATOM/ Gamma, Omega, alpha, delta, xi_squared
NAMELIST /SCANPARAMS/ scan_start, scan_end, scan_step
NAMELIST /TIME/ dt, tau1_max, tau2_max

! Call start time from CPU_TIME
CALL CPU_TIME(start_time)

! Read the parameters from the NAMELIST file
IUNIT = 420
OPEN(IUNIT, FILE=filename_ParamList, STATUS="OLD", DELIM="QUOTE")

READ(IUNIT, NML=ATOM, IOSTAT=ISTAT)
IF (ISTAT .NE. 0) THEN
  BACKSPACE(IUNIT)
  READ(IUNIT, FMT='(A)') LINE
  CLOSE(IUNIT)
  PRINT *, "Invalid line in ATOM namelist: " // TRIM(line)
  CALL EXIT(1)
END IF

READ(IUNIT, NML=SCANPARAMS, IOSTAT=ISTAT)
IF (ISTAT .NE. 0) THEN
  BACKSPACE(IUNIT)
  READ(IUNIT, FMT='(A)') LINE
  CLOSE(IUNIT)
  PRINT *, "Invalid line in SCANPARAMS namelist: " // TRIM(line)
  CALL EXIT(1)
END IF

READ(IUNIT, NML=TIME, IOSTAT=ISTAT)
IF (ISTAT .NE. 0) THEN
  BACKSPACE(IUNIT)
  READ(IUNIT, FMT='(A)') LINE
  CLOSE(IUNIT)
  PRINT *, "Invalid line in TIME namelist: " // TRIM(line)
  CALL EXIT(1)
END IF
CLOSE(IUNIT)

! Number of time-steps
tau_steps = NINT(tau1_max / dt)

! Number of scan steps
number_of_scans = NINT((scan_end - scan_start) / scan_step)

! Set xi
xi = SQRT(xi_squared)

!==============================================================================!
!                          CREATE DATA SUBDIRECTORIES                          !
!==============================================================================!
! Create folder for data files
IF (xi .EQ. SQRT(0.5d0)) THEN
  data_directory = './data_files/scan_xi_1_over_root_2/'
ELSE IF (xi .EQ. 1.0d0) THEN
  data_directory = './data_files/scan_xi_1/'
ELSE IF (xi .EQ. SQRT(2.0d0)) THEN
  data_directory = './data_files/scan_xi_root_2/'
ELSE
  data_directory = './data_files/'
END IF

! Create data directory
PRINT*, data_directory
CALL EXECUTE_COMMAND_LINE("mkdir -p " // TRIM(data_directory))

! Set filenames
filename_parameters = TRIM(data_directory) // "g1_parameters.txt"
filename_g1_real    = TRIM(data_directory) // "g1_corr_real.txt"
filename_g1_imag    = TRIM(data_directory) // "g1_corr_imag.txt"

!==============================================================================!
!                         ALLOCATING ARRAYS AND STUFF                          !
!==============================================================================!
! Set halfwidth array
ALLOCATE(delta_array(0:number_of_scans))
delta_array = 0.0d0
! Set values
DO index = 0, number_of_scans
  delta_array(index) = scan_start + DBLE(index) * scan_step
END DO

! Allocate data matrix
ALLOCATE(corr_matrix(0:tau_steps, 0:number_of_scans))
corr_matrix = 0.0d0

!==============================================================================!
!                           WRITE PARAMETERS TO FILE                           !
!==============================================================================!
! Open file to write time to
OPEN(UNIT=1, FILE=filename_parameters, STATUS='REPLACE', ACTION='WRITE')

! Write parameters
WRITE(1,"(A15,F25.15)") "Gamma =", Gamma
WRITE(1,"(A15,F25.15)") "Omega =", Omega
WRITE(1,"(A15,F25.15)") "alpha =", alpha
! WRITE(1,"(A15,F25.15)") "delta =", delta
WRITE(1,"(A15,F25.15)") "xi =", xi

WRITE(1,"(A15,F25.15)") "dt =", dt
! WRITE(1,"(A15,F25.15)") "Max t =", t_max
WRITE(1,"(A15,F25.15)") "Max tau1 =", tau1_max
! WRITE(1,"(A15,F25.15)") "Max tau2 =", tau2_max

WRITE(1, *) " "
WRITE(1, *) "Halfwidth Scan Values"
DO index = 0, number_of_scans
  WRITE(1, "(A15, F25.15)") "delta =", delta_array(index)
END DO

! Close file
CLOSE(1)

!===============================================================================!
!                  CALCULATE SECOND-ORDER CORRELATION FUNCTION                  !
!===============================================================================!
! Reset run_counter
run_counter = 0

! Set OMP clauses
!$OMP PARALLEL DO PRIVATE(index, delta_scan, g1_array) SHARED(run_counter)

! Cycle through halfwidth values
DO index = 0, number_of_scans
  ! Grab delta value
  delta_scan = delta_array(index)

  ! Calculate g1
  CALL G1_CalculateRK4(Gamma, Omega, alpha, delta_scan, xi, &
                    & dt, tau_steps, &
                    & g1_array, .FALSE., "NONE")

  ! Save data to matrix
  corr_matrix(:, index) = g1_array

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
DO index = 0, tau_steps
  ! Write to file by g1 point.
  ! Each row is a data point, each column is a scan
  WRITE(2, *) REAL(corr_matrix(index, :))
  WRITE(3, *) AIMAG(corr_matrix(index, :))

  ! WRITE(2) REAL(corr_matrix(index, :))
  ! WRITE(3) AIMAG(corr_matrix(index, :))
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


END PROGRAM THREE_LEVEL_ATOM_G1