# Makefile for compiling Omega_scan_spectrum.f90, which scans the driving
# amplitude (Rabi frequency) Omega and computes the first-order correlation
# function g1(tau) for a driven three-level atom, using the routines in
# moment_equations_3LA_functions.f90 for the underlying RK4 integration.
#
# Targets:
#   make          build the Omega_scan_spectrum executable
#   make clean    remove all compiled object, module, and executable files

#------------------------------------------------------------------------------#
#                           FORTRAN COMPILER OPTIONS                           #
#------------------------------------------------------------------------------#
#------------------#
#     Compiler     #
#------------------#
FC = gfortran
# Intel oneAPI compiler -- swap to this instead of gfortran if available
# FC = ifx

#---------------#
#     Flags     #
#---------------#
FFLAGS = -O3 -Wall -Wextra

#-----------------------------#
#     Linear Algebra Libs     #
#-----------------------------#
# Lapack library. Adjust depending on where your precompiled LAPACK
# libraries live.
LAPACK_LIBRARY = /usr/local/lib/

# gfortran: link against the system BLAS/LAPACK installation directly.
LIBS = -I$(LAPACK_LIBRARY) -L$(LAPACK_LIBRARY) -lblas -llapack

# Intel oneAPI + MKL: -qmkl tells ifx to compile and link against Intel's
# MKL implementation of BLAS/LAPACK directly, so none of the explicit
# -I/-L/-lblas/-llapack flags above are needed. Comment out the LIBS line
# above and uncomment this one (together with FC = ifx) to use it instead.
# LIBS = -qmkl

#-------------------#
#     Filenames     #
#-------------------#
PROG = Omega_scan_spectrum
MOD  = moment_equations_3LA_functions

#----------------------#
#     Source Files     #
#----------------------#
MOD_SRC  = $(MOD).f90
MAIN_SRC = $(PROG).f90

#----------------------#
#     Object Files     #
#----------------------#
MOD_OBJ  = $(MOD).o
MAIN_OBJ = $(PROG).o

#------------------------------------------------------------------------------#
#                                PHONY TARGETS                                 #
#------------------------------------------------------------------------------#
.PHONY: all clean

# Default target: typing "make" alone builds this, since it is the first
# target in the file.
all: $(PROG)

$(PROG): $(MAIN_OBJ) $(MOD_OBJ)
	$(FC) $(FFLAGS) -o $@ $^ $(LIBS)

#------------------------------------------------------------------------------#
#                             OBJECT FILE RULES                                #
#------------------------------------------------------------------------------#
# MAIN_OBJ depends on MOD_OBJ as well as its own source file. This forces the
# module to be compiled first (so its .mod interface file already exists by
# the time the main program's USE statement needs it) without using the
# module's object file directly in the compile command itself.
$(MAIN_OBJ): $(MAIN_SRC) $(MOD_OBJ)
	$(FC) $(FFLAGS) -c $<

$(MOD_OBJ): $(MOD_SRC)
	$(FC) $(FFLAGS) $(LIBS) -c $<

#------------------------------------------------------------------------------#
#                                  CLEAN FILES                                 #
#------------------------------------------------------------------------------#
clean:
	rm -f *.o *.mod $(PROG)