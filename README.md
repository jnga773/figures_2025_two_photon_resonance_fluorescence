# figures_2025_two_photon_resonance_fluorescence

Python and Fortran scripts for generating the data and figures in the paper "Two-Photon Resonance Fluorescence of a Three-Level Ladder-Type Atom" by Jacob Ngaha, Scott Parkins, and Howard Carmichael.

## Usage
Each of the `plot_figX.py` files can simply be run as standard Python scripts. They make used of the Matplotlib, NumPy, and QuTiP libraries.

For the `figX_data` folders, the programs are written in Fortran90 and and can be compiled using Intel's oneAPI `ifx` compiler, with the command
```
ifx -O3 -qmkl -o [executable_name] MODULE_atom.f90 [filename.f90]
```
where the `-mkl`(`/Qmkl`) flag links to Intel's MKL libraries, which are included with the compiler, and are needed to access the `LAPACK` routines. Similarly, you can compile them with `gfortran` using the command
```
gfortran -O3 -o [executable_name] MODULE_atom.f90 [filename.f90] -I/path/to/lapack -L/path/to/lapack -lblas -llapack
```
You will need to compile and install the necessary lapack routine files ([BLAS](https://www.netlib.org/blas/), [LAPACK](https://www.netlib.org/lapack), and [LAPACK95](https://www.netlib.org/lapack95)).

The programs take the necessary parameters from the `ParamList.nml` file, which is included in each directory, hence the code only needs to be compiled once.

## Files
- `fig4_data`
  - `data_files`: Folder containing the output data from `Omega_scan_spectrum.f90`.
  - `MODULE_atom.f90`: Contains the Fortran subroutines used to calculate the evolution of the master equation of the three-level atom.
  - `Omega_scan_spectrum.f90`: Calculates the data for Fig. 4 - the first-order correlation function for the parameters set in `ParamList.nml` and for different values of driving amplitude $\Omega$. 
  - `ParamList.nml`: Fortran NameList file for the system and calculation parameters.

- `fig6_data`
  - `data_files`: Folder containing the output data from `delta_scan_spectrum.f90`.
  - `MODULE_atom.f90`: Contains the Fortran subroutines used to calculate the evolution of the master equation of the three-level atom.
  - `delta_scan_spectrum.f90`: Calculates the data for Fig. 6 - the first-order correlation function for the parameters set in `ParamList.nml` and for different values of drive detuning $\delta$. 
  - `ParamList.nml`: Fortran NameList file for the system and calculation parameters.

- `python_files`
  - `dressed_state_functions.py`: Contains the Python functions used to calculate the dressed-state approximation correlation functions.
  - `figure_style.mplstyle`: Matplotlib rcParams file for the figure settings.
  - `three_level_moments.py`: Contains the Python functions used to calculate the moment equations of the three-level atom.

- `plot_fig2.py`: Calculates and plots Fig. 2 - Steady state atomic populations.

- `plot_fig3.py`: Calculates and plots Fig. 3(b) and (c) - Incoherent fluorescence spectrum in the weak- and strong-driving regime.

- `plot_fig4.py`: Calculates and plots Fig. 4 - Incoherent flourescence spectrum as a function of driving amplitude $\Omega$.

- `plot_fig5.py`: Calculates and plots Fig. 5(a-c) - Incoherent fluorescence spectrum in the strong-driving regime for three values of the dipole moment ratio: $\xi = 1 / \sqrt{2}, 1$, and $\sqrt{2}$.

- `plot_fig6.py`: Calculates and plots Fig. 6(a-c) - Incoherent flourescence spectrum as a function of drive detuning $\delta$.

- `plot_fig7.py`: Calculates and plots Fig. 7 - Second-order correlation function for weak driving amplitudes.

- `plot_fig8.py`: Calculates and plots Fig. 8(a-c) - Second-order correlation functionsn in the strong-driving regime for three values of the dipole moment ratio: $\xi = 1 / \sqrt{2}, 1$, and $\sqrt{2}$.

- `plot_fig9.py`: Calculates and plots Fig. 9 - Dressed-state approximation second-order auto-correlations.

- `plot_fig10.py`: Calculates and plots Fig. 10 - Dressed-state approximation second-order cross-correlations.