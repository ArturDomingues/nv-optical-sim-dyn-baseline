# NV Optical Simulation – Dynamics Baseline

This repository contains the baseline code for simulating the dynamics of nitrogen-vacancy (NV) centers in diamond under optical excitation. The simulation models the population dynamics of the NV center's electronic states, accounting for various transition rates.

## Organization

- `core/`:  Contains the simulation code and some utility modules.
  - `models.py`: Core module implementing the dynamics simulation.
  - `plot_funcs.py`: Module defining plotting functions for visualization.
  - `utils.py`: Utility functions for data handling and processing.
- `original_ref_notebooks/`: Contains reference Jupyter notebooks from the original implementation.
  - `NV_Model_Scipy.ipynb`: Notebook solving the NV center dynamics simulation using rate equations with SciPy (implementated by Prof. Dr. Sérgio Ricardo Muniz).
  - `NV_sims_qutip_proto.ipynb`: Prototyping (used to change some parameters and different regimes during development) notebook for solving the NV center dynamics using QuTiP (Quantum Toolbox in Python). This is the latest implementation and used in some of the final results.
  - `NV_sims_qutip.ipynb`: First implementation notebook for solving the NV center dynamics using QuTiP, incorporate the refinements and optimizations from proto, but was the one used for calculating ODMR, Ramsey and B x PL plot with varying angle.
  - `plots.ipynb`: Notebook containing various plots generated from the simulation results, used for refining the visual of the plots and also have some SNR plots.
- `Tutorial - *.ipynb`: Tutorial notebooks that guides users through the simulation process, explaining how to set up and run simulations in different situations, * defines what situation is being simulated.

For now the repository don't have the data we have used and trying to run `plots.ipynb` will not work, unless you run NV_sim_qutip.ipynb fully and use the different values for Rabi frequency in it to generate the data and save it before running `plots.ipynb`, the values need are stated in the name of the files being called in the plots notebook. We are working on adding the data and more documentation to make it easier to use.

## Additional Information

- [QuTiP Documentation](http://qutip.org/docs/latest/index.html): Official documentation for the QuTiP library, which is used for quantum dynamics simulations in this project.
- [SciPy Documentation](https://docs.scipy.org/doc/scipy/): Official documentation for the SciPy library, which provides tools for scientific computing in Python.
- [My Master Dissertation](https://www.teses.usp.br/teses/disponiveis/76/76134/tde-25112025-095217): My master dissertation that includes detailed explanations of the models, methods, and results related to NV center dynamics simulations.
- [Zenodo Repository](https://zenodo.org/): Repository containing datasets and additional resources related to the NV center dynamics simulations(For now, this is a placeholder link).
- [QuaCCAToo](https://github.com/QISS-HZB/QuaCCAToo): Python library for simulating and analyzing spin dynamics of color centers for quantum technology applications. The place where this project is planned to be integrated in the future.
