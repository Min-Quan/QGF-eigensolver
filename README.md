# GFE-engensolver
This repository contains the demonstration code of the article "Quantum Gaussian filter for exploring ground-state properties" ([arXiv:2112.06026](https://arxiv.org/abs/2112.06026)) The simulation is based on package [QuTip](https://github.com/qutip).

## Introduction
Our quantum algorithms are proposed to solve the ground state problem of a given local Hamiltonian. A Gaussian filter operator is a Gaussian function of the Hamiltonian to be solved. The key point is to construct Gaussian filters that applies to an initial state to single out the ground state information while filtering the other eigenstates. 

We propose two approaches to construct Gaussian filters in this paper. The first one is approximating the Gaussian filters by a linear combination of Hamiltonian evolution over varies time, where the coefficients are determined by the parameters of Gaussian filter. In this method, the overlaps between initila state and evolved state are firstly measured; and the estimated energy is a classical weighted summation of the overlaps. So, the estimated energy can be optimized by a classical post-processing. Another approach is to construct the Gaussian filter by an integration of unitary operators, which is completed by entangling qubit and an ancillary qumode since the infinity integration naturally exists in the qumode state.

## Dependecies
- Python 3.7
- Numpy
- Qutip 4.6
- Scipy
- matplotlib

## File description
- [H_generator.py](https://github.com/Min-Quan/QGF-eigensolver/blob/main/H_generator.py): a package to generate Hamiltonians of tranversed field quantum Ising model and its local operator list.
- [GFE.py](https://github.com/Min-Quan/QGF-eigensolver/blob/main/GFE.py): a package to run the proposed algorithms. It contains the functions: 1. solve the approximate Gaussian function for a given input list; 2. generate the coefficients of time evolution; 3. compute the overlaps with ideal time evolutions; 4. compute additional overlaps with ideal time evolutions after changing discrete parameters; 5. compute the overlaps while considering the Trotter decomposition and noises; 6. estimate the ground state energy by classical summation of overlaps; 7.solve the rgound state with the ancillary qumode.
- [Demo_of_approximate_GF.ipynb](https://github.com/Min-Quan/QGF-eigensolver/blob/main/Demo_of_approximate_GF.ipynb): the demonstration of approximate Gaussian functions to show the relation between the relaiton of max phase, variance of the Gaussian function, and the input range can be well approximated.
- [Demo_of_solving_Ising_model.ipynb](https://github.com/Min-Quan/QGF-eigensolver/blob/main/Demo_of_solving_Ising_model.ipynb): the demonstration of solving the ground state energy of an Ising model by the Gaussian filter algorithm with the ideal time evolution. It shows the estimation error as a function of mean value (variance) of Gaussian filters while fixing variance (mean value) of Gaussian filters.
- [Demo_of_solving_Ising_model_iterative_strategy.ipynb](https://github.com/Min-Quan/QGF-eigensolver/blob/main/Demo_of_solving_Ising_model_iterative_strategy.ipynb): the demonstration of solving the ground state energy of an Ising model by the Gaussian filter algorithm with the ideal time evolution. It shows the process of an iterative strategy. While increasing the max phase, the eigensolve computes the additional overlaps and estimates the new result.
[Demo_of_solving_Ising_model_under_noise.ipynb](https://github.com/Min-Quan/QGF-eigensolver/blob/main/Demo_of_solving_Ising_model_under_noise.ipynb): the demonstration of solving the ground state energy of an Ising model by the Gaussian filter algorithm with considering the first order Trotter decomposition and a noise channel.
- [Demo_of_solving_Ising_model_with_qumode.ipynb](https://github.com/Min-Quan/QGF-eigensolver/blob/main/Demo_of_solving_Ising_model_with_qumode.ipynb): the demonstration of solving the ground state energy of an Ising model by the qumode assisted Gaussian filter algorithm.
