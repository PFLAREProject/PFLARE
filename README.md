[![CI](https://img.shields.io/github/actions/workflow/status/PFLAREProject/PFLARE/ci_build.yml?branch=main&label=CI)](https://github.com/PFLAREProject/PFLARE/actions/workflows/ci_build.yml)
[![spack](https://img.shields.io/github/actions/workflow/status/PFLAREProject/PFLARE_spack/ci_build.yml?branch=main&label=spack)](https://github.com/PFLAREProject/PFLARE_spack/actions/workflows/ci_build.yml)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/PFLAREProject/PFLARE/HEAD?urlpath=%2Fdoc%2Ftree%2Fnotebooks%2F01_getting_started.ipynb)

<img align="right" img src="PFLARE_logo.png" width="300" height="300" />

# PFLARE library
#### Created by: Steven Dargaville

This library contains methods which can be used to solve linear systems in parallel with PETSc, with interfaces in C/Fortran/Python. 

It aims to provide fast & scalable iterative methods for asymmetric linear systems, in parallel and on both CPUs and GPUs.
   
Some examples of asymmetric linear systems that PFLARE can scalably solve include:   
   - Advection equations
   - Streaming operators from Boltzmann applications
   - Space-time discretisations
   - Heavily anisotropic Poisson/diffusion equations
   
without requiring Gauss-Seidel methods. This includes time dependent or independent equations, with structured or unstructured grids, with lower triangular structure or without.

## Methods available in PFLARE

PFLARE adds new methods to PETSc, including:
1) Polynomial approximate inverses, e.g., GMRES and Neumann polynomials
2) Reduction multigrids, e.g., AIRG, nAIR and lAIR
3) CF splittings, e.g., PMISR DDC
4) Methods to extract diagonally dominant submatrices

## Quick start

You can get started with PFLARE in one of four ways:

* PFLARE is now available directly through the PETSc configure with: `--download-pflare`, see [docs/installation.md](docs/installation.md)
* To run the Jupyter notebooks in `notebooks/` in your browser without requiring a local install, click the Binder badge above
* To download a Docker image with PFLARE installed, run `docker run -it stevendargaville/pflare && make check`
* To build from source, see [docs/installation.md](docs/installation.md) 

## Documentation

For details about PFLARE, please see:

| Path | Contents |
|---|---|
| [docs/new_methods.md](docs/new_methods.md) | Details on the new methods added by PFLARE |
| [docs/installation.md](docs/installation.md) | How to install PFLARE |
| [docs/use_pflare.md](docs/use_pflare.md) | How to use PFLARE |
| [docs/gpus.md](docs/gpus.md) | Using GPUs with PFLARE |
| [docs/reuse.md](docs/reuse.md) | Re-using components of PFLARE |
| [docs/options.md](docs/options.md) | List of the options available in PFLARE |   
| [docs/faq.md](docs/faq.md) | Frequently asked questions and help! |   

and the Jupyter notebooks:

| Path | Contents |
|---|---|
| [notebooks/01_getting_started.ipynb](notebooks/01_getting_started.ipynb) | Introduce PFLARE |
| [notebooks/02_pcpflareinv.ipynb](notebooks/02_pcpflareinv.ipynb) | Examine some of the approximate inverses found in PCPFLAREINV |
| [notebooks/03_cf_splitting.ipynb](notebooks/03_cf_splitting.ipynb) | Visualise the C/F splitting and explore the PMISR-DDC algorithm |
| [notebooks/04_pcair.ipynb](notebooks/04_pcair.ipynb) | Introduce PCAIR and the AIRG method |
| [notebooks/05_parallel.ipynb](notebooks/05_parallel.ipynb) | Discuss PCAIR, parallelism and GPUs |
| [notebooks/06_reuse.ipynb](notebooks/06_reuse.ipynb) | Discuss PCAIR and reuse |
     
## More examples

For more ways to use the library please see the Fortran/C examples and the Makefile in `tests/`, along with the Python examples in `python/`.

## References \& citing

Please see the references below for more details. If you use PFLARE in your work, please consider citing [1-3].

1. S. Dargaville, R. P. Smedley-Stevenson, P. N. Smith, C. C. Pain, AIR multigrid with GMRES polynomials (AIRG) and additive preconditioners for Boltzmann transport, _Journal of Computational Physics_ 518 (2024) 113342  
2. S. Dargaville, R. P. Smedley-Stevenson, P. N. Smith, C. C. Pain, Coarsening and parallelism with reduction multigrids for hyperbolic Boltzmann transport, _The International Journal of High Performance Computing Applications_ 39(3) (2025) 364-384  
3. S. Dargaville, R. P. Smedley-Stevenson, P. N. Smith, C. C. Pain, Solving advection equations with reduction multigrids on GPUs (2025) http://arxiv.org/abs/2508.17517  
4. T. A. Manteuffel, S. Münzenmaier, J. Ruge, B. Southworth, Nonsymmetric Reduction-Based Algebraic Multigrid, _SIAM Journal on Scientific Computing_ 41 (2019) S242–S268  
5. T. A. Manteuffel, J. Ruge, B. S. Southworth, Nonsymmetric algebraic multigrid based on local approximate ideal restriction (lAIR), _SIAM Journal on Scientific Computing_ 40 (2018) A4105–A4130   
6. A. Ali, J. J. Brannick, K. Kahl, O. A. Krzysik, J. B. Schroder, B. S. Southworth, Constrained local approximate ideal restriction for advection-diffusion problems, _SIAM Journal on Scientific Computing_ (2024) S96–S122  
7. T. Zaman, N. Nytko, A. Taghibakhshi, S. MacLachlan, L. Olson, M. West, Generalizing reduction-based algebraic multigrid, _Numerical Linear Algebra with Applications_ 31(3) (2024) e2543   
8. Loe, J.A., Morgan, R.B, Toward efficient polynomial preconditioning for GMRES. _Numerical Linear Algebra with Applications_ 29 (2022) e2427 
