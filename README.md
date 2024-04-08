# Sea Ice Dynamics
## Dartmouth College (Applied Mathematics Lab)

### Installation
1. `git clone git@github.com:CarterKruse/Sea-Ice-Dynamics.git`
2. `cd Sea-Ice-Dynamics`
3. `pip install clawpack`

### Clawpack
**Overview**

[Clawpack](https://www.clawpack.org/) (Conservation Laws Package) is a collection of finite volume methods for linear and nonlinear hyperbolic systems of conservation laws. Clawpack employs high-resolution Godunov-type methods with limiters in a general framework applicable to many kinds of waves. Clawpack is written in Fortran and Python.

**Key Features**
- *Solution of general hyperbolic PDEs* provided a Riemann solver is given.
- *Adaptive mesh refinement* included in [AMRClaw](https://www.clawpack.org/amrclaw) and [GeoClaw](https://www.clawpack.org/geoclaw).
- *Parallelism* scalable to tens of thousands of cores or more, included in [PyClaw](https://www.clawpack.org/pyclaw/).

Solution of a given PDE system with Clawpack requires specifying an approximate Riemann solver for that system. Solvers for many common applications are bundled with the software. See [Riemann solvers](https://www.clawpack.org/riemann) for more information.
