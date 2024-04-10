r'''
Euler 2D Quadrants
==================
A simple example solving the Euler equations of compressible fluid dynamics.

Link: https://www.clawpack.org/gallery/pyclaw/gallery/quadrants

    \rho_t + (\rho u)_x + (\rho v)_y & = 0 \\
    (\rho u)_t + (\rho u^2 + p)_x + (\rho uv)_y & = 0 \\
    (\rho v)_t + (\rho uv)_x + (\rho v^2 + p)_y & = 0 \\
    E_t + (u (E + p) )_x + (v (E + p))_y & = 0

Here `\rho` is the density, (u, v) is the velocity, and E is the total energy.
The initial condition is one of the 2D Riemann problems from the paper of
Liska and Wendroff.
'''

from clawpack import riemann
from clawpack.riemann.euler_4wave_2D_constants import density, x_momentum, y_momentum, energy, num_eqn

from clawpack.visclaw import colormaps
from clawpack.visclaw.data import ClawPlotData
from clawpack.visclaw.plotclaw import plotclaw

from clawpack.pyclaw.util import run_app_from_main

def solution(density_, velocity_, pressure_, gamma = 1.4, use_petsc = False):
    # PETSc: Portable, Extensible Toolkit for Scientific Computation
    if use_petsc:
        import clawpack.petclaw as pyclaw
    else:
        from clawpack import pyclaw

    # computational domain (unit square): [0, 0] to [1, 1] divided into a 100x100 grid
    domain = pyclaw.Domain([0., 0.], [1., 1.], [100, 100])
    solution = pyclaw.Solution(num_eqn, domain)
    solution.problem_data['gamma'] = gamma

    # cell-centered coordinates for the grids
    xx, yy = domain.grid.p_centers

    # boolean mask for left (l), right (r), bottom (b), and top (t) sections of the domain
    l, r, b, t = xx < 0.8, xx >= 0.8, yy < 0.8, yy >= 0.8

    # set initial data
    solution.q[density,...] = density_[0] * r * t + density_[1] * l * t + density_[2] * l * b + density_[3] * r * b
    u = velocity_[0][0] * r * t + velocity_[1][0] * l * t + velocity_[2][0] * l * b + velocity_[3][0] * r * b
    v = velocity_[0][1] * r * t + velocity_[1][1] * l * t + velocity_[2][1] * l * b + velocity_[3][1] * r * b
    p = pressure_[0] * r * t + pressure_[1] * l * t + pressure_[2] * l * b + pressure_[3] * r * b

    solution.q[x_momentum,...] = solution.q[density,...] * u
    solution.q[y_momentum,...] = solution.q[density,...] * v
    solution.q[energy,...] = 0.5 * solution.q[density,...] * (u ** 2 + v ** 2) + p / (gamma - 1.0)

    return solution

def setup(use_petsc = False, riemann_solver = 'roe'):
    '''
    Sets up the simulation environment, configuring the solver, domain, initial conditions,
    and other simulation parameters based on the Euler equations.
    
    Supports choosing between different solvers and optionally using PETSc for parallel execution.    
    '''

    # PETSc: Portable, Extensible Toolkit for Scientific Computation
    if use_petsc:
        import clawpack.petclaw as pyclaw
    else:
        from clawpack import pyclaw
    
    if riemann_solver.lower() == 'roe':
        solver = pyclaw.ClawSolver2D(riemann.euler_4wave_2D)
        solver.transverse_waves = 2
    elif riemann_solver.lower() == 'hlle':
        solver = pyclaw.ClawSolver2D(riemann.euler_hlle_2D)
        solver.transverse_waves = 0
        solver.cfl_desired = 0.4 # (stability condition)
        solver.cfl_max = 0.5 # (stability condition)
    
    # extrapolate the values at the boundaries from the interior values
    solver.all_bcs = pyclaw.BC.extrap

    density_ = [1.5, 0.532258064516129, 0.137992831541219, 0.532258064516129]
    velocity_ = [(0.0, 0.0), (1.206045378311055, 0.0), (1.206045378311055, 1.206045378311055), (0.0, 1.206045378311055)]
    pressure_ = [1.5, 0.3, 0.029032258064516, 0.3]
    
    claw = pyclaw.Controller()
    claw.tfinal = 0.8 # (time)
    claw.num_output_times = 10
    claw.solution = solution(density_, velocity_, pressure_)
    claw.solver = solver

    claw.output_format = 'ascii'
    claw.outdir = './_output'

    return claw

def setplot(plotdata = None):
    '''
    Configures the plotting settings for the simulation output, representing density.
    '''
    if plotdata is None:
        plotdata = ClawPlotData()
    
    # clear figures, axes, items data
    plotdata.clearfigures()

    # figure for density - pcolor
    plotfigure = plotdata.new_plotfigure(name = 'Density', figno = 0)

    # set up axes
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.xlimits = 'auto'
    plotaxes.ylimits = 'auto'
    plotaxes.scaled = True
    plotaxes.title = 'Density'

    # set up items
    plotitem = plotaxes.new_plotitem(plot_type = '2d_pcolor')
    plotitem.plot_var = density
    plotitem.pcolor_cmap = colormaps.yellow_red_blue
    plotitem.pcolor_cmin = 0.0
    plotitem.pcolor_cmax = 2.0
    plotitem.add_colorbar = True

    # figure for density - schlieren
    plotfigure = plotdata.new_plotfigure(name = 'Schlieren', figno = 1)

    # set up axes
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.xlimits = 'auto'
    plotaxes.ylimits = 'auto'
    plotaxes.scaled = True # aspect ratio = 1
    plotaxes.title = 'Density'

    # set up items
    plotitem = plotaxes.new_plotitem(plot_type = '2d_schlieren')
    plotitem.plot_var = density
    plotitem.schlieren_cmin = 0.0
    plotitem.schlieren_cmax = 1.0
    plotitem.add_colorbar = False

    return plotdata

def setplot_simple(plotdata = None):
    '''
    Configures the plotting settings for the simulation output, representing density, without axes, labels, title, and colorbar.
    '''
    if plotdata is None:
        plotdata = ClawPlotData()
    
    # clear figures, axes, items data
    plotdata.clearfigures()

    # figure for density - pcolor
    plotfigure = plotdata.new_plotfigure(name = 'Density', figno = 0)

    # set up axes
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.xlimits = 'auto'
    plotaxes.ylimits = 'auto'
    plotaxes.scaled = True
    plotaxes.title = ''
    plotaxes.axescmd = 'subplot().set_axis_off()' # remove axes

    # set up items
    plotitem = plotaxes.new_plotitem(plot_type = '2d_pcolor')
    plotitem.plot_var = density
    plotitem.pcolor_cmap = colormaps.yellow_red_blue
    plotitem.pcolor_cmin = 0.0
    plotitem.pcolor_cmax = 2.0
    plotitem.add_colorbar = False

    # figure for density - schlieren
    plotfigure = plotdata.new_plotfigure(name = 'Schlieren', figno = 1)

    # set up axes
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.xlimits = 'auto'
    plotaxes.ylimits = 'auto'
    plotaxes.scaled = True # aspect ratio = 1
    plotaxes.title = ''
    plotaxes.axescmd = 'subplot().set_axis_off()' # remove axes

    # set up items
    plotitem = plotaxes.new_plotitem(plot_type = '2d_schlieren')
    plotitem.plot_var = density
    plotitem.schlieren_cmin = 0.0
    plotitem.schlieren_cmax = 1.0
    plotitem.add_colorbar = False

    return plotdata

if __name__ == '__main__':
    output = run_app_from_main(setup)
    plotclaw('_output', setplot = setplot_simple)