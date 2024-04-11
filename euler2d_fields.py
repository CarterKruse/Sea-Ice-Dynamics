r'''
Euler 2D Fields
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

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def solution(density_field, velocity_field, pressure_field, gamma = 1.4, use_petsc = False):
    # PETSc: Portable, Extensible Toolkit for Scientific Computation
    if use_petsc:
        import clawpack.petclaw as pyclaw
    else:
        from clawpack import pyclaw

    # computational domain (unit square): [0, 0] to [1, 1] divided into a 100x100 grid
    domain = pyclaw.Domain([0., 0.], [1., 1.], density_field.shape)
    solution = pyclaw.Solution(num_eqn, domain)
    solution.problem_data['gamma'] = gamma

    # set initial data
    solution.q[density,...] = density_field
    solution.q[x_momentum,...] = solution.q[density,...] * velocity_field[:, :, 0]
    solution.q[y_momentum,...] = solution.q[density,...] * velocity_field[:, :, 1]
    solution.q[energy,...] = 0.5 * solution.q[density,...] * (velocity_field[:, :, 0] ** 2 + velocity_field[:, :, 1] ** 2) + pressure_field / (gamma - 1.0)

    return solution

def normalize(array, min_val = 0.0, max_val = 1.0, epsilon = 1e-10):
    # calculate the min / max of the array
    min_ = np.min(array)
    max_ = np.max(array)

    normalized = (max_val - min_val) * (array - min_) / (max_ - min_) + min_val + epsilon

    return normalized

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

    # density_field = np.ones((100, 100))
    # pressure_field = np.ones((100, 100))
    # velocity_field = np.zeros((100, 100, 2))

    np.random.seed(42)
    sigma = 4
    density_field = normalize(gaussian_filter(np.random.rand(100, 100) * 2, sigma))
    pressure_field = normalize(gaussian_filter(np.random.rand(100, 100) * 2, sigma))

    x_velocity_field = normalize(gaussian_filter(np.random.rand(100, 100) * 4 - 2, sigma), -1, 1)
    y_velocity_field = normalize(gaussian_filter(np.random.rand(100, 100) * 4 - 2, sigma), -1, 1)
    velocity_field = np.stack((x_velocity_field, y_velocity_field), axis = -1)

    plot_fields(density_field, velocity_field, pressure_field)

    claw = pyclaw.Controller()
    claw.tfinal = 0.8 # (time)
    claw.num_output_times = 10
    claw.solution = solution(density_field, velocity_field, pressure_field)
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
    
    plotdata.plotdir = './_plots'
    
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
    plotitem.pcolor_cmin = 'auto'
    plotitem.pcolor_cmax = 'auto'
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
    plotitem.schlieren_cmin = 'auto'
    plotitem.schlieren_cmax = 'auto'
    plotitem.add_colorbar = False

    return plotdata

def setplot_simple(plotdata = None):
    '''
    Configures the plotting settings for the simulation output, representing density, without axes, labels, title, and colorbar.
    '''
    if plotdata is None:
        plotdata = ClawPlotData()
    
    plotdata.plotdir = './_plots'
    
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
    plotitem.pcolor_cmin = 'auto'
    plotitem.pcolor_cmax = 'auto'
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
    plotitem.schlieren_cmin = 'auto'
    plotitem.schlieren_cmax = 'auto'
    plotitem.add_colorbar = False

    return plotdata

# # # # #

def plot_fields(density_field, velocity_field, pressure_field):
    '''
    Plots the initial density, velocity, and pressure fields using matplotlib.
    '''
    fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize = (14, 12))
    fig.subplots_adjust(hspace = 0.3, wspace = 0.3)

    # density
    ax = axes[0, 0]
    density_img = ax.imshow(density_field, cmap = 'viridis')
    ax.set_title('Density Field')
    fig.colorbar(density_img, ax = ax)

    # velocity magnitude
    ax = axes[0, 1]
    magnitude = np.sqrt(velocity_field[:, :, 0] ** 2 + velocity_field[:, :, 1] ** 2)
    velocity_img = ax.imshow(magnitude, cmap = 'plasma')
    ax.set_title('Velocity Magnitude')
    fig.colorbar(velocity_img, ax = ax)

    # pressure
    ax = axes[1, 0]
    pressure_img = ax.imshow(pressure_field, cmap = 'inferno')
    ax.set_title('Pressure Field')
    fig.colorbar(pressure_img, ax = ax)

    # velocity vectors
    ax = axes[1, 1]
    step = 5 # only plot every 5th vector for clarity
    ax.quiver(np.arange(0, 100, step), np.arange(0, 100, step),
              velocity_field[::step, ::step, 0], velocity_field[::step, ::step, 1],
              magnitude[::step, ::step], cmap = 'plasma', scale = 50)
    ax.set_title('Velocity Vectors')
    ax.set_xlim([0, 100])
    ax.set_ylim([100, 0])

    plt.plot()
    plt.savefig('fields.png')

# # # # #

def x_velocity(solution):
    '''
    Computes the x velocity vector from the current solution data.
    '''
    return solution.q[x_momentum,...] / solution.q[density,...]

def y_velocity(solution):
    '''
    Computes the y velocity vector from the current solution data.
    '''
    return solution.q[y_momentum,...] / solution.q[density,...]

def pressure(solution):
    '''
    Computes the pressure vector from the current solution data.
    '''
    gamma = 1.4
    u = solution.q[x_momentum,...] / solution.q[density,...]
    v = solution.q[y_momentum,...] / solution.q[density,...]
    
    return (solution.q[energy,...] - 0.5 * solution.q[density,...] * (u ** 2 + v ** 2)) * (gamma - 1.0)

def setplot_advanced(plotdata = None):
    '''
    Configures the plotting settings for the simulation output, representing density, velocity, and pressure.
    '''
    if plotdata is None:
        plotdata = ClawPlotData()

    plotdata.plotdir = './_plots'

    # clear figures, axes, items data
    plotdata.clearfigures()

    # figure for density - pcolor
    plotfigure = plotdata.new_plotfigure(name = 'Density')

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
    plotitem.pcolor_cmax = 4.0
    plotitem.add_colorbar = True

    # figure for x velocity - pcolor
    plotfigure = plotdata.new_plotfigure(name = 'X Velocity')

    # set up axes
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.xlimits = 'auto'
    plotaxes.ylimits = 'auto'
    plotaxes.scaled = True
    plotaxes.title = 'X Velocity'

    # set up items
    plotitem = plotaxes.new_plotitem(plot_type = '2d_pcolor')
    plotitem.plot_var = x_velocity
    plotitem.pcolor_cmap = colormaps.yellow_red_blue
    plotitem.pcolor_cmin = -1.0
    plotitem.pcolor_cmax = 1.0
    plotitem.add_colorbar = True

    # figure for y velocity - pcolor
    plotfigure = plotdata.new_plotfigure(name = 'Y Velocity')

    # set up axes
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.xlimits = 'auto'
    plotaxes.ylimits = 'auto'
    plotaxes.scaled = True
    plotaxes.title = 'Y Velocity'

    # set up items
    plotitem = plotaxes.new_plotitem(plot_type = '2d_pcolor')
    plotitem.plot_var = y_velocity
    plotitem.pcolor_cmap = colormaps.yellow_red_blue
    plotitem.pcolor_cmin = -1.0
    plotitem.pcolor_cmax = 1.0
    plotitem.add_colorbar = True

    # figure for pressure - pcolor
    plotfigure = plotdata.new_plotfigure(name = 'Pressure')

    # set up axes
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.xlimits = 'auto'
    plotaxes.ylimits = 'auto'
    plotaxes.scaled = True
    plotaxes.title = 'Pressure'

    # set up items
    plotitem = plotaxes.new_plotitem(plot_type = '2d_pcolor')
    plotitem.plot_var = pressure
    plotitem.pcolor_cmap = colormaps.yellow_red_blue
    plotitem.pcolor_cmin = 0.0
    plotitem.pcolor_cmax = 2.0
    plotitem.add_colorbar = True

    return plotdata

# # # # #

if __name__ == '__main__':
    output = run_app_from_main(setup)
    plotclaw('_output', setplot = setplot_advanced)
