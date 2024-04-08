'''
Test Clawpack
=============
A simple IPython prompt to test the installation of Clawpack.

Link: https://www.clawpack.org/pyclaw/
'''

from clawpack.pyclaw import examples
claw = examples.shock_bubble_interaction.setup()
claw.run()
claw.plot()
