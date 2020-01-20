import fym
import fym.plotting as plotting
import fym.logging


data = fym.logging.load('data/plot_figures/result.h5')  # a simulation result obtained from fym

# data consists of three keys: state, action, time
state = data['state']
# e.g., system name is "main".
# Note: state consists of keys corresponding to each systems.
ctrl = data['control']
time = data['time']

plotter = plotting.Plotter()
plotter.plot2d(time, state)  # tmp
plotter.plot2d(time, state, name='state')
plotter.plot2d(time, ctrl, name='ctrl', xlabel='t (s)', ylabels=['$L (g)$', '$\phi (deg)$'])

plotter.show()
