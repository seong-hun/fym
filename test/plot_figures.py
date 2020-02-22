import numpy as np

import fym.logging as logging
import fym.plotting as plotting


# compatibility check for fym.logging
data = logging.load('data/plot_figures/result.h5')  # tmp

state = data['state']
control = data['control']
time = data['time']

# auxiliary
x = np.linspace(0., 100., num=100)
y = x.reshape(-1, 1)
z = np.random.rand(100, 2)

# Plotter
plotter = plotting.Plotter()
# plot (2d)
plotter.plot(x, y)  # tmp
plotter.plot(
    time, state,
    name="state", ncols=3
)
plotter.plot(
    time, control,
    name="control",
    xlabel='t (s)', ylabels=['$L (g)$', '$\phi (deg)$'],
)

# step
plotter.plot(
    time, control,
    name="control (step)",
    xlabel='t (s)', ylabels=['$L (g)$', '$\phi (deg)$'],
    plot_type="step",
)

# plot (3d)
plotter.plot(x, y, z)

plotter.show()
