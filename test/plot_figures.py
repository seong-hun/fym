import os
import numpy as np
import matplotlib.pyplot as plt

import fym
import fym.plotting as plotting
import fym.logging


data = fym.logging.load('data/plot_figures/result.h5')  # a simulation result obtained from fym

# - class 'Plotter' (will be deprecated)
# data consists of three keys: state, action, time
state = data['state']
# e.g., system name is "main".
# Note: state consists of keys corresponding to each systems.
ctrl = data['control']
time = data['time']
plotter = plotting.Plotter()
plotter.plot2d(time, state)  # tmp

# - function 'plot' (compatible with Matplotlib.pyplot)
data_dict = data
data_dict["state3d"] = data["state"][:, :3]  # for 3d example
data_dict["control_shift"] = data["control"] + np.rad2deg(1)  # broadcasting; for 2d example
# make draw dictionaries
draw_dict = {
    "state_012": {
        "plot": ["state3d"],
        "type": "3d",
        "xlabel": "x0 (m)",
        "ylabel": "x1 (m)",
        "zlabel": "x2 (m)",
        "c": ["b"],
        "label": ["3d_example"]
    },
    "control": {
        "plot": [["time", "control_shift"], ["time", "control"]],
        "type": "2d",
        "xlabel": "t (s)",
        "ylabel": ["u0 (deg)", "u1 (deg)"],
        "c": ["r", "b"],
        "label": ["u_shift", "u"],
        "alpha": [0.5, 0.1],
    },
}
weight_dict = {
    "control": np.rad2deg(1)*np.ones(2),
    "control_shift": np.rad2deg(1)*np.ones(2),
    # None: weight = 1. (In this case, default weights are given for e.g., "time" and "state3d")
}
save_dir = "./data/plot_figures"
os.makedirs(save_dir, exist_ok=True)
figs = plotting.plot(data_dict, draw_dict, weight_dict=weight_dict, save_dir=save_dir)
