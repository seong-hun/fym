import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

import fym
import fym.plotting as plotting
import fym.logging

rc('font', **{
    "size": 22,
})

data = fym.logging.load('data/plot_figures/result.h5')  # a simulation result obtained from fym

# - function 'plot' (compatible with Matplotlib.pyplot)
data_dict = data
data_dict["state3d"] = data["state"][:, :3]  # for 3d example
data_dict["control_shift"] = data["control"] + np.rad2deg(1)  # broadcasting; for 2d example
# make draw dictionaries
draw_dict = {
    "state_012_equal": {
        "plot": ["state3d"],
        "projection": "3d",
        "type": ["plot"],
        "xlabel": "x0 (m)",
        # "ylabel": "x1 (m)",
        "zlabel": "x2 (m)",
        "xlim": [0., 2.],
        "c": ["b"],
        "label": ["3d_example"],
        "axis": "equal",
    },
    "state_012": {
        "plot": ["state3d"],
        "projection": "3d",
        "type": ["plot"],
        "xlabel": "x0 (m)",
        # "ylabel": "x1 (m)",
        "zlabel": "x2 (m)",
        "xlim": [0., 2.],
        "c": ["b"],
        "label": ["3d_example"],
    },
    "control": {
        "plot": [["time", "control_shift"], ["time", "control"]],
        "type": ["plot", "scatter"],
        "projection": "2d",
        "xlabel": "t (s)",
        "ylabel": ["u0 (deg)", "u1 (deg)"],
        "ylim": [[-3e3, 4e3], [1e3, 9e3]],
        "c": ["r", "b"],
        "label": ["u_shift", "u"],
        "alpha": [0.5, 0.1],
        # "axis": "equal",
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
