import os
import numpy as np
import math
import matplotlib.pyplot as plt
from collections import OrderedDict

import fym


class Plotter:
    figures = OrderedDict()  # dictionary for figures
    tmp_name = 0

    def __init__(self, plot_type="plot"):
        self.plot_type = plot_type

    def set_plot_type(self, plot_type):
        self.plot_type = plot_type

    def plot(self, ax, *args):
        if self.plot_type == "plot":
            result = ax.plot(*args)
        elif self.plot_type == "step":
            result = ax.step(*args)
        else:
            raise ValueError("{} is not a supported plot type.".format(self.plot_type))
        return result

    def plot2d(self, x, y, name=None, xlabel='time (s)', ylabels=['x'], ncols=1):
        if not x.shape[0] == y.shape[0]:
            raise ValueError("The length of x must agree with those of y's.")
        if len(y.shape) == 1:
            y = np.expand_dims(y, axis=1)

        nrows = math.ceil(y.shape[1]/ncols)
        fig, ax = plt.subplots(nrows, ncols)
        if nrows == 1:
            ax = np.expand_dims(ax, axis=0)
        if ncols == 1:
            ax = np.expand_dims(ax, axis=1)

        # plot 2d figure
        nplt = 0
        for j in range(ncols):
            for i in range(nrows):
                if nplt == y.shape[1]:
                    break
                else:
                    nplt += 1
                    self.plot(ax[i, j], x, y[:, i])
                    # ax[i, j].plot(x, y[:, i])
                    if len(ylabels) == 1:
                        if y.shape[1] == 1:
                            ax[i, j].set_ylabel(ylabels[0])
                        else:
                            ax[i, j].set_ylabel(ylabels[0]+'{}'.format(nplt))
                    elif len(ylabels) == y.shape[1]:
                        ax[i, j].set_ylabel(ylabels[i])
                    else:
                        raise ValueError("The number of labels must agree with the number of y's.")
            ax[-1][j].set_xlabel(xlabel)

        # add an element into figures dictionary
        if name is None:
            self.tmp_name += 1
            name = 'tmp{}'.format(self.tmp_name)
        elif isinstance(name, str):
            pass
        else:
            raise ValueError("Figure name has to be string or None (defalut value).")
        self.figures[name] = [fig, ax]

    def show(self):
        plt.show()


def plot(data_dict, draw_dict, weight_dict={}, save_dir="./",
         option={"savefig": {"dpi": 150, "transparent": False}},):
    figs = {}
    for fig_name in draw_dict:
        figs[fig_name] = plt.figure()
        fig_dict = draw_dict[fig_name]
        if fig_dict["type"] == "3d":
            _plot3d(figs, fig_name, fig_dict, data_dict, weight_dict)
        elif fig_dict["type"] == "2d":
            _plot2d(figs, fig_name, fig_dict, data_dict, weight_dict)
        os.makedirs(save_dir, exist_ok=True)
        fig_path = os.path.join(save_dir, fig_name)
        plt.savefig(fig_path, **option["savefig"])
    plt.close("all")
    return figs


def _plot3d(figs, fig_name, fig_dict, data_dict, weight_dict):
    # 3d graph
    ax = figs[fig_name].add_subplot(1, 1, 1, projection="3d")
    for i_plt, plot_name in enumerate(fig_dict["plot"]):
        x, y, z = [data_dict[plot_name][:, i] for i in range(3)]
        # ax.set_aspect("equal")  # not supported
        # weight
        weights_xyz = weight_dict.get(plot_name)
        if weights_xyz is None:
            w_x, w_y, w_z = [np.ones(1), np.ones(1), np.ones(1)]  # broadcasting
        else:
            w_x, w_y, w_z = [weights_xyz[i] for i in range(3)]
        # plot properties
        plot_property_dict = {}
        for key in ["c", "label", "alpha"]:
            plot_property_dict[key] = _get_plot_property(fig_dict, key, i_plt)
        ax.plot(w_x*x, w_y*y, w_z*z, **plot_property_dict)
        ax.set_xlabel(fig_dict["xlabel"])
        ax.set_ylabel(fig_dict["ylabel"])
        ax.set_zlabel(fig_dict["zlabel"])
        ax.set_title(fig_name)
    if fig_dict.get("label") is not None:
        ax.legend()


def _plot2d(figs, fig_name, fig_dict, data_dict, weight_dict):
    # 2d graph
    ax = []
    for i_plt, [x_name, y_name] in enumerate(fig_dict["plot"]):
        data_x, data_y = data_dict[x_name], data_dict[y_name]
        if len(data_y.shape) == 2:
            data_y_dim = data_y.shape[1]
        else:
            data_y = np.expand_dims(data_y, axis=1)
            data_y_dim = data_y.shape[1]
        for i in range(data_y_dim):
            if i_plt == 0:
                ax.append(figs[fig_name].add_subplot(data_y_dim, 1, i+1))
            # weight
            w_x = weight_dict.get(x_name)
            if w_x is None:
                w_x = np.ones(1)  # broadcasting
            w_ys = weight_dict.get(y_name)
            if w_ys is None:
                w_y = np.ones(1)  # broadcasting
            else:
                w_y = w_ys[i]
            # plot properties
            plot_property_dict = {}
            for key in ["c", "label", "alpha"]:
                plot_property_dict[key] = _get_plot_property(fig_dict, key, i_plt)
            ax[i].plot(w_x*data_x, w_y*data_y[:, i], **plot_property_dict)
            ax[i].set_xlabel(fig_dict["xlabel"])
            ax[i].set_ylabel(fig_dict["ylabel"][i])
            if "ylim" in fig_dict:
                ax[i].set_ylim(fig_dict["ylim"][i])
    ax[0].set_title(fig_name)
    if fig_dict.get("label") is not None:
        ax[0].legend()


def _get_plot_property(fig_dict, key, i_plt):
    # e.g., key = "label"
    # i_plt: plot index (zero-indexing)
    values = fig_dict.get(key)
    if values is None:
        value = None
    else:
        if len(values) < (i_plt+1):
            value = None  # default value
        else:
            value = values[i_plt]
    return value


if __name__ == '__main__':
    import fym.logging
    # - class 'Plotter' (will be deprecated)
    data = fym.logging.load('data/main/result.h5')  # result obtained from fym.logging
    # data consists of three keys: state, action, time
    state = data['state']
    # e.g., system name is "main".
    # Note: state consists of keys corresponding to each systems.
    ctrl = data['control']
    time = data['time']
    plotter = Plotter()
    plotter.plot2d(time, state)  # tmp
