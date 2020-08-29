import os
import numpy as np
import math
import matplotlib.pyplot as plt
from collections import OrderedDict

import fym


def plot(data_dict, draw_dict, weight_dict={}, save_dir="./",
         option={"savefig": {"dpi": 150, "transparent": False}},):
    figs = {}
    for fig_name in draw_dict:
        figs[fig_name] = plt.figure()
        fig_dict = draw_dict[fig_name]
        if fig_dict["projection"] == "3d":
            _plot3d(figs, fig_name, fig_dict, data_dict, weight_dict)
        elif fig_dict["projection"] == "2d":
            _plot2d(figs, fig_name, fig_dict, data_dict, weight_dict)
        plt.tight_layout()  # tight layout
        os.makedirs(save_dir, exist_ok=True)
        fig_path = os.path.join(save_dir, fig_name)
        plt.savefig(fig_path, **option["savefig"])
    plt.close("all")
    return figs


def _plot3d(figs, fig_name, fig_dict, data_dict, weight_dict):
    # 3d graph
    ax = figs[fig_name].add_subplot(1, 1, 1, projection="3d")
    for i_plt, plot_name in enumerate(fig_dict["plot"]):
        data_x, data_y, data_z = [data_dict[plot_name][:, i] for i in range(3)]
        # ax.set_aspect("equal")  # not supported
        # weight
        weights_xyz = weight_dict.get(plot_name)
        if weights_xyz is None:
            w_x, w_y, w_z = [np.ones(1), np.ones(1), np.ones(1)]  # broadcasting
        else:
            w_x, w_y, w_z = [weights_xyz[i] for i in range(3)]
        X, Y, Z = w_x*data_x, w_y*data_y, w_z*data_z
        # plot properties
        plot_property_dict = {}
        for key in ["c", "label", "alpha"]:
            plot_property_dict[key] = _get_plot_property(fig_dict, key, i_plt)
        plot_type = fig_dict.get("type")
        if plot_type is None:
            ax.plot(X, Y, Z, **plot_property_dict)  # default
        elif plot_type[i_plt] == "scatter":
            ax.scatter(X, Y, Z, **plot_property_dict)
        else:
            ax.plot(X, Y, Z, **plot_property_dict)  # default
        # label, lim
        if fig_dict.get("xlabel") is not None:
            ax.set_xlabel(fig_dict["xlabel"])
        if fig_dict.get("ylabel") is not None:
            ax.set_ylabel(fig_dict["ylabel"])
        if fig_dict.get("zlabel") is not None:
            ax.set_zlabel(fig_dict["zlabel"])
        if fig_dict.get("xlim") is not None:
            xlim = fig_dict["xlim"]
            ax.set_xlim3d(*xlim)
        else:
            xlim = [X.min(), X.max()]
        if fig_dict.get("ylim") is not None:
            ylim = fig_dict["ylim"]
            ax.set_ylim3d(*ylim)
        else:
            ylim = [Y.min(), Y.max()]
        if fig_dict.get("zlim") is not None:
            zlim = fig_dict["zlim"]
            ax.set_ylim3d(*zlim)
        else:
            zlim = [Z.min(), Z.max()]
        if fig_dict.get("axis") == "equal":
            lims = [xlim, ylim, zlim]
            _axis_equal(ax, lims, projection="3d")
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
            X, Y = w_x*data_x, w_y*data_y[:, i]
            # plot properties
            plot_property_dict = {}
            for key in ["c", "label", "alpha"]:
                plot_property_dict[key] = _get_plot_property(fig_dict, key, i_plt)
            plot_type = fig_dict.get("type")
            if plot_type is None:
                ax[i].plot(X, Y, **plot_property_dict)  # default
            elif plot_type[i_plt] == "scatter":
                ax[i].scatter(X, Y, **plot_property_dict)
            else:
                ax[i].plot(X, Y, **plot_property_dict)  # default
            # label, lim
            if fig_dict.get("xlabel") is not None:
                ax[i].set_xlabel(fig_dict["xlabel"])
            if fig_dict.get("ylabel") is not None:
                ax[i].set_ylabel(fig_dict["ylabel"][i])
            if fig_dict.get("xlim") is not None:
                ax[i].set_xlim(*fig_dict["xlim"])
            if fig_dict.get("ylim") is not None:
                ax[i].set_ylim(*fig_dict["ylim"][i])
            if fig_dict.get("axis") == "equal":
                _axis_equal(ax[i])
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


def _axis_equal(ax, lims=None, projection="2d"):
    if projection == "2d":
        ax.axis("equal")
    elif projection == "3d":
        xlim, ylim, zlim = lims
        # ax should be a 3d figure
        # Create cubic bounding box to simulate equal aspect ratio
        max_range = np.array([xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0]]).max()
        Xb = 0.5*max_range*np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5*(xlim[1]+xlim[0])
        Yb = 0.5*max_range*np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5*(ylim[1]+ylim[0])
        Zb = 0.5*max_range*np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5*(zlim[1]+zlim[0])
        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax.plot([xb], [yb], [zb], 'w')


if __name__ == '__main__':
    pass
