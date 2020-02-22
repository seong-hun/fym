import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import OrderedDict

import fym


class Plotter:
    figures = OrderedDict()  # dictionary for figures
    tmp_name = 0

    def __init__(self):
        pass

    def _plot_raw(self, ax, plot_type, *args):
        if plot_type == "plot":
            result = ax.plot(*args)
        elif plot_type == "step":
            result = ax.step(*args)
        else:
            raise ValueError(
                "{} is not a supported plot type.".format(plot_type)
            )
        return result

    def set_cols_and_rows(self, num_figures, ncols, dim):
        nrows = math.ceil(num_figures/ncols)
        if dim == 2:
            fig, ax = plt.subplots(nrows, ncols)
            if nrows == 1:
                ax = np.expand_dims(ax, axis=0)
            if ncols == 1:
                ax = np.expand_dims(ax, axis=1)
        elif dim == 3:
            raise ValueError("3d not supported.")
        return fig, ax, nrows

    def fit_shape(self, *args):
        if len(args) == 2:
            x, y = args
        elif len(args) == 3:
            x, y, z = args
            # check z
            if x.shape[0] != z.shape[0]:
                raise ValueError(
                    "The length of x must agree with those of z's."
                )
            if len(z.shape) == 1:
                z = np.expand_dims(z, axis=1)

        # check y
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                "The length of x must agree with those of y's."
            )
        if len(y.shape) == 1:
            y = np.expand_dims(y, axis=1)
        # result
        if len(args) == 2:
            result = [x, y]
        if len(args) == 3:
            result = [x, y, z]
        return result

    def _plot3d(self, ncols, xlabel, ylabels, zlabels, plot_type, *args):
        # shape check
        x, y, z = self.fit_shape(*args)

        # set cols and rows
        num_figures = y.shape[1] * z.shape[1]
        nrows = math.ceil(num_figures/ncols)
        fig = plt.figure()

        # TODO: merge filling subplots of 2d and that of 3d
        # plot 3d figures
        nplt = 0
        for j in range(ncols):  # y
            for i in range(nrows):  # z
                if nplt == num_figures:
                    break
                else:
                    nplt += 1
                    ax = fig.add_subplot(nrows, ncols, nplt, projection='3d')
                    self._plot_raw(ax, plot_type, x, y[:, j], z[:, i])
                    # auto ylabel
                    if len(ylabels) == 1:
                        if y.shape[1] == 1:
                            ax.set_ylabel(ylabels[0])
                        else:
                            ax.set_ylabel(ylabels[0]+'{}'.format(nplt))
                    elif len(ylabels) == y.shape[1]:
                        ax.set_ylabel(ylabels[i])
                    else:
                        raise ValueError(
                            "The number of labels must agree"
                            + " with the number of y's."
                        )
                    # auto zlabel
                    if len(zlabels) == 1:
                        if z.shape[1] == 1:
                            ax.set_zlabel(zlabels[0])
                        else:
                            ax.set_zlabel(zlabels[0]+'{}'.format(nplt))
                    elif len(zlabels) == z.shape[1]:
                        ax.set_zlabel(zlabels[i])
                    else:
                        raise ValueError(
                            "The number of labels must agree"
                            + " with the number of z's."
                        )
            ax.set_xlabel(xlabel)

        return fig, ax

    def _plot2d(self, ncols, xlabel, ylabels, plot_type, *args):
        # shape check
        x, y = self.fit_shape(*args)

        # set cols and rows
        num_figures = y.shape[1]
        fig, ax, nrows = self.set_cols_and_rows(
            num_figures, ncols, len(args)
        )

        # plot 2d figures
        nplt = 0
        for j in range(ncols):
            for i in range(nrows):
                if nplt == num_figures:
                    break
                else:
                    nplt += 1
                    self._plot_raw(ax[i, j], plot_type, x, y[:, i])
                    if len(ylabels) == 1:
                        if y.shape[1] == 1:
                            ax[i, j].set_ylabel(ylabels[0])
                        else:
                            ax[i, j].set_ylabel(ylabels[0]+'{}'.format(nplt))
                    elif len(ylabels) == y.shape[1]:
                        ax[i, j].set_ylabel(ylabels[i])
                    else:
                        raise ValueError(
                            "The number of labels must agree"
                            + " with the number of y's."
                        )
            ax[-1][j].set_xlabel(xlabel)
        return fig, ax

    def plot(
        self, *args, name=None, ncols=1,
        xlabel="x", ylabels=["y"], zlabels=["z"], plot_type="plot",
    ):

        # plot 2d figure
        if len(args) == 2:
            # plot
            fig, ax = self._plot2d(
                ncols, xlabel, ylabels, plot_type,
                *args)
        elif len(args) == 3:
            # ignore plot_type
            if plot_type != "plot":
                plot_type = "plot"
                print("plot_type = {plot_type} is ignored for 3d plot.")
            # plot
            fig, ax = self._plot3d(
                ncols, xlabel, ylabels, zlabels, plot_type,
                *args)
        else:
            raise ValueError("Invalid dimension for plotting.")

        # add an element into figures dictionary
        self.add_dict(fig, ax, name)

    def add_dict(self, fig, ax, name):
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


if __name__ == '__main__':
    pass
    # for more detail, see "../test/plot_figures.py"
