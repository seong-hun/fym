import numpy as np
import math
import matplotlib.pyplot as plt
from collections import OrderedDict

import fym


class Plotter:
    figures = OrderedDict()  # dictionary for figures
    tmp_name = 0

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
                    ax[i, j].plot(x, y[:, i])
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


if __name__ == '__main__':
    import fym.logging
    data = fym.logging.load('data/main/result.h5')  # result obtained from fym.logging

# data consists of three keys: state, action, time
    state = data['state']
# e.g., system name is "main".
# Note: state consists of keys corresponding to each systems.
    ctrl = data['control']
    time = data['time']

    plotter = Plotter()
    plotter.plot2d(time, state)  # tmp
