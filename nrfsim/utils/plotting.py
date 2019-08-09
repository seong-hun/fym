"""
plotting.py:
simple module for plot figures

input: (data, type) -> output: figures
data includes obs, state, control, etc.
"""
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d


class PltModule:
    units = {'time': 's', 'distance': 'm', 'speed': 'm/s', 'angle': 'deg'}

    def __init__(self, time_series: float, data: dict, variables: dict, quantities: dict) -> None:
        self.time_series = time_series
        self.data = data
        self.variables = variables
        self.quantities = quantities

    def plot_time(self, labels: tuple) -> None:
        for label in labels:
            plt.figure()
            for i in range(len(self.quantities[label])):
                plt.subplot(len(self.quantities[label]), 1, i+1)
                if self.quantities[label][i] == 'angle':
                    plt.plot(self.time_series, np.rad2deg(self.data[label][:, i]))
                else:
                    plt.plot(self.time_series, self.data[label][:, i])
                plt.xlabel('t' + ' [' + self.units['time'] + ' ]')
                plt.ylabel(self.variables[label][i] + ' [' + self.units[self.quantities[label][i]] + ' ]')
        plt.show()

    def plot_traj(self, labels: tuple) -> None:
        if 'traj' in labels:
            plt.figure()
            x = self.data['traj'][:, 0]
            y = self.data['traj'][:, 1]
            if len(self.variables['traj']) > 3:
                print('Trajectory cannot be displayed in more than 3-dimensions.')
            elif len(self.variables['traj']) == 2:
                plt.plot(x, y)
                plt.xlabel(self.variables['traj'][0] + ' [' + self.units[self.quantities['traj'][0]] + ' ]')
                plt.ylabel(self.variables['traj'][1] + ' [' + self.units[self.quantities['traj'][1]] + ' ]')
            elif len(self.variables['traj']) == 3:
                z = self.data['traj'][:, 2]
                ax = plt.axes(projection='3d')
                plt.plot3D(x, y, z)
                plt.xlabel(self.variables['traj'][0] + ' [' + self.units[self.quantities['traj'][0]] + ' ]')
                plt.ylabel(self.variables['traj'][1] + ' [' + self.units[self.quantities['traj'][1]] + ' ]')

                plt.zlabel(self.variables['traj'][2] + ' [' + self.units[self.quantities['traj'][2]] + ' ]')
            plt.show()
