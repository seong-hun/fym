import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

import fym.logging as logging


BASE_DATA_DIR = "data"

plt.rc("font", **{
    "family": "sans-serif",
    # "sans-serif": ["Helvetica"],
})
# plt.rc("text", usetex=True)
plt.rc("lines", linewidth=1.3)
plt.rc("axes", grid=True)
plt.rc("grid", linestyle="--", alpha=0.8)


canvas = []
fig, axes = plt.subplots(2, 1, sharex=True)
axes[0].set_ylabel(r"$\beta$ [deg]")
axes[1].set_ylabel(r"$r$ [deg/sec]")
axes[1].set_xlabel("Time [sec]")
canvas.append((fig, axes))

fig, axes = plt.subplots(2, 1, sharex=True)
axes[0].set_ylabel(r"$\phi$ [deg]")
axes[1].set_ylabel(r"$p$ [deg/sec]")
axes[1].set_xlabel("Time [sec]")
canvas.append((fig, axes))

# fig, axes = plt.subplots(2, 1, sharex=True)
# axes[0].set_ylabel(r"$\delta_a$ [deg]")
# axes[1].set_ylabel(r"$\delta_r$ [deg]")
# axes[1].set_xlabel("Time [sec]")
# canvas.append((fig, axes))


def get_data(exp, prefix=BASE_DATA_DIR):
    path = os.path.join(prefix, exp + ".h5")
    return logging.load(path)


def plot_single(data, color="k", name=None):
    time = data["time"]
    beta, phi, p, r = data["state"]["main"][:, :4].T
    # aileron, rudder = data["control"].T

    canvas[0][1][0].plot(time, beta, color=color, label=name)
    canvas[0][1][1].plot(time, r, color=color)

    canvas[1][1][0].plot(time, phi, color=color, label=name)
    canvas[1][1][1].plot(time, p, color=color)

    # canvas[2][1][0].plot(time, aileron, color=color, label=name)
    # canvas[2][1][1].plot(time, rudder, color=color)


def plot_mult(dataset, color_cycle=None, names=None):
    if color_cycle is None:
        color_cycle = cycler(
            color=plt.rcParams["axes.prop_cycle"].by_key()["color"]
        )

    if names is not None:
        for data, color, name in zip(dataset, color_cycle(), names):
            plot_single(data, color=color["color"], name=name)

        for fig, axes in canvas:
            axes[0].legend(*axes[0].get_legend_handles_labels())
    else:
        for data, color in zip(dataset, color_cycle()):
            plot_single(data, color=color["color"])

    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("file", nargs="*")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    color_cycle = cycler(color=plt.rcParams["axes.prop_cycle"].by_key()["color"])

    path_list = args.file
    if args.all:
        path_list = sorted(glob.glob(os.path.join(BASE_DATA_DIR, "*.h5")))
    else:
        path_list = sorted(filter(lambda x: x.endswith(".h5"), path_list))

    dataset = []
    for path, c in zip(path_list, color_cycle()):
        dataset.append(logging.load(path))

    names = [os.path.basename(os.path.splitext(path)[0]) for path in path_list]

    print("Plotting ...")
    print("\n".join(path_list))
    plot_mult(dataset, color_cycle, names=names)
