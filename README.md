# Aerodynamic Simulator for Testing Deep RL Algorithms

## Installation

- Clone the repo and change the directory to it
  ```
	git clone https://github.com/fdcl-nrf/fym.git
	cd fym
	```

- Install the fym package
	```
	pip install -e .
	```


## Plotting module
- File name: ``plotting.py``
- How to use
	1. import fym.plotting
	2. load data as dictionary
	3. set ``draw_dict``
	4. set ``weight_dict``
	5. set ``save_dir`` (optional)
	6. set ``option`` (optional)
- How to set
	1. set ``draw_dict`` example
		```
		draw_dict = {
		    "position": {                   # figure name
		    "projection": "2d",             # 2d or 3d
		    "plot": [["time", "position"]], # If 2d, the form of value is as
						    [[key for x-axis, key for y-axis]].
						    `key for y-axis` could be series of
						    vector. It will split in subplot.
						    If 3d, [key of data] is enough.
						    The data has to be series of 3-dim
						    vectors.
		    "type": None,                   # none or scatter
						    It has to be list like
						    [None, "scatter", ...] for subplot.
						    This is the same since.
		    "label": "x",                   # legend. If there are several
						    legend. It has to be list like
						    ["legend1", "legend2", ...].
		    "c": "r",                       # color
		    "alpha": 1,                     # transparent. 0 is perfectly
						    transparent, 1 is non-transparent.
		    "xlabel": "time [s]",           # label for x-axis.
		    "ylabel": "position [m]",       # label for y-axis.
		    "xlim": [0, config.FINAL_TIME], # limitation of x-axis.
		    "ylim": [-5, 5],                # limitation of y-axis.
		    "axis": None, # none or equal   # If set equal, axis will be equal.
		},
		```
	2. set ``weight_dict``
	
	3. set ``save_dir``
	
	4. set ``option``
	
