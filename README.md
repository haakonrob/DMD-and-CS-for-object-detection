# DMD-and-CS-for-object-detection

This repo contains my investigations into how Dynamic Mode Decomposition and Compressed Sesning can be applied to video object detection.

## Installation
To install, `cd` into the top directory and run:
```
pip install -e .
```
The `-e` flag installs the package in editable mode, so you can keep developing the package without having to reinstall it.


## Data
The dataset used is called the [Change Detection dataset](http://jacarini.dinf.usherbrooke.ca/dataset2014/). Specifically, we use the "pedestrians" category. To download this and unpack it, run the following script:

```bash
bin/fetch_dataset_change_detection
```

The `matlab` folder contains code and examples taken from the Dynamic Mode Decomposition book by Kutz et al, implemented in MATLAB.




