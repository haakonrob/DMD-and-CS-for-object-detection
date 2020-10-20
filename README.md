# DMD-and-CS-for-object-detection

This repo contains my investigations into how Dynamic Mode Decomposition and Compressed Sesning can be applied to video object detection.

## Installation
To install, `cd` into the top directory and run:
```
pip install -e .
```
The `-e` flag installs the package in editable mode, so you can keep developing the package without having to reinstall it.

## Usage
After installing the package, you can run the main demo like this:
```
csdmd play
```
This runs the code on the system webcam (if there is no webcam, it'll fail). Alternatively, the code can be run on a video file by giving the path:
```
csdmd play -s path/to/my/video
```
where the `-s` is short for `--source`. This will run a looping demo where you can experiment with some of the different background models. The controls for this demo are:

- (q): quit
- (w/s): increase/decrease the maximum rank of the method (applies to DMD methods only)
- (d/a): increase/decrease the number of frames to analyse at a time (applies to DMD only)
- (t): change the prediction time (applies only to DMD, instead of predicting the objects of the last frame, it tries to predict the next frame to reduce delay)
- (+/-): increase/decrease the resolution of the image via downsampling.

If you instead want to process a webcam stream or a video, you can use:
```
csdmd process -d save/output/here
```
or
```
csdmd process -s path/to/my/video -d save/output/here
```
where the `-d` is short for `--destination`.


## Data
The dataset used is called the [Change Detection dataset](http://jacarini.dinf.usherbrooke.ca/dataset2014/). Specifically, we use the "pedestrians" category. To download this and unpack it, run the following script:

```bash
bin/fetch_dataset_change_detection
```

The `matlab` folder contains code and examples taken from the Dynamic Mode Decomposition book by Kutz et al, implemented in MATLAB.




