# from .pipeline.pipeline import Pipeline, scale, processor, Saver, resize, downsample, ProcessEveryNthFrame, SkipEveryNthFrame, StopAfterNFrames
import cv2 as cv
import numpy as np
from math import floor
import os

from .bgmodel.KNN import KNN
from .bgmodel.MOG2 import MOG2
from .bgmodel.StreamingSnapshots import SlidingDMD


PARAMS = {
    'dilation_iterations': 5,
    'filter_kernel_size': 5,
}


# Utility functions
def downsample(frame, factor):
    if not (type(factor) is list or type(factor) is tuple): 
        factor = (factor, factor)
    shape = ( floor(frame.shape[1]/factor[1]) , floor(frame.shape[0]/factor[0]) )
    return cv.resize(frame, shape)


def black_and_white(frame):
    return cv.cvtColor(frame, cv.COLOR_BGR2GRAY)


class Saver:
    """
    Module that can be called to save images as name_%i.ext
    """
    def __init__(self, folder, name='img',ext='.jpg'):
        """
            Folder: where to save.
            name: prefix of the file names. A number will be appended later.
        """
        self.count = 0    
        self.folder = folder
        self.name = name
        self.ext = ext
    
    def __call__(self, frame, i=None):
        """
        Saves each image with an increasing number. You can also override the counter by calling with i.
        """
        if i is None:
            self.count += 1
        else:
            self.count = i

        fp = os.path.join(self.folder, f"{self.name}_{self.count}{self.ext}")
        cv.imwrite(fp, frame)
        return frame



def run(src, dest):
    """
        src: can be an int (0 = webcam usually), or a path to a video that you want to process
        dest: the destination folder where all of the results will be saved.
    """

    cap = cv.VideoCapture(src)

    # This function reads the image source and lets you iterate over the results. When there are
    # no more frames to be read, the iteration will stop
    def get_frames():
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame 
            else:
                break

    # Choose the background model 
    # TODO: let the user call run() with a specific algorithm
    # bgmodel = KNN()
    # bgmodel = MOG2()
    bgmodel = SlidingDMD( N=10, max_rank = 1, dt=1/60)
    save = Saver(folder=dest, name='frame',ext='.jpg')

    # Process the frames
    for i, frame_raw in enumerate(get_frames()):
        if frame_raw is None:
            continue

        # Preprocessing
        frame_raw = downsample(frame_raw, factor=4)
        frame = black_and_white(frame_raw)

        # Apply the background model
        fgmask = bgmodel.apply(frame) 
        fgmask[fgmask>0] = 255

        # Apply filters 
        k = PARAMS['filter_kernel_size']
        fgmask = cv.medianBlur(fgmask, ksize=k)   # KNN and MOG2 yield a lot of noise, so this gets rid of that
        kernel = np.ones((k,k),np.uint8)
        fgmask = cv.dilate(fgmask,kernel,iterations = PARAMS['dilation_iterations'])

        # Multiply each channel of the image with the boolean mask
        frame = np.einsum('ij,ijk->ijk', fgmask > 0, frame_raw)

        # frame = fgmask

        # Save the result
        if (i % 10) == 0:   # Only save every nth frame to save on space
            save(frame, i)
            print(f"Processed frame {i}")

        

