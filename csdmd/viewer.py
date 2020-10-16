import cv2 as cv
import numpy as np


class cvViewer:
    def __init__(self, shape=(360, 240)):
        self.shape = shape  # The desired size of each given frame in pixels

    def show(self, frames, infostring=""):
        if frames is None:
            cv.imshow('frame', np.zeros(self.shape))
            return 

        # Because we concatenate all given frames later, we just pack singles frames
        # into a tuple here
        if type(frames) is not tuple:
            frames = (frames,)

        # Detect frames that are formatted as (0-255), and rescale to (0-1)
        frames = tuple(
            np.interp(cv.resize(frame, self.shape), (0,255), (0,1)) if frame.max() > 1 else frame
            for frame in frames
        )

        # Resize all given frames to the desired viewing size
        frames = tuple(cv.resize(frame, self.shape) for frame in frames)

        # Concatenate the results for a side by side comparison
        im = np.concatenate(frames, axis=1)   

        
        # Put any desired string onto the page
        im = cv.putText(
            im, 
            infostring, 
            org=(0,20), 
            fontFace=cv.FONT_HERSHEY_SIMPLEX,  
            fontScale=0.5, 
            color=(0,0,0), 
            thickness=2)

        im = cv.putText(
            im, 
            infostring, 
            org=(0,20), 
            fontFace=cv.FONT_HERSHEY_SIMPLEX,  
            fontScale=0.45, 
            color=(255,255,255), 
            thickness=2)

        cv.imshow('frame', im)