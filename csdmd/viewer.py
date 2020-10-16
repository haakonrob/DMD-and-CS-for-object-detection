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
 
        # Put given string into the image
        im = self.putText(im, infostring)

        # Draw the frame
        cv.imshow('frame', im)

    def putText(self,im,text,org=(0,20),fontFace=cv.FONT_HERSHEY_PLAIN,fontScale=1,thickness=1):
        # We draw the text twice with differenct thicknesses to get
        # white text with a black border
        im = cv.putText(
            im, 
            text, 
            org=org, 
            fontFace=fontFace,  
            fontScale=fontScale, 
            color=(0,0,0), 
            thickness=thickness+3
        )
        im = cv.putText(
            im, 
            text, 
            org=org, 
            fontFace=fontFace,  
            fontScale=fontScale, 
            color=(255,255,255), 
            thickness=thickness
        )
        return im