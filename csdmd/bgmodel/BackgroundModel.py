import cv2 as cv
import numpy as np
from math import floor

from csdmd.bgmodel.Diff import Diff
from csdmd.bgmodel.SDMD import SDMD
from csdmd.bgmodel.STDMD import STDMD
from csdmd.bgmodel.StreamingSnapshots import StreamingSnapshots, SlidingDMD


# Overview of the implemented algorithms and their arguments. 
# This lets us suddenly switch models via the reset() method.
# The models are indexed by their placement in the list.
algs = [
    {   
        'name': 'Sliding DMD',
        'class': StreamingSnapshots, 
        'type':'DMD', 
        'args': ['N', 'max_rank', 'dt']
    }, 
    {
        'name': 'Difference',
        'class': Diff, 
        'type':'DMD', 
        'args': []
    }, 
    {
        'name': 'MOG2',
        'class': cv.createBackgroundSubtractorMOG2,
         'type':'CV', 
         'args': []
    }, 
    {   
        'name': 'KNN',
        'class': cv.createBackgroundSubtractorKNN,
         'type':'CV', 
         'args': []
    }, 
]

# Supported params and their defaults. Models are initialised with 
# the appropriate parameters from this list
default_params = {
    'max_rank':     10,     # maximum rank of the internal representation
    'N':            10,     # number of snapshots to analyse at a time
    'downsample':   4,      # factor to downscale the raw frames (performance)
    'dt':           1/30,   # The timestep between successive frames
    'T':            0,      # When modelling, model T*dt into the future (helps avoid delay, but reduces accuracy) 
}


class BackgroundModel:
    def __init__(self):
        self.model = None
        self.params = None
        self._bg = None
        self._objects = None
        self.reset(0, default_params)

    def reset(self, new_model=None, new_params=None):
        if new_model is None or new_model < len(algs):
            if new_model is not None:
                self.model_i = new_model
            if new_params is not None:
                self.params = new_params
            self.modeltype = algs[self.model_i]['type']
            args = algs[self.model_i]['args']
            self.model = algs[self.model_i]['class'](**{k:self.params[k] for k in args})
        else:
            print("Invalid algorithm")
        
    @property
    def name(self):
        return algs[self.model_i]['name']

    @property
    def background(self):
        return self._bg

    @property
    def objects(self):
        return self._objects

    def update(self, frame):
        self.frame_shape = frame.shape

        if self.modeltype == 'DMD':
            # Convert to grayscale and scale down
            self._update_DMD(frame)
        elif self.modeltype == 'CV':
            self._update_CV(frame)
    

    def _update_DMD(self,frame): 
        shape = self.get_downsampled_shape()     
        y = cv.resize(frame, shape).T.flatten()
        y = np.interp(y, (0, 255), (0,1))

        ready = self.model.stream(y) 
        if ready:
            modes, _ = self.model.compute_modes()
            if modes is not None:
                bg = self.model.reconstruct(t=self.params['T']).real.clip(0,1).flatten()
                objects = (y - bg).real.clip(0,1)

                self._bg = bg.reshape(shape).T
                self._objects = objects.reshape(shape).T
        else:
            self._bg = np.zeros(shape).T
            self._objects = np.zeros(shape).T
        
            
    def _update_CV(self, frame):
        # Downsample to improve performance
        shape = self.get_downsampled_shape()   
        frame = cv.resize(frame, shape)

        # Performance should be better with the raw image
        fgmask = self.model.apply(frame)

        # Median filter, this methods tend to yield a lot of noise
        fgmask = cv.medianBlur(fgmask, 5)

        self._bg = fgmask
        self._objects = fgmask


    def get_downsampled_shape(self):
        return tuple( int(floor(x)/self.params['downsample']) for x in self.frame_shape[:2] )