import cv2 as cv
from time import sleep, time
from math import floor
from traceback import print_tb

from csdmd.fps import FPS
from csdmd.utils import isDigit

from csdmd.streaming.Diff import Diff
from csdmd.streaming.SDMD import SDMD
from csdmd.streaming.STDMD import STDMD
from csdmd.streaming.StreamingSnapshots import StreamingSnapshots, SlidingDMD
# from .OnlineDMD import OnlineDMD

import numpy as np
import cv2 as cv


class cvInputParser:
    def handle_inputs(self, bgmodel):
        new_model = None
        params = None
        will_quit = False

        try:
            key = cv.waitKey(1)

            if isDigit(key):
                i = int(chr(key))
                new_model = i
            elif key == ord('q'):
                will_quit = True 
            elif key == ord('w'):
                params = bgmodel.params
                params["max_rank"] += 1
            elif key == ord('a'):
                params = bgmodel.params
                params["N"] -= 5
            elif key == ord('s'):
                params = bgmodel.params
                params["max_rank"] -= 1
            elif key == ord('d'):
                params = bgmodel.params
                params["N"] += 5
            elif key == ord('-'):
                params = bgmodel.params
                params['downsample'] += 1
            elif key == ord('='):
                params = bgmodel.params
                params['downsample'] = max(1,params['downsample']-1)
            elif key == ord('t'):
                params = bgmodel.params
                params['T'] = 1/30 if params['T'] == 0 else 0
            elif key > -1: 
                print(key)
        except Exception as e:
            print(e)

        return new_model, params, will_quit


class BackgroundModel:
    algs = [
        {'class': StreamingSnapshots, 'type':'DMD', 'args': ['N', 'max_rank', 'dt']}, 
        {'class': SlidingDMD, 'type':'DMD', 'args': ['N', 'max_rank', 'dt']}, 
        {'class': Diff, 'type':'DMD', 'args': []}, 
        {'class': cv.createBackgroundSubtractorMOG2, 'type':'CV', 'args': []}, 
        {'class': cv.createBackgroundSubtractorKNN, 'type':'CV', 'args': []}, 
    ]

    def __init__(self):
        self.x = None
        self.model = None
        self.params = None
        self.bg = None
        self.objects = None
        self.reset(model=0, params=dict(max_rank=10, N=10, downsample=4, T=0, dt=1))

    def reset(self, model=None, params=None):
        if model is not None:
            self.model_i = model
        if params is not None:
            self.params = params

        assert self.model_i < len(self.algs), "Invalid algorithm"

        self.modeltype = self.algs[self.model_i]['type']
        args = self.algs[self.model_i]['args']
        self.model = self.algs[self.model_i]['class'](**{k:self.params[k] for k in args})
        
        
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

                self.bg = bg.reshape(shape).T
                self.objects = objects.reshape(shape).T

                # bg = np.interp(bg,(0,1), (0,255))
                # objects = np.interp(objects,(0,1), (0,255))
        else:
            self.bg = np.zeros(shape).T
            self.objects = np.zeros(shape).T
        
            
    def _update_CV(self, frame):
        shape = self.get_downsampled_shape()    
        # Performance should be better with the raw image
        fgmask = self.model.apply(frame)

        # Median filter, this methods tend to yield a lot of noise
        fgmask = cv.medianBlur(fgmask, 5)

        # Resize the images to the appropriate shape
        self.bg = cv.resize(fgmask, shape)
        self.objects = cv.resize(fgmask, shape)


    def get_downsampled_shape(self):
        return tuple( int(floor(x)/self.params['downsample']) for x in self.frame_shape[:2] )
        

class Viewer:
    def __init__(self, shape=(360, 240), side_by_side=True):
        self.shape = shape
        self.side_by_side = side_by_side

    def show(self, frame, model=None, infostring=""):
        if frame is None or model.bg is None or model.objects is None:
            cv.imshow('frame', np.zeros(self.shape))
            return 

        objects = cv.resize(model.objects, self.shape)
        
        if self.side_by_side:
            # # Resize to the desired viewing size
            frame = np.interp(cv.resize(frame, self.shape), (0,255), (0,1))
            bg = cv.resize(model.bg, self.shape)

            # Concatenate the results for a side by side comparison
            im = np.concatenate((frame, bg, objects), axis=1)   

        else:
            im = objects
        
        # Put info on the page
        im = cv.putText(
            im, 
            infostring, 
            org=(0,20), 
            fontFace=cv.FONT_HERSHEY_SIMPLEX,  
            fontScale=0.5, 
            color=(0,0,0), 
            thickness=2)
        cv.imshow('frame', im)



def run(src=0):
    try:
        cap = cv.VideoCapture(src)
        if not cap.isOpened():
            print("Cannot open camera")
            exit()

        ip = cvInputParser()
        model = BackgroundModel()
        viewer = Viewer(shape = (360,240), side_by_side=True)
        fps = FPS()

        # Initialise with a black screen
        viewer.show(None)

        while True:
            # Handle user inputs for changes to the model or model parameters
            new_model, new_params, will_quit = ip.handle_inputs(model)
            if will_quit:
                break
            elif new_model is not None or new_params is not None:
                model.reset(new_model, new_params)
            
            # Limit the FPS (also necessary for keeping track of the fps)
            toofast = fps.update()
            if toofast:
                # This never happens when using the webcam, because it is IO bound 
                continue

            _, frame_raw = cap.read()
            frame = cv.cvtColor(frame_raw, cv.COLOR_BGR2GRAY)

            model.update(frame)

            # Info to show to overlay on the output
            infostring = "FPS: {:.1f}, rank: {}, #snapshots: {}, Downsample: {}, Predict: {}".format(
                fps.fps, 
                model.params["max_rank"], 
                model.params["N"], 
                model.params['downsample'], 
                "now" if model.params['T'] == 0 else "future"
            )
            
            viewer.show(frame, model, infostring)

    finally:
        # When everything done, release the webcam or video file and close all windows
        cap.release()
        cv.destroyAllWindows()
