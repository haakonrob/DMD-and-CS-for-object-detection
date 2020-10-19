import cv2 as cv
import numpy as np
from time import sleep, time
from math import floor
from traceback import print_tb

from csdmd.inputParser import cvInputParser
from csdmd.viewer import cvViewer

from csdmd.fps import FPS
from csdmd.bgmodel.BackgroundModel import BackgroundModel
# from .OnlineDMD import OnlineDMD


def run(src=0):
    try:
        cap = cv.VideoCapture(src)
        if not cap.isOpened():
            print("Cannot open camera")
            exit()

        ip = cvInputParser()
        ip = cvInputParser()
        model = BackgroundModel()
        viewer = cvViewer(shape = (360,240))
        fps = FPS()

        # Initialise with a black screen
        viewer.show(None)

        while True:
            # Handle user inputs for changes to the model or model parameters
            new_model, new_params, will_quit = ip.handle_inputs(model.params)
            if will_quit:
                break
            elif new_model is not None or new_params is not None:
                model.reset(new_model=new_model, new_params=new_params)
            
            # Limit the FPS (also necessary for keeping track of the fps)
            toofast = fps.update()
            if toofast:
                # This never happens when using the webcam, because it is IO bound 
                continue

            ret, frame_raw = cap.read()
            if not ret:
                # Restart
                cap = cv.VideoCapture(src)
                continue

            frame = cv.cvtColor(frame_raw, cv.COLOR_BGR2GRAY)

            model.update(frame)

            # Info to show to overlay on the output
            infostring = "FPS: {:.1f}, rank: {}, #snapshots: {}, Downsample: {}, Model time: {:.2f}, Algorithm: {}".format(
                fps.fps, 
                model.params["max_rank"], 
                model.params["N"], 
                model.params['downsample'], 
                model.params['T'],
                model.name,
            )
            
            viewer.show((frame, model.background, model.objects), infostring)


    finally:
        # When everything done, release the webcam or video file and close all windows
        cap.release()
        cv.destroyAllWindows()
