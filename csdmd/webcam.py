import cv2 as cv
from time import sleep, time
from math import floor

from .SDMD2 import SDMD
from .STDMD import STDMD
from .StreamingSnapshots import StreamingSnapshots, SlidingDMD
# from .OnlineDMD import OnlineDMD

import numpy as np
import cv2 as cv



viewshape = (360, 240)
last_time = 0
FPS = 120
curr_fps = FPS
T_INTERVAL = 1/FPS
params = dict(max_rank=10, N=10)


algs = [
    SlidingDMD, 
    StreamingSnapshots
]

def fps_limit():
    global last_time, T_INTERVAL, curr_fps
    t = time()
    elapsed = t - last_time
    curr_fps = 0.8*curr_fps + 0.2/elapsed
    if elapsed > T_INTERVAL:
        last_time = t
        return True, curr_fps
    return False, curr_fps

def isDigit(key):
    return 48 <= key <= 57 

def resetAlg(i):
    print(i)
    if i < len(algs):
        return algs[i](**params)
    else:
        print("Invalid algorithm")
        return None

def run():
    downsample = 10


    cap = cv.VideoCapture(0)
    cap.set(cv.CV_CAP_PROP_FPS, 60)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    ret, frame = cap.read()
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    shape = tuple(reversed(frame.shape))
    dshape = lambda: ( floor(shape[0]/downsample), floor(shape[1]/downsample) )

    # print(frame.min(),frame.max(),frame.mean())
    cv.imshow('frame', np.zeros(viewshape))
    
    i = 0
    dmd = resetAlg(i)
    
    x,y = None, None


    while True:
        try:
            key = cv.waitKey(1)
            if key == ord('q'):
                print("quit")
                break
            elif isDigit(key):
                i_ = int(chr(key))
                dmd_ = resetAlg(i_)
                if dmd_ is not None:
                    dmd = dmd_
                    i = i_        
            elif key == ord('w'):
                params["max_rank"] += 1
                dmd = resetAlg(i)
            elif key == ord('a'):
                params["N"] -= 5
                dmd = resetAlg(i)
            elif key == ord('s'):
                params["max_rank"] -= 1
                dmd = resetAlg(i)
            elif key == ord('d'):
                params["N"] += 5
                dmd = resetAlg(i)
            elif key == ord('-'):
                downsample += 1
                dmd = resetAlg(i)
            elif key == ord('='):
                downsample = max(1,downsample-1)
                dmd = resetAlg(i)
            elif key > -1: 
                print(key)
        except Exception as e:
            print(e)

        
        # Capture frame-by-frame
        ret, frame_raw = cap.read()
        
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        # Grayscale frame to reduce compute cost 
        frame = cv.cvtColor(frame_raw, cv.COLOR_BGR2GRAY)
        
        # If we're going too fast, rate limit. This helps with responsiveness to key presses
        toofast, fps = fps_limit()
        # if toofast:
        #     # This almost never happens
        #     continue

        # Convert to grayscale and scale down
        y = cv.resize(frame, dshape()).T.flatten()
        y = np.interp(y, (0, 255), (0,1))

        if x is not None and y is not None:
            ready = dmd.stream(x,y)
            if ready:
                modes, _ = dmd.compute_modes()
                if modes is not None:
                    bg = dmd.reconstruct(0).real.clip(0,1).flatten()
                    objects = (y - bg).real.clip(0,1)

                    bg = bg.reshape(dshape()).T
                    objects = objects.reshape(dshape()).T

                    # bg = np.interp(bg,(0,1), (0,255))
                    # objects = np.interp(objects,(0,1), (0,255))

                    bg = cv.resize(bg, viewshape)
                    objects = cv.resize(objects, viewshape)
            else:
                bg = np.zeros(viewshape).T
                objects = np.zeros(viewshape).T

            # Grab the frame and resize it for the screen
            viewframe = np.interp(cv.resize(frame, viewshape), (0,255), (0,1))
            # print(viewframe.shape,bg.shape,objects.shape)
            # Concatenate the results for a side by side comparison
            im = np.concatenate((viewframe, bg, objects), axis=1)   
            # print(im.min(), im.max(), im.mean())
            # Put info on the page
            im = cv.putText(
                im, 
                "FPS: {:.1f}, rank: {}, #snapshots: {}, Downsample: {}".format(fps, params["max_rank"], params["N"], downsample), 
                org=(0,20), 
                fontFace=cv.FONT_HERSHEY_SIMPLEX,  
                fontScale=0.5, 
                color=(0,0,0), 
                thickness=2)
            cv.imshow('frame', im)

        # Shift
        x = y
        # sleep(0.1)

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()