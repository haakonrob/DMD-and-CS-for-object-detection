import csdmd.webcam as webcam
import sys

if len(sys.argv) > 1:
    vid = sys.argv[1]
else:
    vid = 0
    
webcam.run(vid)


