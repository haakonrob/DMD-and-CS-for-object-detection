import cv2 as cv

class MOG2:
    def __init__(self):
        self.model = cv.createBackgroundSubtractorMOG2()

    def apply(self, frame):
        return self.model.apply(frame)
