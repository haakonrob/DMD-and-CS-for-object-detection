import cv2 as cv

class KNN:
    def __init__(self):
        self.model = cv.createBackgroundSubtractorKNN()

    def apply(self, frame):
        return self.model.apply(frame)