from time import time


class FPS:
    def __init__(self, fps=60, alpha=0.8):
        self.last = 0
        self.fps = fps
        self.interval = 1/fps
        self.alpha = alpha  # smoothing parameter
    
    def update(self):
        t = time()
        elapsed = t - self.last
        if elapsed > self.interval:
            self.last = t
            self.fps = self.alpha*self.fps + (1-self.alpha)/elapsed
            return True
        return False