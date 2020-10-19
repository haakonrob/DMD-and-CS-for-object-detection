from time import time


class FPS:
    """
        Keeps track of how many times per second the update() method is called.
        Can be used as a rate-limiter by specifying the max_fps.
    """
    def __init__(self, max_fps=60, alpha=0.8):
        self.last = 0
        self.fps = 0
        self.interval = 1/max_fps if max_fps is not None else 0
        self.alpha = alpha  # smoothing parameter
    
    def update(self):xxxxxxx    
        t = time()
        elapsed = t - self.last
        if elapsed > self.interval:
            self.last = t
            self.fps = self.alpha*self.fps + (1-self.alpha)/elapsed
            return True
        return False