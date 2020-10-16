class Diff:
    """
        Simple dummy class that just returns the difference between two consecutive snapshots 
    """
    def __init__(self, *args, **kwargs):
        self.x = None
        return 

    def stream(self,y):
        if self.x is None:
            self.x = y
            return False
        
        self.diff = 2*y-self.x
        self.x = y
        return True

    def compute_modes(self, *args):
        return 1, None

    def reconstruct(self, t):
        return self.diff