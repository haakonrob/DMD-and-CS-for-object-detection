import numpy as np


class StreamingSnapshots:
    def __init__(self, N=10):
        self.iters = 0
        self.N = N
        self.X = None
        self.XX = None

    def stream(self, x):
        

        U = self.X @ V @ Σ_

        A_ = U.conj.T @ 

class StreamSVD:
    def __init__(self, N=10):
        self.iters = 0
        self.N = N
        self.X = None
        self.XX = None

    def update(self, x):
        if self.X is None:
            self.X = np.zeros(len(x), self.N)

        self.iters += 1
        if self.iters < self.N:
            self.X[:,self.iters] = x
            return None, None

        self.XX[:-1,:-1] = self.XX[1:,1:]
        self.XX[:,-1] = self.X.T @ self.X[:,-1]
        self.XX[-1,:] = self.XX[:,-1].T
        Λ, V = np.linalg.eig(self.XX)

        Σ = np.sqrt(abs(Λ))
        idxs = np.argsort(-Σ_)

        return Σ[idxs], V[:,idxs], 





