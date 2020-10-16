import numpy as np


class STDMD:
    """
    Implementation of the Streaming Total DMD (STDMD) algorithm from 
    "De-biasing the Dynamic Mode Decomposition for Applied Koopman Spectral Analysis" 
    (Hemati et al 2016)
    """
    def __init__(self, max_rank=None, ngram=5, epsilon=np.finfo(float).eps):
        self.x = None
        self.Qx = None
        self.Qy = None
        self.Gx = None
        self.Gy = None
        self.A = None
        
        self.Qz = None
        self.Gz = None
        self.max_rank = max_rank   # Maximum rank that our bases can have
        self.ngram = ngram         # Number of times to reapply Gram-Schmidt
        self.precision = epsilon
    
    def stream(self, y):
        x = self.x
        if x is None:
            self.x = y
            return False
        
        n = x.ravel().shape[0]
        x = np.asmatrix(x).reshape((n,1))
        y = np.asmatrix(y).reshape((n,1))
        
        z = np.concatenate((x,y), axis=0)
        normz = np.linalg.norm(z)
        
        # Initialise bases on first iteration
        if self.Qz is None or self.Gz is None:
            self.Qz = z/normz
            self.Gz = np.matrix(normz**2)
            return False
        
        # Gram-Schmidt reorthonormalisation
        ztilde = np.zeros((self.Qz.shape[1],1))
        ez = z
        for _ in range(self.ngram):
            dz = self.Qz.T @ ez
            ztilde += dz
            ez -= self.Qz @ dz
            
        # Update basis if necessary
        if np.linalg.norm(ez) / normz > self.precision:
            self.Qz = np.concatenate((self.Qz, ez / normz), axis=1)
            self.Gz = np.pad(self.Gz, ((0,1), (0,1)))
        
        ztilde = self.Qz.T @ z
        self.Gz += (ztilde @ ztilde.T)
        
        
        if self.max_rank is not None:
            if self.Qz.shape[1] > self.max_rank:
                eigval, eigvec  = np.linalg.eig(self.Gz)
                idxs = np.argsort(-eigval)
                qz = eigvec[:,idxs[:self.max_rank]]
                self.Qz = self.Qz @ qz
                self.Gz = np.diag(eigval[idxs[:self.max_rank]])

        return True
            
    def compute_modes(self):
        n = self.Qz.shape[0] // 2
        Qx, Rx = np.linalg.qr(self.Qz[:n,:])
        if self.max_rank:
            Qx = Qx[:,:self.max_rank]
            Rx = Rx[:self.max_rank, :]
            
        Gx = Rx @ self.Gz @ Rx.T
        A = Qx.T @ self.Qz[n:, :] @ self.Gz @ self.Qz[:n,:].T @ Qx  # TODO: check that these are all good, I found a type in the sample code
        
        Ktilde = A @ np.linalg.pinv(Gx)
        eigvals, eigvecs = np.linalg.eig(Ktilde)
        modes = Qx @ eigvecs
        idxs = np.argsort(-eigvals)
        return modes[:,idxs], eigvals[idxs]