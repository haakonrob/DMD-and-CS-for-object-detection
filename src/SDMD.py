import os
import sys
import imageio
import numpy as np
import skimage
from skimage import io
from skimage import transform
import matplotlib.pyplot as plt

class SDMD:
    def __init__(self, max_rank=None, ngram=5, epsilon=np.finfo(float).eps):
        self.Qx = None
        self.Qy = None
        self.Gx = None
        self.Gy = None
        self.A = None
        self.max_rank = max_rank   # Maximum rank that our bases can have
        self.ngram = ngram         # Number of times to reapply Gram-Schmidt
        self.precision = epsilon
    
    def stream(self,x,y):
        normx = np.linalg.norm(x)
        normy = np.linalg.norm(y)
            
        # The data stream might contain a blank input
        if normx < self.precision or normy < self.precision:
            return 
        
        # Initialise data as columns matrices
        x = np.matrix(x)
        y = np.matrix(y)
        if x.shape[0] == 1: x = x.T;
        if y.shape[0] == 1: y = y.T;
            
        # Initialise bases on first iteration
        if self.Qx is None or self.Qy is None:
            self.Qx = x / normx
            self.Qy = y / normy
            self.Gx = np.matrix(normx**2)
            self.Gy = np.matrix(normy**2)
            self.A = np.matrix(normx * normy)
            return
        
        # Gram-Schmidt reorthonormalization
        xtilde = np.zeros((self.Qx.shape[1],1))
        ytilde = np.zeros((self.Qy.shape[1],1))
        ex = x
        ey = y
        for _ in range(self.ngram):
            dx = self.Qx.T @ ex
            dy = self.Qy.T @ ey
            xtilde = xtilde + dx
            ytilde = ytilde + dy
            ex = ex - self.Qx @ dx
            ey = ey - self.Qy @ dy
            
        # If the residuals are not described by the current bases, we need to update the bases 
        # to accomodate the new data. We then also need to zero pad the G and A matrices 
        if np.linalg.norm(ex) / normx > self.precision:
            self.Qx = np.concatenate((self.Qx, ex), axis=1)
            self.Gx = np.pad(self.Gx, ((0,1),(0,1)))
            self.A = np.pad(self.A, ((0,0),(0,1)))
            
        if np.linalg.norm(ey) / normy > self.precision:
            self.Qy = np.concatenate((self.Qy, ey), axis=1)
            self.Gy = np.pad(self.Gy, ((0,1),(0,1)))
            self.A = np.pad(self.A, ((0,1),(0,0)))
            
        # If the rank is too high, then we need to compress the matrices again using POD
        if self.max_rank is not None:
            if self.Qx.shape[1] > self.max_rank:
                eigval, eigvec  = np.linalg.eig(self.Gx)
                idxs = np.argsort(-eigval)
                qx = eigvec[:,idxs[:self.max_rank]]
                self.Qx = self.Qx @ qx
                self.A  = self.A @ qx
                self.Gx = np.diag(eigval[idxs[:self.max_rank]]);
            if self.Qy.shape[1] > self.max_rank:
                eigval, eigvec = np.linalg.eig(self.Gy)
                idxs = np.argsort(-eigval)
                qy = eigvec[:,idxs[:self.max_rank]]
                self.Qy = (self.Qy @ qy);
                self.A  = (qy.T @ self.A);
                self.Gy = np.diag(eigval[idxs[:self.max_rank]]);
        
        
        # Update matrices using the outer products of the data pair
        xtilde = (self.Qx.T @ x)
        ytilde = (self.Qy.T @ y)

        # update A, Gx, Gy
        self.Gx = self.Gx + xtilde @ xtilde.T;
        self.Gy = self.Gy + ytilde @ ytilde.T;
        self.A = self.A + ytilde @ xtilde.T;
        
    def compute_modes(self):
#         print(self.Qx, self.Qy, self.A, self.Gx)
        if any(x is None for x in (self.Qx, self.Qy, self.A, self.Gx)):
            return None, None
        Ktilde = self.Qx.T @ self.Qy @ self.A @ np.linalg.pinv(self.Gx) # @ self.A @ np.array(np.linalg.pinv(self.Gx))
#         print(Ktilde.shape)
        eigvals, eigvecs  = np.linalg.eig(Ktilde)
        eigvals = np.matrix(np.diag(eigvals)).T
        modes = self.Qx @ np.vstack(eigvecs)
        return modes, eigvals