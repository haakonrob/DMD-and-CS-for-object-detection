import numpy as np


class SlidingDMD:
    def __init__(self, N=60, max_rank = 10, dt=1):
        self.N = N
        self.X, self.Y = None, None
        self.XX = None
        self.dt = dt
        self.max_rank = max_rank
        self.iters = 0

    def stream(self,x,y):
        self.iters += 1
        # Initialise data as columns matrices
        # x = np.matrix(x)
        # y = np.matrix(y)
        # if x.shape[0] == 1: x = x.T
        # if y.shape[0] == 1: y = y.T

        if self.XX is None:
            self.XX = np.zeros((y.shape[0], self.N))

        self.XX[:,:-1] = self.XX[:,1:]
        self.XX[:,-1] = y

        # After this point, its just normal DMD
        self.X, self.Y = self.XX[:,:-1], self.XX[:,1:]

        if self.iters < self.N:
            return False

        # Perform SVD on the data and 
        U,S,V = np.linalg.svd(self.X, full_matrices=False)
        V = V.conj().T  # This was killing me, why does numpy transpose this??

        # Truncate to r dimensions
        r = self.max_rank
        self.U_, self.S_, self.V_ = U[:,:r], S[:r], V[:,:r]

        # Calculate the reduced dim A and the eigvals / eigvecs
        self.A_ = (self.U_.T.conj() @ self.Y @ self.V_)  * np.reciprocal(np.vstack(self.S_))
        self.L, self.W = np.linalg.eig(self.A_)

        return True
        
    def compute_modes(self):
        # Compute the DMD modes
        self.phi = ((self.Y @ self.V_) * np.reciprocal(self.S_))  @ self.W
        # self.phi = self.U_ @ self.W

        return self.phi

    def reconstruct(self, t, nmodes=None):
        if nmodes is None or nmodes > self.phi.shape[1]:
            r = self.phi.shape[1]
        else:
            r = nmodes    
        # Find the initial mode amplitudes
        b, _, _, _ = np.linalg.lstsq(self.phi[:,:r], self.X[:,0], rcond=None)
        omega = np.log(np.complex128(self.L[:r]))/self.dt  # complex128 because sometimes there are negative real eigvals 
        time_dynamics = np.exp(np.outer(omega, t))
        self.td = time_dynamics
        return self.phi[:,:r] @ (time_dynamics * b[:,None])



class StreamingSnapshots:
    def __init__(self, N=60, max_rank = 10, dt=1):
        self.N = N
        self._X = None
        self.XX = None
        self.dt = dt
        self.max_rank = max_rank
        self.iters = 0

    def stream(self,x,y):
        self.iters += 1
        # Initialise data as columns matrices
        # x = np.matrix(x)
        # y = np.matrix(y)
        # if x.shape[0] == 1: x = x.T
        # if y.shape[0] == 1: y = y.T

        if self._X is None:
            self._X = np.zeros((y.shape[0], self.N))

        self._X[:,:-1] = self._X[:,1:]
        self._X[:,-1] = y

        if self.iters < self.N:
            return False

        if self.XX is None:
            self.XX = self._X[:,:-1].T @ self._X[:,:-1]
        else:
            self.XX[:-1,:-1] = self.XX[1:,1:]
            self.XX[:,-1] = self._X[:,:-1].T @ y
            self.XX[-1,:] = self.XX[:,-1]

       
        self.X, self.Y = self._X[:,:-1], self._X[:,1:]

        # Perform SVD on the data and 
        # U,S,V = np.linalg.svd(self.X, full_matrices=False)
        # V = V.conj().T  # This was killing me, why does numpy transpose this??
        V, L = np.linalg.eig(self.XX)
        S = np.sqrt(abs(L)) 
        U = (self.X @ V) * np.reciprocal(S)
        
        
        # Truncate to r dimensions
        r = self.max_rank
        self.U_, self.S_, self.V_ = U[:,:r], S[:r], V[:,:r]

        # Calculate the reduced dim A and the eigvals / eigvecs
        self.A_ = (self.U_.T.conj() @ self.Y @ self.V_)  * np.reciprocal(np.vstack(self.S_))
        self.L, self.W = np.linalg.eig(self.A_)

        return True
        
    def compute_modes(self):
        # Compute the DMD modes
        self.phi = ((self.Y @ self.V_) * np.reciprocal(self.S_))  @ self.W
        # self.phi = self.U_ @ self.W

        return self.phi

    def reconstruct(self, t, nmodes=None):
        if nmodes is None or nmodes > self.phi.shape[1]:
            r = self.phi.shape[1]
        else:
            r = nmodes    
        # Find the initial mode amplitudes
        b, _, _, _ = np.linalg.lstsq(self.phi[:,:r], self.X[:,0], rcond=None)
        omega = np.log(np.complex128(self.L[:r]))/self.dt  # complex128 because sometimes there are negative real eigvals 
        time_dynamics = np.exp(np.outer(omega, t))
        self.td = time_dynamics
        return self.phi[:,:r] @ (time_dynamics * b[:,None])




class StreamingSVD:
    def __init__():
        pass