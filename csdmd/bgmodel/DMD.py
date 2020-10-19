import numpy as np
# import pydmd

class DMD:
    """
    Computes standard DMD on a given data matrix X. The modes
    and the reconstructed data can be obtained by calling the 
    corresponding methods.
    """
    def __init__(self, dt, max_rank = 10, **kwargs):
        self.dt = dt
        self.max_rank = max_rank

    def fit(self, X):
        self.X, self.Y = X[:,:-1], X[:,1:]

        # Perform SVD on the data and 
        U,S,V = np.linalg.svd(self.X, full_matrices=False)
        V = V.conj().T  # This was killing me, why does numpy transpose this??

        # Truncate to r dimensions
        r = self.max_rank
        self.U_, self.S_, self.V_ = U[:,:r], S[:r], V[:,:r]

        # Calculate the reduced dim A and the eigvals / eigvecs
        self.A_ = (self.U_.T.conj() @ self.Y @ self.V_) * np.reciprocal(self.S_) 
        self.L, self.W = np.linalg.eig(self.A_)
       
            
    def compute_modes(self):
        # Compute the DMD modes
        self.phi = ((self.Y @ self.V_) * np.reciprocal(self.S_))  @ self.W
        # self.phi = self.U_ @ self.W

        # Find the initial mode amplitudes
        self.b, _, _, _ = np.linalg.lstsq(self.phi, self.X[:,0], rcond=None)

        return self.phi, self.b

    def reconstruct(self, t):
        # Reconstruct the data
        # Eigenvalues of A_tilde 
        omega = np.log(np.complex128(self.L))/self.dt  # complex128 because sometimes there are negative real eigvals 
        time_dynamics = np.exp(np.outer(omega, t))
        self.td = time_dynamics
        return self.phi @ (time_dynamics * self.b[:,None])