import numpy as np
import pandas as pd

def ahp_weights(pairwise_matrix: np.ndarray) -> np.ndarray:
    """Compute AHP weights via principal eigenvector"""
    eigvals, eigvecs = np.linalg.eig(pairwise_matrix)
    max_idx = np.argmax(eigvals)
    w = np.real(eigvecs[:, max_idx])
    return w / w.sum()

if __name__ == '__main__':
    # Example pairwise for 3 criteria
    mat = np.array([[1,2,4],[0.5,1,3],[0.25,0.333,1]])
    w = ahp_weights(mat)
    print('Weights:', w)