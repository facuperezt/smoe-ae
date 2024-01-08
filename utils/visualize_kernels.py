#%%
from typing import List
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

__all__ = [
    "plot_kernels",
    "plot_kernel_centers",
]

# Funtion that upsamples image by factor of n

def _upsample(img, n):
    img = np.repeat(img, n, axis=0)
    img = np.repeat(img, n, axis=1)
    return img

# Function that plots the one sigma contour of a multivariate gaussian distribution
    
def _plot_gaussian_contour(mean: np.ndarray, cov: np.ndarray, ax: plt.Axes, color='r', linewidth: int = 10, alpha: float = 1, res: int = 1000, calls: int = 0) -> None:
    if calls > 10:
        return
    m_x, m_y = mean
    (a,_), (b,c) = cov
    l1 = ((a+c)/2) + (((a-c)/2)**2 + b**2)**0.5
    l2 = ((a+c)/2) - (((a-c)/2)**2 + b**2)**0.5

    if l2 < 0:
        print("Invalid Covariance matrix.")
        print(f"{b=}")
        m = min(np.abs(a), np.abs(c))
        b = np.clip(b, -m, m)
        b *= 0.99
        cov = np.array([
            [a,0],
            [b,c],
        ])
        _plot_gaussian_contour(mean, rectify_covariance_matrix(cov), ax, color=color, linewidth=linewidth, alpha=alpha, res=res, calls=calls+1)

    if b == 0 and a >= c:
        phi = 0
    elif b == 0 and a < c:
        phi = np.pi/2
    else:
        phi = np.arctan2(l1 - a, b)

    t = np.linspace(0, 2*np.pi, res)
    x = np.sqrt(l1) * np.cos(phi) * np.cos(t) - np.sqrt(l2) * np.sin(phi) * np.sin(t)
    y = np.sqrt(l1) * np.sin(phi) * np.cos(t) + np.sqrt(l2) * np.cos(phi) * np.sin(t)

    x += m_x
    y += m_y

    ax.plot(x,y, color=color, linewidth=linewidth, alpha=alpha)


def plot_kernels(smoe_vector: torch.Tensor, ax: plt.Axes, padding: List[int] = None, n_kernels: int = 4) -> None:
    """
    Plots the kernels of a smoe vector in ax.
    """
    if padding is None:
        padding = [0, 0, 0, 0]
    smoe_vector = smoe_vector.detach().cpu().numpy()
    means_x = smoe_vector[:n_kernels] + padding[0]
    means_y = smoe_vector[n_kernels:2*n_kernels] + padding[2]
    nus = smoe_vector[2*n_kernels:3*n_kernels]
    covs = smoe_vector[3*n_kernels:].reshape(-1, 2, 2)
    covs = np.tril(covs)
    for mx, my, nu, cov in zip(means_x, means_y, nus, covs):
        _plot_gaussian_contour(np.array([mx, my]), cov, ax, alpha=nu)


def plot_kernel_centers(smoe_vector: torch.Tensor, ax: plt.Axes, padding: List[int] = None, n_kernels: int = 4) -> None:
    """
    Plots the kernels of a smoe vector in ax.
    """
    if padding is None:
        padding = [0, 0, 0, 0]
    smoe_vector = smoe_vector.detach().cpu().numpy()
    means_x = smoe_vector[:n_kernels] + padding[0]
    means_y = smoe_vector[n_kernels:2*n_kernels] + padding[2]
    for mx, my in zip(means_x, means_y):
        ax.scatter(mx, my, color="r", marker="x", s=50)


def rectify_covariance_matrix(cov):
    """
    Rectify a covariance matrix to ensure it is symmetric and positive semidefinite.

    Parameters:
    cov (np.array): A covariance matrix that might be invalid.

    Returns:
    np.array: A rectified, valid covariance matrix.
    """
    if (cov[0,0] < 0) and (cov[1,1]) < 0 and (cov[1,0] > 0):
        cov = -cov

    # Ensure the matrix is symmetric
    if cov[0,1] == 0:
        cov[0,1] += cov[1,0]
        cov_sym = cov
    else:
        cov_sym = (cov + cov.T) / 2

    # Perform eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_sym)

    # Set any negative eigenvalues to a small positive value (e.g., 1e-10)
    eigenvalues_rectified = np.clip(eigenvalues, 1e-9, None)

    # Reconstruct the matrix
    cov_rectified = eigenvectors @ np.diag(eigenvalues_rectified) @ eigenvectors.T

    return cov_rectified



if __name__ == "__main__":
    # fig, ax = plt.subplots()
    # for b in np.arange(-1, 1.1, .02):
    #     plot_gaussian_contour(np.array([0,0]), np.array([[1,b], [b,1]]), ax)

    # x_lim, y_lim = plt.xlim(), plt.ylim()
    # xy_lim = max(*x_lim, *y_lim)
    # plt.xlim(-xy_lim, xy_lim)
    # plt.ylim(-xy_lim, xy_lim)
    img = np.random.uniform(0, 1, (16,16))
    n = 100
    # img = _upsample(img, n)
    mean = np.array([7.5,7.5])
    b = 1
    cov = np.array([
        [-4,0],
        [b,-2],
        ]
    )
    fig, ax = plt.subplots()
    ax: plt.Axes
    ax.imshow(img)
    _plot_gaussian_contour(mean, rectify_covariance_matrix(cov), ax, alpha=0.4)
# %%
