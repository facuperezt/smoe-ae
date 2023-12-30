#%%
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

__all__ = [
    "plot_gaussian_contour",
]

# Funtion that upsamples image by factor of n

def _upsample(img, n):
    img = np.repeat(img, n, axis=0)
    img = np.repeat(img, n, axis=1)
    return img

# Function that plots the one sigma contour of a multivariate gaussian distribution
    
def plot_gaussian_contour(mean: np.ndarray, cov: np.ndarray, ax: plt.Axes, color='r', linewidth: int = 10, alpha: float = 1, res: int = 1000):
    m_x, m_y = mean
    (a,b), (_,c) = cov
    l1 = ((a+c)/2) + (((a-c)/2)**2 + b**2)**0.5
    l2 = ((a+c)/2) - (((a-c)/2)**2 + b**2)**0.5

    if l2 < 0:
        print("Invalid Covariance matrix.")
        print(f"{b=}")
        return

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
    b = -1
    cov = np.array([
        [4,b],
        [b,10],
        ]
    )
    fig, ax = plt.subplots()
    ax: plt.Axes
    ax.imshow(img)
    plot_gaussian_contour(mean, cov, ax, alpha=0.4)
# %%
