import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

u1 = np.array([-2.5, 1.5])
s1 = np.array([
    [1, 0.1],
    [1.5, 5],
])

u2 = np.array([2.5, 1.5])
s2 = np.array([
    [1, 0.3],
    [0.1, 5],
])

def distance_between_dist(u1, s1, u2, s2):
    factor = (np.linalg.det(s1)**0.25 * np.linalg.det(s2)**0.25)/np.linalg.det((s1+s2)/2)**0.5
    exponent = (-1/8)*((u1 - u2).T@np.linalg.inv((s1+s2)/2)@(u1 - u2))
    return 1 - (factor * np.exp(exponent))

def combine_dist_complicated(u1, s1, u2, s2, n1, n2):
    u = (n1*u1 + n2*u2)/(n1 + n2)
    s = ((n1**2/(n1+n2)**2)*s1) + ((n2**2/(n1+n2)**2)*s2)
    return u, s

def combine_dist(u1, s1, u2, s2, *args):
    u = (u1+u2)/2
    s = (s1**2 + s2**2)**0.5
    return u, s

p = 1
u, s = combine_dist(u1, s1, u2, s2, 0.1, 0.1)

fig, axs = plt.subplots(figsize=(10, 10))
ax0 = axs
for _u, _s in zip([u1, u2, u], [s1, s2, s]):
    #Create grid and multivariate normal
    x = np.linspace(-10,10,500)
    y = np.linspace(-10,10,500)
    X, Y = np.meshgrid(x,y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X; pos[:, :, 1] = Y
    rv = multivariate_normal(_u, _s, allow_singular=True)

    ax0.contour(X, Y, rv.pdf(pos).reshape(500, 500))
    ax0.set_title(distance_between_dist(u2, s2, u1, s1))
plt.show()