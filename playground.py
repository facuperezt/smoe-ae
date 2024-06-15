#%%
import numpy as np
import matplotlib.pyplot as plt

# pdf of a 2d normal distribution
def pdf(x, y, mu, sigma):
    X = np.stack([x, y]).transpose(1, 2, 0)
    X = X.reshape(-1, 2)
    out = np.exp(-0.5*np.dot(np.dot((X-mu).T, np.linalg.inv(sigma)), X-mu))/(2*np.pi*np.sqrt(np.linalg.det(sigma)))
    return out.reshape(x.shape[0], y.shape[0])

# get two distributions
mu1 = np.array([0.25, 0.25])
sigma1 = np.array([[1, -0.1], [-0.1, 0.5]])
mu2 = np.array([0.75, 0.75])
sigma2 = np.array([[0.5, 0.1], [0.1, 1]])

# get meshgrid
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y)

# get pdfs
Z1 = pdf(X, Y, mu1, sigma1)
Z2 = pdf(X, Y, mu2, sigma2)

# plot
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].contour(X, Y, Z1, levels=10)
axs[0].contour(X, Y, Z2, levels=10)
axs[1].contour(X, Y, Z1+Z2, levels=10)
plt.show()
#%%
import torch
from torch_linear_assignment import batch_linear_assignment

cost = torch.arange(1500*4*4).reshape(-1, 4, 4).cuda()

assignment = batch_linear_assignment(cost)
print(assignment.shape)
# %%
