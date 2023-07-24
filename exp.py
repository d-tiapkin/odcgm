"""
A simple example of the ODCGM algorithm on Procrustes problem
===============================================================
Given n pairs of matrices in an array A and B, we want to solve
in parallel the procrustes problems min_X ||XA - B|| where X is
from Stiefel manifold
"""
from time import time

from tqdm import tqdm
import pickle
import torch
import geoopt
from geoopt.optim import RiemannianSGD

from algo import ODCGM_SGD
from landing import LandingSGD


torch.manual_seed(1)

# generate random matrices

time_max = 60
sigma = 0.

n = 100
p = 60
q = 40
A = torch.randn(n, q, q)
B = torch.randn(n, p, q)
init_weights = torch.randn(n, p, q)

# Compute closed-form solution from svd, used for monitoring.
loss_star = None
with torch.no_grad():
    u, _, v = torch.svd(B.matmul(A.transpose(-1, -2)))
    w_star = u.matmul(v.transpose(-1, -2))
    loss_star = ((torch.matmul(w_star, A) - B) ** 2).sum() / n
    loss_star = loss_star.item()

alpha = 5

def generate_odcgm(algo_type):
    def fn(params, lr):
        return ODCGM_SGD(params, lr, alpha=alpha, algorithm_type=algo_type)
    return fn

method_names = [
    "ODCGM",  
    "Reduced ODCGM", 
    "Geometric ODCGM",  
    "Riemannian GD (Euclidean metric)",
    "Riemannian GD (Canonical metric)"
]
methods = [
    generate_odcgm("full"),
    generate_odcgm("reduced"),
    generate_odcgm("geom"),
    RiemannianSGD,
    RiemannianSGD,
]
is_canonical = [
    False,
    False, 
    False,
    False,
    True
]


learning_rate = 0.01
n_epochs = 100000

experiment_results = {}

for method_name, method, is_canon in zip(method_names, methods, is_canonical):
    loss_list = []
    time_list = []
    distance_list = []

    param = geoopt.ManifoldParameter(
        init_weights.clone(), manifold=geoopt.Stiefel(canonical=is_canon)
    )
    with torch.no_grad():
        param.proj_()
    optimizer = method((param,), lr=learning_rate)
    t0 = time()
    with tqdm(total=time_max) as pbar:
        for _ in range(n_epochs):
            if (time() - t0 > time_max+1):
                break
            optimizer.zero_grad()
            res = torch.matmul(param, A) - B + sigma * torch.randn_like(B)
            loss = (res ** 2).sum() / n
            loss.backward()
            optimizer.step()
            time_list.append(time() - t0)
            pbar.update(int(time_list[-1]) - pbar.n)
            loss_list.append(loss.item() - loss_star)
            d = (
                torch.norm((param.data.transpose(-1, -2).matmul(param.data)) - torch.eye(q))
                / n
            )
            distance_list.append(d.item())

    experiment_results[method_name] = (time_list.copy(), distance_list.copy(), loss_list.copy())

with open('results.pkl', 'wb') as f:
    pickle.dump(experiment_results, f)