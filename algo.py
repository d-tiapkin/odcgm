# Code was inspired by https://github.com/pierreablin/landing/blob/main/landing/optimizer.py
from multiprocessing.sharedctypes import Value
import torch
import numpy as np
import torch.optim.optimizer

import geoopt
from geoopt.tensor import ManifoldParameter, ManifoldTensor
from geoopt.optim.mixin import OptimMixin

__all__ = ["ODCGM_SGD"]


def _check_stiefel(param):
    if not hasattr(param, "manifold"):
        raise TypeError("Parameter should be a geoopt parameter")
    if not isinstance(
        param.manifold, geoopt.manifolds.stiefel.CanonicalStiefel
    ) and not isinstance(
        param.manifold, geoopt.manifolds.stiefel.EuclideanStiefel
    ):
        raise TypeError("Parameters should be on the Stiefel manifold")

@torch.no_grad()
def _get_direction(point, u, alpha):
    *_, p, q = point.shape
    # We start from computing svd
    try:
        U, sigma, Vh = torch.linalg.svd(point)   
    except:                     # torch.svd may have convergence issues for GPU and CPU.
        U, sigma, Vh = torch.linalg.svd(point + 1e-4 * point.mean() * torch.rand_like(point))

    # Next we need to project Z to ()^T @ X + X^T @ () = 0
    distance = torch.matmul(point.transpose(-1, -2), point) - torch.eye(
        q, device=point.device
    )
    B = Vh @ (point.transpose(-1, -2) @ u + u.transpose(-1,-2) @ point) @ Vh.transpose(-1,-2)
    sigma_sums = sigma[..., :, None]**2 + sigma[..., None, :]**2  # TODO: Check me
    Q = B / sigma_sums
    P = Vh.transpose(-1,-2) @ Q @ Vh
    P_sym = 0.5 * (P + P.transpose(-1,-2))
    return u - point @ P_sym - alpha * torch.matmul(point, distance)

@torch.no_grad()
def _get_direction_reduced(point, u, alpha):
    *_, p, q = point.shape
    distance = torch.matmul(point.transpose(-1, -2), point) - torch.eye(
        q, device=point.device
    )
    h = 1/4 * torch.linalg.norm(distance, dim=(-1,-2))**2
    grad_h = torch.matmul(point, distance)
    norm_grad_h = torch.linalg.norm(grad_h, dim=(-1,-2))
    inner_prod = (u * grad_h).sum(dim=(-1,-2))

    lmbd = inner_prod / norm_grad_h**2
    return u - lmbd[...,None,None] * grad_h - alpha * grad_h

@torch.no_grad()
def _get_direction_geom(point, u, alpha):
    *_, p, q = point.shape
    u_t_point = torch.matmul(u, point.transpose(-1, -2))
    psi_u = u_t_point - u_t_point.transpose(-1, -2)
    distance = torch.matmul(point, point.transpose(-1, -2)) - torch.eye(
        p, device=point.device
    )
    return torch.matmul(psi_u - alpha * distance, point)



class ODCGM_SGD(OptimMixin, torch.optim.Optimizer):
    r"""
    Orthogonal Directions Constrained Gradient Method on the Stiefel manifold 
    with the same API as :class:`torch.optim.SGD`.
    Parameters
    ----------
    params : iterable
        iterable of parameters to optimize or dicts defining
        parameter groups. Must contain square orthogonal matrices.
    lr : float
        learning rate
    momentum : float (optional)
        momentum factor (default: 0)
    weight_decay : float (optional)
        weight decay (L2 penalty) (default: 0)
    dampening : float (optional)
        dampening for momentum (default: 0)
    nesterov : bool (optional)
        enables Nesterov momentum (default: False)
    alpha : float (optional)
        the hyperparameter alpha that controls the tradeoff between
        optimization in f and constraint convergence (default: 1.)
    check_type : bool (optional)
        whether to check that the parameters are all orthogonal matrices
    algorithm_type : ['full', 'reduced', 'geom' ], default: 'geom'
        type of algorithms will be used: 
        - 'full' computes Eucliedean projections; 
        - 'reduced' use 1-d projections;
        - 'geom' use change of the Riemannian metric to simplfy projections.
    Other Parameters
    ----------------
    stabilize : int
        Stabilize parameters if they are off-manifold due to numerical
        reasons every ``stabilize`` steps (default: ``None`` -- no stabilize)
    """

    def __init__(
        self,
        params,
        lr,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        stabilize=None,
        alpha=1, 
        check_type=True,
        algorithm_type='geom'
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay)
            )
        if alpha < 0.0:
            raise ValueError(
                "Invalid alpha value: {}".format(alpha)
            )
        if algorithm_type not in ['full', 'reduced', 'geom']:
            raise ValueError(
                "Invalid type of the algorithm {}".format(algorithm_type)
            )
        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            alpha=alpha,
            check_type=check_type,
            algorithm_type=algorithm_type
        )
        for param in params:
            with torch.no_grad():
                param.proj_()
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError(
                "Nesterov momentum requires a momentum and zero dampening"
            )
        super().__init__(params, defaults, stabilize=stabilize)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        with torch.no_grad():
            for group in self.param_groups:
                if "step" not in group:
                    group["step"] = 0
                weight_decay = group["weight_decay"]
                momentum = group["momentum"]
                dampening = group["dampening"]
                nesterov = group["nesterov"]
                learning_rate = group["lr"]
                alpha = group["alpha"]
                check_type = group["check_type"]
                algo_type = group["algorithm_type"]
                group["step"] += 1
                if algo_type == 'reduced':
                    step = group["step"]
                    learning_rate /= step**(1/3)
                for point in group["params"]:
                    if check_type:
                        _check_stiefel(point)
                    grad = point.grad
                    if grad is None:
                        continue
                    if grad.is_sparse:
                        raise RuntimeError(
                            "ConsLinSGD does not support sparse gradients"
                        )
                    state = self.state[point]

                    # State initialization
                    if len(state) == 0:
                        if momentum > 0:
                            state["momentum_buffer"] = grad.clone()

                    grad.add_(point, alpha=weight_decay)
                    if momentum > 0:
                        momentum_buffer = state["momentum_buffer"]
                        momentum_buffer.mul_(momentum).add_(
                            grad, alpha=1 - dampening
                        )
                        if nesterov:
                            grad = grad.add_(momentum_buffer, alpha=momentum)
                        else:
                            grad = momentum_buffer

                    if algo_type == 'full':
                        direction = _get_direction(point, -grad, alpha)
                    elif algo_type == 'reduced':
                        direction = _get_direction_reduced(point, -grad, alpha)
                    elif algo_type == 'geom':
                        direction = _get_direction_geom(point, -grad, alpha)
                    new_point = point + learning_rate * direction
                    #print(direction)
                    # use copy only for user facing point
                    point.copy_(new_point)

                if (
                    group["stabilize"] is not None
                    and group["step"] % group["stabilize"] == 0
                ):
                    self.stabilize_group(group)
        return loss

    @torch.no_grad()
    def stabilize_group(self, group):
        for p in group["params"]:
            if not isinstance(p, (ManifoldParameter, ManifoldTensor)):
                continue
            manifold = p.manifold
            momentum = group["momentum"]
            p.copy_(manifold.projx(p))
            if momentum > 0:
                param_state = self.state[p]
                if not param_state:  # due to None grads
                    continue
                if "momentum_buffer" in param_state:
                    buf = param_state["momentum_buffer"]
                    buf.copy_(manifold.proju(p, buf))