import imp
import pytest

import torch
import geoopt

from algo import ODCGM_SGD

torch.manual_seed(1)


@pytest.mark.parametrize("momentum", [0, 0.5])
@pytest.mark.parametrize("shape", [(3, 3), (4, 3, 3), (5, 4, 3, 3)])
def test_forward(shape, momentum):
    param = geoopt.ManifoldParameter(torch.randn(*shape), manifold=geoopt.Stiefel())
    optimizer = ODCGM_SGD((param,), lr=0.1, momentum=momentum)
    optimizer.zero_grad()
    loss = (param ** 2).sum()
    loss.backward()
    optimizer.step()


def test_convergence():
    p = 4
    param = geoopt.ManifoldParameter(torch.randn(p, p), manifold=geoopt.Stiefel())
    optimizer = ODCGM_SGD((param,), lr=0.1, alpha=1.)
    n_epochs = 100
    # Trace maximization: should end up in identity
    for _ in range(n_epochs):
        optimizer.zero_grad()
        loss = -torch.trace(param)
        loss.backward()
        optimizer.step()
    assert loss.item() + p < 1e-5
    orth_error = torch.norm(param.t().mm(param) - torch.eye(p))
    assert orth_error < 1e-5 