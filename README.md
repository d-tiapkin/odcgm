# Orthogonal Directions Constrained Gradient Method: from non-linear equality constraints to Stiefel manifold

The official implementation for the paper ["Orthogonal Directions Constrained Gradient Method: from non-linear equality constraints to Stiefel manifold"](https://proceedings.mlr.press/v195/schechtman23a.html). 

The presented algorithm generalized the Landing algorithm (see https://github.com/pierreablin/landing) from the manifold of orthogonal matrices to Stiefel manifold. The implementation is inpired by the code of Landing algorithm.

## Use

The main algorithmic part is presented in the file ``algo.py``. There is an pytorch optimizer ``ODCGM_SGD`` that mimics geoopt's ``RiemannianSGD``.

