from .model import RandPointCNN
from .util_funcs import knn_indices_func_gpu
from .util_layers import Dense

# C_in, C_out, D, N_neighbors, dilution, N_rep, r_indices_func, C_lifted = None, mlp_width = 2
# (a, b, c, d, e) == (C_in, C_out, N_neighbors, dilution, N_rep)
# Abbreviated PointCNN constructor.
AbbPointCNN = lambda a, b, c, d, e: RandPointCNN(a, b, 3, c, d, e, knn_indices_func_gpu)

__all__ = [
    'RandPointCNN',
    'AbbPointCNN',
    'Dense',
]
