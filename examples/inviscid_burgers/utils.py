import scipy.io
import numpy as np


def get_dataset():
    # Exact soluiton to Riemann problem
    burger_data = np.load('data/inviscid_burger.npy', allow_pickle=True).item()

    u_ref = burger_data['u']
    t_star = burger_data['t']
    x_star = burger_data['x']

    return u_ref, t_star, x_star
