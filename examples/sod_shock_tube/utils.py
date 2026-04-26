import numpy as np


def get_dataset():
    data = dict(np.load("data/sod_shock_tube.npz"))
    rho_ref = data["rho"]
    u_ref = data["u"]
    p_ref = data["p"]
    T = data["T"]
    X = data["X"]
    t_star = T[0]
    x_star = X[:, 0]
    left_coords = np.stack((t_star, X[0]), axis=-1)
    right_coords = np.stack((t_star, X[-1]), axis=-1)
    return rho_ref, u_ref, p_ref, T, X, t_star, x_star, left_coords, right_coords
