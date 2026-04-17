import scipy.io


def get_dataset(time_range=[0.0, 1.0]):
    # Load data from the file
    data = scipy.io.loadmat("data/ginzburg_landau_square.mat")

    u_ref = data["usol"]
    v_ref = data["vsol"]

    # PDE parameters
    eps = data["eps"].flatten()[0]
    k = data["k"].flatten()[0]

    t_star = data["t"].flatten()
    x_star = data["x"].flatten()
    y_star = data["y"].flatten()

    start_time_step = int(time_range[0] * len(t_star))
    end_time_step = int(time_range[1] * len(t_star))

    u_ref = u_ref[start_time_step:end_time_step, :, :]
    v_ref = v_ref[start_time_step:end_time_step, :, :]
    t_star = t_star[start_time_step: end_time_step] - t_star[start_time_step]

    # Return the processed data
    return u_ref, v_ref, t_star, x_star, y_star, eps, k
