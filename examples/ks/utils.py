import scipy.io


def get_dataset(time_range=[0.0, 1.0]):
    # Load data
    data = scipy.io.loadmat("data/ks_chaotic.mat")
    u_ref = data["usol"]
    t_star = data["t"].flatten()
    x_star = data["x"].flatten()

    # Only use a fraction of the data
    start_time_step = int(time_range[0] * len(t_star))
    end_time_step = int(time_range[1] * len(t_star))

    u_ref = u_ref[start_time_step:end_time_step]
    t_star = t_star[start_time_step: end_time_step] - t_star[start_time_step]

    return u_ref, t_star, x_star
