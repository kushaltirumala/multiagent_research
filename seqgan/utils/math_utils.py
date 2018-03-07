import numpy as np


def normal_entropy(std):
    var = std.pow(2)
    entropy = 0.5 + 0.5 * np.log(2 * var * math.pi)
    return entropy.sum(-1, keepdim=True)


def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * var) - 0.5 * np.log(2 * math.pi) - log_std
    return log_density.sum(-1, keepdim=True)
