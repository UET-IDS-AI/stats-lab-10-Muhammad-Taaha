import numpy as np


# -------------------------------------------------
# Question 1: Joint Gaussian PDF and Marginals
# -------------------------------------------------

def joint_gaussian_pdf(x, y, mu_x=1, mu_y=-2, sigma_x=2, sigma_y=3, rho=0.6):

    coeff = 1 / (2 * np.pi * sigma_x * sigma_y * np.sqrt(1 - rho**2))

    q = (
        ((x - mu_x) ** 2) / (sigma_x ** 2)
        - 2 * rho * ((x - mu_x) * (y - mu_y)) / (sigma_x * sigma_y)
        + ((y - mu_y) ** 2) / (sigma_y ** 2)
    )

    return coeff * np.exp(-q / (2 * (1 - rho**2)))


def marginal_pdf_x(x, mu_x=1, sigma_x=2):

    return (1 / (np.sqrt(2 * np.pi) * sigma_x)) * np.exp(
        -((x - mu_x) ** 2) / (2 * sigma_x ** 2)
    )


def marginal_pdf_y(y, mu_y=-2, sigma_y=3):

    return (1 / (np.sqrt(2 * np.pi) * sigma_y)) * np.exp(
        -((y - mu_y) ** 2) / (2 * sigma_y ** 2)
    )


def covariance_matrix(sigma_x=2, sigma_y=3, rho=0.6):

    return np.array([
        [sigma_x**2, rho * sigma_x * sigma_y],
        [rho * sigma_x * sigma_y, sigma_y**2]
    ])


def joint_pdf_grid_integral(mu_x=1, mu_y=-2, sigma_x=2, sigma_y=3, rho=0.6, n=250):

    x_vals = np.linspace(mu_x - 4 * sigma_x, mu_x + 4 * sigma_x, n)
    y_vals = np.linspace(mu_y - 4 * sigma_y, mu_y + 4 * sigma_y, n)

    dx = x_vals[1] - x_vals[0]
    dy = y_vals[1] - y_vals[0]

    total = 0.0

    for x in x_vals:
        for y in y_vals:
            total += joint_gaussian_pdf(x, y, mu_x, mu_y, sigma_x, sigma_y, rho)

    return total * dx * dy


# -------------------------------------------------
# Question 2: Simulation and Independence
# -------------------------------------------------

def generate_joint_gaussian_samples(
    n=100000,
    mu_x=1,
    mu_y=-2,
    sigma_x=2,
    sigma_y=3,
    rho=0.6,
    seed=0
):

    np.random.seed(seed)

    mean = [mu_x, mu_y]
    cov = covariance_matrix(sigma_x, sigma_y, rho)

    samples = np.random.multivariate_normal(mean, cov, size=n)

    return samples[:, 0], samples[:, 1]


def sample_means(x_samples, y_samples):

    return np.mean(x_samples), np.mean(y_samples)


def sample_covariance_matrix(x_samples, y_samples):

    n = len(x_samples)

    mean_x = np.mean(x_samples)
    mean_y = np.mean(y_samples)

    cov_xx = np.sum((x_samples - mean_x) ** 2) / (n - 1)
    cov_yy = np.sum((y_samples - mean_y) ** 2) / (n - 1)
    cov_xy = np.sum((x_samples - mean_x) * (y_samples - mean_y)) / (n - 1)

    return np.array([
        [cov_xx, cov_xy],
        [cov_xy, cov_yy]
    ])


def sample_correlation(x_samples, y_samples):

    cov_matrix = sample_covariance_matrix(x_samples, y_samples)

    std_x = np.sqrt(cov_matrix[0, 0])
    std_y = np.sqrt(cov_matrix[1, 1])

    return cov_matrix[0, 1] / (std_x * std_y)


def gaussian_independence_check(rho):

    return rho == 0


def zero_rho_covariance_check(n=100000):

    x, y = generate_joint_gaussian_samples(n=n, rho=0)

    cov_matrix = sample_covariance_matrix(x, y)

    return bool(abs(cov_matrix[0, 1]) < 0.05)


def nonzero_rho_covariance_check(n=100000):

    sigma_x = 2
    sigma_y = 3
    rho = 0.6

    x, y = generate_joint_gaussian_samples(
        n=n,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        rho=rho
    )

    cov_matrix = sample_covariance_matrix(x, y)

    expected = rho * sigma_x * sigma_y

    return bool(abs(cov_matrix[0, 1] - expected) < 0.15)