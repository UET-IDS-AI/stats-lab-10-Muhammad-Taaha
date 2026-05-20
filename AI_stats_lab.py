import numpy as np


# -------------------------------------------------
# Question 1: Joint Gaussian PDF and Marginals
# -------------------------------------------------

def joint_gaussian_pdf(x, y, mu_x=1, mu_y=-2, sigma_x=2, sigma_y=3, rho=0.6):
    """
    Return the bivariate Gaussian PDF f_XY(x,y).

    Use the formula:

    f_XY(x,y) =
    1 / (2*pi*sigma_x*sigma_y*sqrt(1-rho^2))
    *
    exp( -Q / (2*(1-rho^2)) )
    """
    pass


def marginal_pdf_x(x, mu_x=1, sigma_x=2):
    """
    Return marginal Gaussian PDF of X.
    """
    pass


def marginal_pdf_y(y, mu_y=-2, sigma_y=3):
    """
    Return marginal Gaussian PDF of Y.
    """
    pass


def covariance_matrix(sigma_x=2, sigma_y=3, rho=0.6):
    """
    Return covariance matrix:

    [[sigma_x^2, rho*sigma_x*sigma_y],
     [rho*sigma_x*sigma_y, sigma_y^2]]
    """
    pass


def joint_pdf_grid_integral(mu_x=1, mu_y=-2, sigma_x=2, sigma_y=3, rho=0.6, n=250):
    """
    Numerically approximate integral of joint Gaussian PDF
    over the rectangle:

    [mu_x - 4*sigma_x, mu_x + 4*sigma_x]
    x
    [mu_y - 4*sigma_y, mu_y + 4*sigma_y]

    Use a rectangular grid or trapezoidal numerical integration.
    """
    pass


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
    """
    Generate n samples from a jointly Gaussian distribution.

    Return two arrays:
    x_samples, y_samples

    Hint:
    Use np.random.multivariate_normal.
    """
    pass


def sample_means(x_samples, y_samples):
    """
    Return sample means of X and Y.
    """
    pass


def sample_covariance_matrix(x_samples, y_samples):
    """
    Return 2 by 2 sample covariance matrix.

    Use denominator n-1.
    """
    pass


def sample_correlation(x_samples, y_samples):
    """
    Return sample correlation coefficient.
    """
    pass


def gaussian_independence_check(rho):
    """
    For jointly Gaussian variables:
    return True if rho is zero, otherwise False.
    """
    pass


def zero_rho_covariance_check(n=100000):
    """
    Generate samples with rho=0 and check that
    sample covariance is approximately zero.
    Return True or False.
    """
    pass


def nonzero_rho_covariance_check(n=100000):
    """
    Generate samples with rho=0.6 and check that
    sample covariance is close to rho*sigma_x*sigma_y.
    Return True or False.
    """
    pass
