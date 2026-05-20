import numpy as np

from AI_stats_lab import (
    joint_gaussian_pdf,
    marginal_pdf_x,
    marginal_pdf_y,
    covariance_matrix,
    joint_pdf_grid_integral,
    generate_joint_gaussian_samples,
    sample_means,
    sample_covariance_matrix,
    sample_correlation,
    gaussian_independence_check,
    zero_rho_covariance_check,
    nonzero_rho_covariance_check,
)


# -------------------------------------------------
# Question 1 Tests
# -------------------------------------------------

def test_joint_gaussian_pdf_center():

    val = joint_gaussian_pdf(1, -2)

    expected = (
        1 /
        (2 * np.pi * 2 * 3 * np.sqrt(1 - 0.6**2))
    )

    assert np.isclose(val, expected, atol=1e-6)


def test_joint_gaussian_pdf_off_center():

    val = joint_gaussian_pdf(3, 1)

    sigma_x = 2
    sigma_y = 3
    rho = 0.6
    mu_x = 1
    mu_y = -2

    q = (
        ((3 - mu_x) ** 2) / sigma_x**2
        - 2 * rho * ((3 - mu_x) * (1 - mu_y)) / (sigma_x * sigma_y)
        + ((1 - mu_y) ** 2) / sigma_y**2
    )

    expected = (
        1 /
        (2 * np.pi * sigma_x * sigma_y * np.sqrt(1 - rho**2))
    ) * np.exp(-q / (2 * (1 - rho**2)))

    assert np.isclose(val, expected, atol=1e-6)


def test_marginal_pdfs():

    fx = marginal_pdf_x(1)
    fy = marginal_pdf_y(-2)

    expected_fx = 1 / (np.sqrt(2 * np.pi) * 2)
    expected_fy = 1 / (np.sqrt(2 * np.pi) * 3)

    assert np.isclose(fx, expected_fx, atol=1e-6)
    assert np.isclose(fy, expected_fy, atol=1e-6)


def test_covariance_matrix():

    cm = covariance_matrix()

    expected = np.array([
        [4.0, 3.6],
        [3.6, 9.0]
    ])

    assert cm.shape == (2, 2)
    assert np.allclose(cm, expected)


def test_joint_pdf_grid_integral():

    val = joint_pdf_grid_integral(n=200)

    assert 0.98 < val < 1.02


# -------------------------------------------------
# Question 2 Tests
# -------------------------------------------------

def test_generate_joint_gaussian_samples_shape():

    x, y = generate_joint_gaussian_samples(n=5000)

    assert len(x) == 5000
    assert len(y) == 5000


def test_sample_means():

    x, y = generate_joint_gaussian_samples(
        n=100000,
        seed=42
    )

    mx, my = sample_means(x, y)

    assert abs(mx - 1) < 0.05
    assert abs(my + 2) < 0.05


def test_sample_covariance_matrix():

    x, y = generate_joint_gaussian_samples(
        n=100000,
        seed=42
    )

    cm = sample_covariance_matrix(x, y)

    assert cm.shape == (2, 2)

    assert abs(cm[0, 0] - 4) < 0.1
    assert abs(cm[1, 1] - 9) < 0.15
    assert abs(cm[0, 1] - 3.6) < 0.15


def test_sample_correlation():

    x, y = generate_joint_gaussian_samples(
        n=100000,
        seed=42
    )

    r = sample_correlation(x, y)

    assert abs(r - 0.6) < 0.03


def test_gaussian_independence_check():

    assert gaussian_independence_check(0) is True
    assert gaussian_independence_check(0.6) is False


def test_zero_rho_covariance_check():

    assert zero_rho_covariance_check() is True


def test_nonzero_rho_covariance_check():

    assert nonzero_rho_covariance_check() is True
