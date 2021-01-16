import lab as B
import numpy as np
import pytest
from matrix import Dense, Diagonal
from stheno import EQ

from oilmm import OILMM, ILMMPP
from .util import approx


@pytest.fixture()
def construct_oilmm():
    # Setup model.
    kernels = [EQ(), 2 * EQ().stretch(1.5)]
    u, s_sqrt = B.svd(B.randn(3, 2))[:2]
    u = Dense(u)
    s_sqrt = Diagonal(s_sqrt)

    def construct_iolmm(noise_amplification=1):
        noise_obs = 0.1 * noise_amplification
        noises_latent = np.array([0.1, 0.2]) * noise_amplification
        return OILMM(kernels, u, s_sqrt, noise_obs, noises_latent)

    return construct_iolmm


@pytest.fixture()
def x():
    return B.linspace(0, 3, 5)


def test_compare_ilmm():
    # Setup models.
    kernels = [EQ(), 2 * EQ().stretch(1.5)]
    noise_obs = 0.1
    noises_latent = np.array([0.1, 0.2])
    u, s_sqrt = B.svd(B.randn(3, 2))[:2]
    u = Dense(u)
    s_sqrt = Diagonal(s_sqrt)

    # Construct models.
    ilmm = ILMMPP(kernels, u @ s_sqrt, noise_obs, noises_latent)
    oilmm = OILMM(kernels, u, s_sqrt, noise_obs, noises_latent)

    # Construct data.
    x = B.linspace(0, 3, 5)
    y = ilmm.sample(x, latent=False)
    x2 = B.linspace(4, 7, 5)
    y2 = ilmm.sample(x2, latent=False)

    # Check LML before conditioning.
    approx(ilmm.logpdf(x, y), oilmm.logpdf(x, y))
    approx(ilmm.logpdf(x2, y2), oilmm.logpdf(x2, y2))

    ilmm = ilmm.condition(x, y)
    oilmm = oilmm.condition(x, y)

    # Check LML after conditioning.
    approx(ilmm.logpdf(x, y), oilmm.logpdf(x, y))
    approx(ilmm.logpdf(x2, y2), oilmm.logpdf(x2, y2))

    # Predict.
    means_pp, lowers_pp, uppers_pp = ilmm.predict(x2)
    means, lowers, uppers = oilmm.predict(x2)

    # Check predictions.
    approx(means_pp, means)
    approx(lowers_pp, lowers)
    approx(uppers_pp, uppers)


def test_logpdf_missing_data():
    # Setup model.
    m = 3
    noise = 1e-2
    latent_noises = 2e-2 * B.ones(m)
    kernels = [0.5 * EQ().stretch(0.75) for _ in range(m)]
    x = B.linspace(0, 10, 20)

    # Concatenate two orthogonal matrices, to make the missing data
    # approximation exact.
    u1 = B.svd(B.randn(m, m))[0]
    u2 = B.svd(B.randn(m, m))[0]
    u = Dense(B.concat(u1, u2, axis=0) / B.sqrt(2))

    s_sqrt = Diagonal(B.rand(m))

    # Construct a reference model.
    oilmm_pp = ILMMPP(kernels, u @ s_sqrt, noise, latent_noises)

    # Sample to generate test data.
    y = oilmm_pp.sample(x, latent=False)

    # Throw away data, but retain orthogonality.
    y[5:10, 3:] = np.nan
    y[10:, :3] = np.nan

    # Construct OILMM to test.
    oilmm = OILMM(kernels, u, s_sqrt, noise, latent_noises)

    # Check that evidence is still exact.
    approx(oilmm_pp.logpdf(x, y), oilmm.logpdf(x, y), atol=1e-7)


def test_sample_noiseless(construct_oilmm, x):
    oilmm = construct_oilmm(noise_amplification=1000)
    sample = oilmm.sample(x, latent=True)

    # Test that sample has low variance.
    assert B.std(sample) < 3


def test_sample_noisy(construct_oilmm, x):
    oilmm = construct_oilmm(noise_amplification=1000)
    sample = oilmm.sample(x, latent=False)

    # Test that sample has high variance.
    assert B.std(sample) > 10


def test_predict_noiseless(construct_oilmm, x):
    oilmm = construct_oilmm(noise_amplification=1e-10)

    y = oilmm.sample(x)
    oilmm = oilmm.condition(x, y)
    means, lowers, uppers = oilmm.predict(x, latent=True)

    # Test that predictions match sample and have low uncertainty.
    approx(means, y, atol=1e-3)
    assert B.all(uppers - lowers < 1e-4)

    # Test that variances can be returned.
    means, variances = oilmm.predict(x, latent=True, return_variances=True)
    approx((uppers - lowers) / 3.92, variances ** 0.5)


def test_predict_noisy(construct_oilmm, x):
    oilmm = construct_oilmm(noise_amplification=1000)

    y = oilmm.sample(x)
    oilmm = oilmm.condition(x, y)
    means, lowers, uppers = oilmm.predict(x, latent=False)

    # Test that predictions have high uncertainty.
    assert B.all(uppers - lowers > 10)

    # Test that variances can be returned.
    means, variances = oilmm.predict(x, latent=False, return_variances=True)
    approx((uppers - lowers) / 3.92, variances ** 0.5)
