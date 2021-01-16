import numpy as np
import pytest
from lab import B
from matrix import Dense
from stheno import EQ

from oilmm.ilmm import ILMMPP
from .util import approx


@pytest.fixture()
def construct_ilmm():
    # Setup model.
    kernels = [EQ(), 2 * EQ().stretch(1.5)]
    h = Dense(B.randn(3, 2))

    def construct_ilmm(noise_amplification=1):
        noise_obs = 0.1 * noise_amplification
        noises_latent = np.array([0.1, 0.2]) * noise_amplification
        return ILMMPP(kernels, h, noise_obs, noises_latent)

    return construct_ilmm


@pytest.fixture()
def x():
    return B.linspace(0, 3, 5)


def test_missing_data(construct_ilmm, x):
    ilmm = construct_ilmm()
    y = ilmm.sample(x, latent=False)

    # Throw away random data points and check that the logpdf computes.
    y2 = y.copy()
    y2[0, 0] = np.nan
    y2[2, 2] = np.nan
    y2[4, 1] = np.nan
    assert not np.isnan(ilmm.logpdf(x, y2))

    # Throw away an entire time point and check correctness.
    y2 = y.copy()
    y2[1, :] = np.nan
    approx(ilmm.logpdf(x[[0, 2, 3, 4]], y[[0, 2, 3, 4]]), ilmm.logpdf(x, y2))

    # Check LML after conditioning.
    ilmm = ilmm.condition(x, y2)
    approx(ilmm.logpdf(x[[0, 2, 3, 4]], y[[0, 2, 3, 4]]), ilmm.logpdf(x, y2))


def test_sample_noiseless(construct_ilmm, x):
    ilmm = construct_ilmm(noise_amplification=1000)
    sample = ilmm.sample(x, latent=True)

    # Test that sample has low variance.
    assert B.std(sample) < 3


def test_sample_noisy(construct_ilmm, x):
    ilmm = construct_ilmm(noise_amplification=1000)
    sample = ilmm.sample(x, latent=False)

    # Test that sample has high variance.
    assert B.std(sample) > 10


def test_predict_noiseless(construct_ilmm, x):
    ilmm = construct_ilmm(noise_amplification=1e-10)

    y = ilmm.sample(x)
    ilmm = ilmm.condition(x, y)
    means, lowers, uppers = ilmm.predict(x, latent=True)

    # Test that predictions match sample and have low uncertainty.
    approx(means, y, atol=1e-3)
    assert B.all(uppers - lowers < 1e-4)


def test_predict_noisy(construct_ilmm, x):
    ilmm = construct_ilmm(noise_amplification=1000)

    y = ilmm.sample(x)
    ilmm = ilmm.condition(x, y)
    means, lowers, uppers = ilmm.predict(x, latent=False)

    # Test that predictions have high uncertainty.
    assert B.all(uppers - lowers > 10)
