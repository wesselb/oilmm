import numpy as np
import pytest
from lab import B
from stheno import EQ

from oilmm.ilmm import _to_tuples, ILMMPP
from .util import allclose, approx


@pytest.fixture()
def construct_ilmm():
    # Setup model.
    kernels = [EQ(), 2 * EQ().stretch(1.5)]
    h = B.randn(3, 2)

    def construct_ilmm(noise=True):
        if noise:
            noise_obs = .1
            noises_latent = np.array([.1, .2])
        else:
            noise_obs = 1e-8
            noises_latent = B.zeros(2)

        return ILMMPP(kernels, h, noise_obs, noises_latent)

    return construct_ilmm


@pytest.fixture()
def x():
    return B.linspace(0, 3, 5)


def test_to_tuples():
    x = B.linspace(0, 1, 3)[:, None]
    y = B.randn(3, 2)
    y[0, 0] = np.nan
    y[1, 1] = np.nan

    # Check correctness.
    (x1, i1, y1), (x2, i2, y2) = _to_tuples(x, y)
    allclose(x1, x[[1, 2]])
    assert i1 == 0
    allclose(y1, y[[1, 2], 0])
    allclose(x2, x[[0, 2]])
    assert i2 == 1
    allclose(y2, y[[0, 2], 1])

    # Test check that any data is extracted.
    y_nan = y.copy()
    y_nan[:] = np.nan
    with pytest.raises(ValueError):
        _to_tuples(x, y_nan)


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
    allclose(ilmm.logpdf(x[[0, 2, 3, 4]], y[[0, 2, 3, 4]]),
             ilmm.logpdf(x, y2))

    # Check LML after conditioning.
    ilmm = ilmm.condition(x, y2)
    allclose(ilmm.logpdf(x[[0, 2, 3, 4]], y[[0, 2, 3, 4]]),
             ilmm.logpdf(x, y2))


class TrackedIterator:
    """An iterator that keeps track of whether it has been used.

    Attributes:
        used (bool): Boolean indicating whether it has been usd.

    Args:
        wrap (object): Object to wrap.
    """
    instances = []

    def __init__(self, wrap):
        self.used = False
        self.wrap = wrap
        TrackedIterator.instances.append(self)

    def __iter__(self):
        self.used = True
        return iter(self.wrap)

    @staticmethod
    def reset():
        """Reset all."""
        for instance in TrackedIterator.instances:
            instance.used = False


def test_sample(construct_ilmm):
    ilmm = construct_ilmm()

    ilmm.fs = TrackedIterator(ilmm.fs)
    ilmm.ys = TrackedIterator(ilmm.ys)

    # Test latent samples.
    x = B.randn(10)
    assert isinstance(ilmm.sample(x, latent=True), B.NPNumeric)
    assert ilmm.fs.used
    assert not ilmm.ys.used

    TrackedIterator.reset()

    # Test observed samples.
    assert isinstance(ilmm.sample(x, latent=False), B.NPNumeric)
    assert not ilmm.fs.used
    assert ilmm.ys.used


def test_predict(construct_ilmm, x):
    ilmm = construct_ilmm(noise=False)

    y = ilmm.sample(x)
    ilmm = ilmm.condition(x, y)
    means, lowers, uppers = ilmm.predict(x)

    # Test that predictions match sample and have low uncertainty.
    approx(means, y, decimal=3)
    assert B.all(uppers - lowers < 1e-4)
