import lab as B
import pytest
from oilmm.test import test_sample_prior

# noinspection PyUnresolvedReferences
from .util import approx, oilmm


def different(x, y, rtol=None, atol=None):
    if rtol is None and atol is None:
        rtol = 1e-2
    adiff = B.abs(B.to_numpy(x) - B.to_numpy(y))
    rdiff = adiff / B.maximum(B.to_numpy(x), B.to_numpy(y))
    if atol is not None:
        assert B.mean(adiff) >= atol
    if rtol is not None:
        assert B.mean(rdiff) >= rtol


@pytest.fixture()
def x():
    return B.randn(50, 1)


def test_sample(oilmm, x):
    B.set_random_seed(0)
    sample1 = oilmm.sample(x)
    sample2 = oilmm.sample(x)
    different(sample1, sample2)
