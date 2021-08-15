import lab as B
import numpy as np
import pytest
from numpy.testing import assert_allclose
from plum import Dispatcher
from stheno import GP, EQ

from oilmm import OILMM

__all__ = ["increased_regularisation", "oilmm", "approx"]

_dispatch = Dispatcher()


@pytest.fixture
def increased_regularisation():
    old_epsilon = B.epsilon
    B.epsilon = 1e-10
    yield
    B.epsilon = old_epsilon


@pytest.fixture()
def oilmm():
    return OILMM(
        np.float64,
        lambda _: [
            (1.0 * GP(EQ().stretch(0.8)), 1e-2),
            (1.1 * GP(EQ().stretch(0.7)), 2e-2),
            (1.2 * GP(EQ().stretch(0.6)), 3e-2),
        ],
        num_outputs=6,
    )


def approx(x, y, **kw_args):
    assert_allclose(B.to_numpy(x), B.to_numpy(y), **kw_args)
