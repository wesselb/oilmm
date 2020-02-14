import lab as B
from numpy.testing import assert_allclose, assert_array_almost_equal
from plum import Dispatcher

__all__ = ['allclose', 'approx']

_dispatch = Dispatcher()

approx = assert_array_almost_equal


@_dispatch(object, object)
def allclose(x, y):
    allclose(B.to_numpy(x), B.to_numpy(y))


@_dispatch(B.Numeric, B.Numeric)
def allclose(x, y):
    assert_allclose(x, y)
