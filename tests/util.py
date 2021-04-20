import lab as B
from numpy.testing import assert_allclose
from plum import Dispatcher

__all__ = ["approx"]

_dispatch = Dispatcher()


@_dispatch
def approx(x, y, **kw_args):
    approx(B.to_numpy(x), B.to_numpy(y), **kw_args)


@_dispatch
def approx(x: B.Numeric, y: B.Numeric, **kw_args):
    assert_allclose(x, y, **kw_args)