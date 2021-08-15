import lab as B
from plum import Dispatcher

__all__ = ["count", "parse_input"]

_dispatch = Dispatcher()


@_dispatch
def parse_input(x):
    """Get the noise component of an input, if a noise component is specified.

    Args:
        x (input): Input.

    Returns:
        tuple: Input and noise.
    """
    return x, None


@_dispatch
def parse_input(x_and_noise: tuple):
    x, noise = x_and_noise
    return x, noise


def count(a):
    """Count the number of non-NaN values in a tensor.

    Args:
        a (tensor): Tensor to count.

    Returns:
        scalar: Number of non-NaN values.
    """
    return B.sum(B.cast(B.dtype(a), ~B.isnan(a)))
