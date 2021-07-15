import lab as B

__all__ = ["count"]


def count(a):
    """Count the number of non-NaN values in a tensor.

    Args:
        a (tensor): Tensor to count.

    Returns:
        scalar: Number of non-NaN values.
    """
    return B.sum(B.cast(B.dtype(a), ~B.isnan(a)))
