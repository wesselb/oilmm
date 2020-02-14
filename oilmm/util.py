import lab as B
from matrix import AbstractMatrix, Kronecker
import numpy as np

__all__ = ['normalise']


def normalise(y):
    """Normalise data and create an unnormaliser.

    Args:
        y (matrix): Data to normalise.

    Returns:
        tuple: Tuple containing the normalised data and a function that
            undoes the normalisation.
    """
    scale = np.nanstd(y, axis=0, keepdims=True)
    mean = np.nanmean(y, axis=0, keepdims=True)

    def unnormalise(*ys_norm):
        ys_unnorm = ()
        for y in ys_norm:
            ys_unnorm += (y * scale + mean,)
        return ys_unnorm[0] if len(ys_unnorm) == 1 else ys_unnorm

    return (y - mean) / scale, unnormalise


@B.dispatch(AbstractMatrix)
def pd_inv(a):
    """Compute the inverse of a positive-definite matrix.

    Args:
        a (matrix): Matrix to compute inverse of.

    Returns:
        matrix: Inverse of `a`.
    """
    return B.cholsolve(B.chol(a), B.eye(a))


@B.dispatch(Kronecker)
def pd_inv(a):
    return Kronecker(B.pd_inv(a.left), B.pd_inv(a.right))


B.pd_inv = pd_inv


@B.dispatch(AbstractMatrix)
def pinv(a):
    """Compute the left pseudo-inverse.

    Args:
        a (matrix): Matrix to compute left pseudo-inverse of.

    Returns:
        matrix: Left pseudo-inverse of `a`.
    """
    return B.cholsolve(B.chol(B.matmul(a, a, tr_a=True)), B.transpose(a))


@B.dispatch(Kronecker)
def pinv(a):
    return Kronecker(B.pinv(a.left), B.pinv(a.right))


B.pinv = pinv
