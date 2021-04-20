import lab as B
import numpy as np
from matrix import AbstractMatrix, Kronecker

__all__ = ["Normaliser"]


def _to_multiarg(f):
    def f_multiarg(self, *xs):
        ys = ()
        for x in xs:
            ys += (f(self, x),)
        return ys[0] if len(ys) == 1 else ys

    return f_multiarg


class Normaliser:
    """Create a data normaliser.

    Args:
        y (matrix): Data to base normalisation on.
    """

    def __init__(self, y):
        self.mean = np.nanmean(y, axis=0, keepdims=True)
        self.scale = np.nanstd(y, axis=0, keepdims=True)

    @_to_multiarg
    def normalise(self, y):
        """Perform normalisation.

        Accepts multiple arguments.

        Args:
            y (matrix): Data to normalise.

        Returns:
            matrix: Normalised data.
        """
        return (y - self.mean) / self.scale

    @_to_multiarg
    def unnormalise(self, y):
        """Undo normalisation.

        Accepts multiple arguments.

        Args:
            y (matrix): Data to unnormalise.

        Returns:
            matrix: Unnormalised data.
        """
        return y * self.scale + self.mean

    @_to_multiarg
    def unnormalise_variance(self, y):
        """Undo normalisation for a variance.

        Accepts multiple arguments.

        Args:
            y (matrix): Variances to unnormalise.

        Returns:
            matrix: Unnormalised variance.
        """
        return y * self.scale ** 2

    @_to_multiarg
    def normalise_logdet(self, y):
        """Compute the log-determinant of the Jacobian of the normalisation.

        Accepts multiple arguments.

        Args:
            y (matrix): Data that was transformed.

        Returns:
            scalar: Log-determinant.
        """
        return -B.shape(y)[0] * B.sum(B.log(self.scale))


@B.dispatch
def pinv(a: AbstractMatrix):
    """Compute the left pseudo-inverse.

    Args:
        a (matrix): Matrix to compute left pseudo-inverse of.

    Returns:
        matrix: Left pseudo-inverse of `a`.
    """
    return B.cholsolve(B.chol(B.matmul(a, a, tr_a=True)), B.transpose(a))


@B.dispatch
def pinv(a: Kronecker):
    return Kronecker(B.pinv(a.left), B.pinv(a.right))


B.pinv = pinv
