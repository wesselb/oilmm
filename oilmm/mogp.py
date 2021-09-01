from functools import wraps

import lab as B
from matrix import AbstractMatrix, TiledBlocks
from plum import Dispatcher
from plum import Union
from probmods import Model, instancemethod, cast, parse as _parse_transform
from stheno import Obs, cross, Measure, GP, Delta
from varz import minimise_l_bfgs_b

from .util import count, parse_input

__all__ = ["MOGP"]

_dispatch = Dispatcher()


@_dispatch
def _invert_kron(a: Union[AbstractMatrix, B.Numeric], m: B.Int, n: B.Int):
    """Invert the order of a Krocker product of square matrices.

    Args:
        a (matrix): Kronecker product of square matrices.
        m (int): Size of the left factor.
        n (int): Size of the right factor.

    Returns:
        matrix: Kronecker product the other way around.
    """
    a = B.reshape(a, m, n, m, n)
    a = B.transpose(a, perm=(1, 0, 3, 2))
    return B.reshape(a, n * m, n * m)


@_dispatch
def _resolve_noise(noise: TiledBlocks, n: B.Int, p: B.Int):
    """Resolve observation noise for multi-output observations.

    Args:
        noise (matrix): Observation noise.
        n (int): Number of observations.
        p (int): Number of outputs.

    Returns:
        matrix: Appropriate noise matrix, which is output-wise.
    """
    # `noise` is given observation-wise. Stheno expectes the noise matrix to
    # be output-wise, so we change the order.
    return _invert_kron(B.block_diag(noise), n, p)


@_dispatch
def _resolve_noise(noise: Union[AbstractMatrix, B.Numeric], p: B.Int, n: B.Int):
    return noise


@_dispatch
def _resolve_noise(noise: None, p: B.Int, n: B.Int):
    return None


class MOGP(Model):
    """A general multi-output GP.

    Args:
        processes (function): Function that returns a list of tuples of
            :class:`stheno.GP`s and noises, which correspond to the models for the
            outputs.
    """

    def __init__(self, processes):
        # If no measure is specified, set a default one to ensure that all the processes
        # live on the same measure.

        @wraps(processes)
        def processes_wrapped(*args, **kw_args):
            with Measure():
                return processes(*args, **kw_args)

        self.processes = processes_wrapped

    def __prior__(self):
        self.processes = self.processes(self.ps.processes)
        self.num_outputs = len(self.processes)

    def __noiseless__(self):
        self.processes = [(f, 0) for f, _ in self.processes]

    def _obs_y(self, fs, noises, x, y, noise):
        # Transform observations into a vector.
        y_flat = B.reshape(B.transpose(B.dense(y)), -1)  # Careful with the ordering!

        # Construct observations.
        ys = [
            f + GP(f_noise * Delta(), measure=f.measure)
            for f, f_noise in zip(fs, noises)
        ]
        obs = Obs(cross(*ys)(x, _resolve_noise(noise, *B.shape(y))), y_flat)

        return obs, y

    @cast
    def __condition__(self, x, y):
        x, noise = parse_input(x)
        fs, noises = zip(*self.processes)
        obs, _ = self._obs_y(fs, noises, x, y, noise)
        post = fs[0].measure | obs
        self.processes = [(post(f), noise) for f, noise in zip(fs, noises)]

    @instancemethod
    @cast
    def logpdf(self, x, y):
        x, noise = parse_input(x)
        fs, noises = zip(*self.processes)
        obs, y = self._obs_y(fs, noises, x, y, noise)
        return fs[0].measure.logpdf(obs)

    def _ys(self, x):
        return [
            f + GP(noise * Delta(), measure=f.measure) for f, noise in self.processes
        ]

    @instancemethod
    @cast
    def predict(self, x):
        # Predict the full joint.
        y = cross(*self._ys(x))
        pred = y(x)

        # Transform means into a matrix. Careful with the ordering!
        mean = B.transpose(B.reshape(pred.mean, self.num_outputs, -1))

        # Extract the right blocks from the predictive variance.
        var = pred.var
        # To allow contiguous extraction, we reverse the order of the Kronecker product.
        n, p = B.shape(mean)
        var = _invert_kron(var, p, n)
        var = B.stack(
            *[
                var[
                    i * self.num_outputs : (i + 1) * self.num_outputs,
                    i * self.num_outputs : (i + 1) * self.num_outputs,
                ]
                for i in range(B.shape(x, 0))
            ],
            axis=0
        )

        return mean, var

    @instancemethod
    @cast
    def sample(self, x):
        ys = self._ys(x)
        sample = B.concat(*ys[0].measure.sample(*[y(x) for y in ys]), axis=1)
        return sample
