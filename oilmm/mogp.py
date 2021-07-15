from functools import wraps

import lab as B
from matrix import AbstractMatrix, TiledBlocks
from plum import Dispatcher
from plum import Union
from probmods.bijection import parse as _parse_transform
from probmods.model import MultiOutputModel
from stheno import Obs, cross, Measure, GP, Delta
from varz import minimise_l_bfgs_b

from .util import count

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


class MOGP(MultiOutputModel):
    """A general multi-output GP.

    Args:
        dtype (dtype): Data type.
        processes (function): Function that returns a list of tuples of
            :class:`stheno.GP`s and noises, which correspond to the models for the
            outputs.
        data_transform (str or :class:`.bijection.Bijection`, optional): Specification
            for a data transformation. Defaults to no transform.
    """

    def __init__(self, dtype, processes, data_transform=None):
        MultiOutputModel.__init__(self, dtype)

        # If no measure is specified, set a default one to ensure that all the processes
        # live on the same measure.

        @wraps(processes)
        def processes_wrapped(*args, **kw_args):
            with Measure():
                return processes(*args, **kw_args)

        self.processes = processes_wrapped
        self.data_transform = _parse_transform(data_transform)
        self.num_outputs = len(self.processes(self.params.processes))

    @property
    def noiseless(self):

        def build_processes(params):
            return [(f, 0) for f, _ in self.processes(params)]

        return MOGP(
            dtype=self.dtype,
            processes=build_processes,
            data_transform=self.data_transform,
        ).bind(self)

    def _obs_y(self, fs, noises, x, y, noise):
        y = self.data_transform.transform(y)

        # Transform observations into a vector.
        y_flat = B.reshape(B.transpose(B.dense(y)), -1)  # Careful with the ordering!

        # Construct observations.
        ys = [
            f + GP(f_noise * Delta(), measure=f.measure)
            for f, f_noise in zip(fs, noises)
        ]
        obs = Obs(cross(*ys)(x, _resolve_noise(noise, *B.shape(y))), y_flat)

        return obs, y

    def logpdf(self, x, y, noise=None):
        fs, noises = zip(*self.processes(self.params.processes))
        obs, y = self._obs_y(fs, noises, x, y, noise)
        return fs[0].measure.logpdf(obs) + self.data_transform.logdet(y)

    def condition(self, x, y, noise=None):
        def construct_processes(_):
            fs, noises = zip(*self.processes(self.params.processes))
            obs, _ = self._obs_y(fs, noises, x, y, noise)
            post = fs[0].measure | obs
            return [(post(f), noise) for f, noise in zip(fs, noises)]

        return MOGP(
            dtype=self.dtype,
            processes=construct_processes,
            data_transform=self.data_transform,
        ).bind(self)

    def fit(self, x, y, noise=None, **kw_args):
        def negative_log_marginal_likelihood(vs):
            with self.use_vs(vs):
                return -self.logpdf(x, y, noise) / count(y)

        minimise_l_bfgs_b(negative_log_marginal_likelihood, self.vs, **kw_args)

    def _ys(self, x):
        return [
            f + GP(noise * Delta(), measure=f.measure)
            for f, noise in self.processes(self.params.processes)
        ]

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

        return self.data_transform.untransform((mean, var))

    def sample(self, x):
        ys = self._ys(x)
        sample = B.concat(*ys[0].measure.sample(*[y(x) for y in ys]), axis=1)
        return self.data_transform.untransform(sample)
