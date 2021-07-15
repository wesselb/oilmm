import lab as B
from matrix import TiledBlocks
from plum import Dispatcher
from probmods.bijection import parse as _parse_transform
from probmods.model import MultiOutputModel
from stheno import Obs, PseudoObs
from varz import minimise_l_bfgs_b

from .util import count

__all__ = ["IMOGP"]

_dispatch = Dispatcher()


@_dispatch
def _noise_diagonals_to_matrix(noise: TiledBlocks):
    """Take the diagonals of the observation noise at every time and concatenate those
    diagonals into a matrix.

    Args:
        noise (matrix): Noise matrices.

    Returns:
        matrix: Concatenation of the diagonals of `noise`.
    """
    if noise.axis != 0:
        raise AssertionError(f"Axis of tiled noise blocks must be 0.")
    blocks = []
    for block, rep in zip(noise.blocks, noise.reps):
        blocks.append(B.broadcast_to(B.diag(block)[None, :], rep, B.shape(block, 1)))
    return B.concat(*blocks, axis=0)


@_dispatch
def _noise_diagonals_to_matrix(noise: None):
    class _Zero:
        def __getitem__(self, item):
            return 0

    return _Zero()


class IMOGP(MultiOutputModel):
    """A multi-output GP consisting of independent GPs.

    Args:
        dtype (dtype): Data type.
        processes (function): Function that returns a list of tuples of
            :class:`stheno.GP`s and noises, which correspond to the models for the
            outputs.
        data_transform (str or :class:`.bijection.Bijection`, optional): Specification
            for a data transformation. Defaults to no transform.
        x_ind (tensor, optional): Initialisation for inducing inputs.
    """

    def __init__(self, dtype, processes, data_transform=None, x_ind=None):
        MultiOutputModel.__init__(self, dtype)
        self.processes = processes
        self.data_transform = _parse_transform(data_transform)
        self._x_ind_init = x_ind
        self.num_outputs = len(self.processes(self.params.processes))

    @property
    def x_ind(self):
        """tensor or None: Inputs of inducing points."""
        if self._x_ind_init is None:
            return None
        else:
            return self.params.x_ind.unbounded(self._x_ind_init)

    @property
    def noiseless(self):

        def build_processes(params):
            return [(f, 0) for f, _ in self.processes(params)]

        return IMOGP(
            dtype=self.dtype,
            processes=build_processes,
            data_transform=self.data_transform,
            x_ind=self.x_ind,
        ).bind(self)

    def _compute_obs(self, f, x, y, noise):
        if self.x_ind is None:
            return Obs(f(x, noise), y)
        else:
            return PseudoObs(f(self.x_ind), f(x, noise), y)

    def logpdf(self, x, y, noise=None):
        noise = _noise_diagonals_to_matrix(noise)
        y = self.data_transform.transform(y)

        logpdf = 0
        for i, (f, f_noise) in enumerate(self.processes(self.params.processes)):
            obs = self._compute_obs(f, x, y[:, i], noise[:, i] + f_noise)
            logpdf = logpdf + f.measure.logpdf(obs)
        return logpdf + self.data_transform.logdet(y)

    def condition(self, x, y, noise=None):
        noise = _noise_diagonals_to_matrix(noise)
        y = self.data_transform.transform(y)

        def construct_processes(_):
            posterior_processes = []
            for i, (f, f_noise) in enumerate(self.processes(self.params.processes)):
                obs = self._compute_obs(f, x, y[:, i], noise[:, i] + f_noise)
                posterior_processes.append((f | obs, f_noise))
            return posterior_processes

        return IMOGP(
            dtype=self.dtype,
            processes=construct_processes,
            data_transform=self.data_transform,
            x_ind=self.x_ind,
        ).bind(self)

    def fit(self, x, y, noise=None, **kw_args):
        noise = _noise_diagonals_to_matrix(noise)

        for i in range(self.num_outputs):

            def negative_log_marginal_likelihood(vs):
                with self.use_vs(vs):
                    # Perform the data transform within the objective, because it might
                    # have learnable parameters.
                    yi = self.data_transform.transform(y[:, i], i)

                    # Compute logpdf for output `i`.
                    f, f_noise = self.processes(self.params.processes)[i]
                    obs = self._compute_obs(f, x, yi, noise[:, i] + f_noise)
                    logpdf = f.measure.logpdf(obs) + self.data_transform.logdet(yi, i)
                    return -logpdf / count(yi)

            minimise_l_bfgs_b(
                negative_log_marginal_likelihood,
                self.vs,
                names=self.params.processes[i].all(),
                **kw_args,
            )

    def predict(self, x):
        processes = self.processes(self.params.processes)
        # Compute means and marginal variances.
        mean = B.stack(*[B.squeeze(f.mean(x)) for f, _ in processes], axis=1)
        var = B.stack(*[B.squeeze(f.kernel.elwise(x)) for f, _ in processes], axis=1)
        # Add noise.
        var = var + B.stack(*(noise for _, noise in processes), axis=0)[None, :]
        return self.data_transform.untransform((mean, var))

    def sample(self, x):
        processes = self.processes(self.params.processes)
        sample = B.concat(*[f(x, noise).sample() for f, noise in processes], axis=1)
        return self.data_transform.untransform(sample)
