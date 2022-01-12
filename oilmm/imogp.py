import lab as B
from matrix import TiledBlocks
from plum import Dispatcher
from probmods.model import Model, Transformed, cast, fit, instancemethod
from stheno import Obs, PseudoObs
from varz import minimise_l_bfgs_b

from .util import count, parse_input

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


class IMOGP(Model):
    """A multi-output GP consisting of independent GPs.

    Args:
        processes (function): Function that returns a list of tuples of
            :class:`stheno.GP`s and noises, which correspond to the models for the
            outputs.
        x_ind (tensor, optional): Initialisation for inducing inputs.
    """

    def __init__(self, processes, x_ind=None):
        self._processes = processes
        self._x_ind = x_ind

    def __prior__(self):
        self.processes = self._processes(self.ps.processes)
        self.num_outputs = len(self.processes)
        if self._x_ind is None:
            self.x_ind = None
        else:
            self.x_ind = self.ps.x_ind.unbounded(self._x_ind)

    @cast
    def __condition__(self, x, y):
        x, noise = parse_input(x)
        noise = _noise_diagonals_to_matrix(noise)
        posterior_processes = []
        for i, (f, f_noise) in enumerate(self.processes):
            obs = self._compute_obs(f, x, y[:, i], noise[:, i] + f_noise)
            posterior_processes.append((f | obs, f_noise))
        self.processes = posterior_processes

    def __noiseless__(self):
        self.processes = [(f, 0) for f, _ in self.processes]

    def _compute_obs(self, f, x, y, noise):
        if self.x_ind is None:
            return Obs(f(x, noise), y)
        else:
            return PseudoObs(f(self.x_ind), f(x, noise), y)

    @instancemethod
    @cast
    def logpdf(self, x, y):
        x, noise = parse_input(x)
        noise = _noise_diagonals_to_matrix(noise)
        logpdf = 0
        for i, (f, f_noise) in enumerate(self.processes):
            obs = self._compute_obs(f, x, y[:, i], noise[:, i] + f_noise)
            logpdf = logpdf + f.measure.logpdf(obs)
        return logpdf

    @instancemethod
    @cast
    def predict(self, x):
        processes = self.processes
        # Compute means and marginal variances.
        mean = B.stack(*[B.squeeze(f.mean(x)) for f, _ in processes], axis=1)
        var = B.stack(*[B.squeeze(f.kernel.elwise(x)) for f, _ in processes], axis=1)
        # Add noise.
        var = var + B.stack(*(noise for _, noise in processes), axis=0)[None, :]
        return mean, var

    @instancemethod
    @cast
    def sample(self, x):
        return B.concat(*[f(x, noise).sample() for f, noise in self.processes], axis=1)


@fit.dispatch
def fit(
    model: Transformed[IMOGP],
    x,
    y,
    minimiser=minimise_l_bfgs_b,
    trace=True,
    **kw_args,
):
    x, noise = parse_input(x)
    noise = _noise_diagonals_to_matrix(noise)

    # Make sure that the data transform is fit.
    model.transform.transform(y)

    for i in range(model().num_outputs):

        def normalised_negative_log_marginal_likelihood(vs):
            instance = model(vs)

            # Transform the data.
            yi = y[:, i]
            yi_transformed = instance.transform.transform(yi, i)

            # Compute logpdf for output `i`.
            f, f_noise = instance.model.processes[i]
            obs = instance.model._compute_obs(
                f,
                x,
                yi_transformed,
                noise[:, i] + f_noise,
            )
            logpdf = f.measure.logpdf(obs) + instance.transform.logdet(yi, i)
            return -logpdf / count(yi)

        minimiser(
            normalised_negative_log_marginal_likelihood,
            model.vs,
            names=model.vs.struct.processes[i].all(),
            trace=trace,
            **kw_args,
        )
