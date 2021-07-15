import warnings
from types import FunctionType

import lab as B
import numpy as np
from matrix import TiledBlocks, AbstractMatrix
from plum import Dispatcher, convert
from varz import minimise_l_bfgs_b
from probmods.bijection import parse as _parse_transform
from probmods.model import MultiOutputModel

from .imogp import IMOGP
from .mogp import MOGP
from .util import count

__all__ = ["OILMM", "ILMM"]

_dispatch = Dispatcher()


class AbstractILMM(MultiOutputModel):
    """Instantaneous Linear Mixing Model.

    Args:
        dtype (dtype): Data type.
        latent_processes (function or :class:`wbml.model.ProbabilisticModel`): Function
            which takes in a parameters struct and returns a list of tuples of
            :class:`stheno.GP`s and noises, which correspond to the models for the
            latent processes or another model.
        noise (scalar, optional): Observation noise. Defaults to `1e-2`.
        mixing_matrix (str or tensor or function, optional): Either the string "random",
            an initial value, or a function which takes in a parameter struct, a height,
            and a width and returns the mixing matrix.
        data_transform (str or :class:`.bijection.Bijection`, optional): Specification
            for a data transformation. Defaults to "normalise".
        num_outputs (int, optional): Number of outputs. Will be automatically inferred.
    """

    def __init__(
        self,
        dtype,
        latent_processes,
        noise=1e-2,
        mixing_matrix=None,
        data_transform="normalise",
        num_outputs=None,
    ):
        MultiOutputModel.__init__(self, dtype)
        self.latent_processes = _parse_latent_processes(self, latent_processes)
        self._noise = noise
        self._mixing_matrix = _parse_mixing_matrix(self, mixing_matrix)
        self.data_transform = _parse_transform(data_transform)
        self.num_outputs = num_outputs

    @property
    def noise(self):
        """scalar: Observation noise."""
        if self._noise is 0:
            return 0
        else:
            return self.params.noise.positive(self._noise)

    @property
    def mixing_matrix(self):
        """matrix: Mixing matrix."""
        h = self._mixing_matrix(
            self.params.mixing_matrix,
            self.num_outputs,
            self.latent_processes.num_outputs,
        )
        if B.shape(h) != (self.num_outputs, self.latent_processes.num_outputs):
            raise RuntimeError(
                f"Constructor for mixing matrix construct a matrix of shape "
                f"{B.shape(h)}, but shape "
                f"({self.num_outputs}, {self.latent_processes.num_outputs}) was "
                f"expected."
            )
        return h

    @property
    def noiseless(self):
        # This is either an ILMM or OILMM. We need to preserve the type.
        return type(self)(
            dtype=self.dtype,
            latent_processes=self.latent_processes.noiseless,
            noise=0,
            mixing_matrix=self._mixing_matrix,
            data_transform=self.data_transform,
            num_outputs=self.num_outputs,
        ).bind(self)

    def _init(self, y):
        self.num_outputs = B.shape(y, 1)

    def logpdf(self, x, y):
        self._init(y)
        y, proj_x, proj_y, proj_n, reg = self.project(x, y)
        return (
            self.latent_processes.logpdf(proj_x, proj_y, proj_n)
            - reg
            + self.data_transform.logdet(y)
        )

    def condition(self, x, y):
        self._init(y)
        _, proj_x, proj_y, proj_n, _ = self.project(x, y)
        # This is either an ILMM or OILMM. We need to preserve the type.
        return type(self)(
            dtype=self.dtype,
            latent_processes=self.latent_processes.condition(proj_x, proj_y, proj_n),
            noise=self._noise,
            mixing_matrix=self._mixing_matrix,
            data_transform=self.data_transform,
            num_outputs=self.num_outputs,
        ).bind(self)

    def fit(self, x, y, **kw_args):
        self._init(y)

        def negative_log_marginal_likelihood(vs):
            with self.use_vs(vs):
                return -self.logpdf(x, y) / count(y)

        minimise_l_bfgs_b(negative_log_marginal_likelihood, self.vs, **kw_args)

    def project(self, x, y):
        """Project data.

        Args:
            x (matrix): Locations of data.
            y (matrix): Observations of data.

        Returns:
            tuple: Tuple containing the transformed data, the locations of the
                projection, the projected data, the projected noise, and a
                regularisation term.
        """
        self._init(y)

        # Perform data transformation and check for missing data.
        y = self.data_transform(y)
        # We convert `available` to NumPy to efficiently compute the available patterns.
        available = B.jit_to_numpy(~B.isnan(y))

        # We will need the mixing matrix multiple times.
        h = self.mixing_matrix

        # Optimise the case where all data is available.
        if B.all(available):
            return (y,) + self._project_pattern(
                x, y, np.array([True] * self.num_outputs), h
            )

        # Extract patterns. We convert to bytes for hashing.
        available_patterns = [row.tobytes() for row in available]
        patterns = list(set(available_patterns))

        if len(patterns) > 30:
            warnings.warn(
                f"Detected {len(patterns)} patterns, which is more "
                f"than 30 and can be slow.",
                category=UserWarning,
            )

        # Per pattern, find data points that belong to it.
        patterns_index = {pattern: i for i, pattern in enumerate(patterns)}
        patterns_inds = [[] for _ in patterns]
        for i, pattern in enumerate(available_patterns):
            patterns_inds[patterns_index[pattern]].append(i)

        # Per pattern, perform the projection.
        projs = []
        total_reg = 0
        for pattern_inds in patterns_inds:
            # Extract a mask by just taking the first index.
            mask = available[pattern_inds[0]]
            proj_x, proj_y, proj_n, reg = self._project_pattern(
                B.take(x, pattern_inds),
                B.take(y, pattern_inds),
                mask,
                h,
            )
            projs.append((proj_x, proj_y, proj_n))
            total_reg = total_reg + reg

        # Concatenate the projections for all patterns and return.
        proj_xs, proj_ys, proj_ns = zip(*projs)
        return (
            y,
            B.concat(*proj_xs, axis=0),
            B.concat(*proj_ys, axis=0),
            B.concat(*proj_ns, axis=0),
            total_reg,
        )

    def _project_pattern(self, x, y, mask, h):
        m = self.latent_processes.num_outputs

        if not B.all(mask):
            # Data is missing. Pick the available entries.
            y = B.take(y, mask, axis=1)
            h = B.take(h, mask, axis=0)

        # Ensure that `h` is a structured matrix for dispatch.
        h = convert(h, AbstractMatrix)

        # Get number of data points and outputs in this part of the data.
        n = B.shape(x, 0)
        p = sum(mask)

        # Perform projection.
        proj_y = B.matmul(y, B.pinv(h), tr_b=True)

        # Compute projected noise.
        h_square = B.matmul(h, h, tr_a=True)
        proj_n = B.multiply(self.noise, B.pd_inv(h_square))

        # Compute Frobenius norm.
        frob = B.sum(y ** 2)
        frob = frob - B.sum(proj_y * B.matmul(proj_y, h_square))

        # Compute regularising term.
        reg = 0.5 * (
            n * (p - m) * B.log(2 * B.pi * self.noise)
            + frob / self.noise
            + n * B.logdet(h_square)
        )

        # Repeat the projected noise for every time stamp.
        proj_n = TiledBlocks(proj_n, n, axis=0)

        return x, proj_y, proj_n, reg

    def predict(self, x):
        # Make predictions for the latent processes.
        mean, var = self.latent_processes.predict(x)

        # Pull means and variances through mixing matrix.
        h = self.mixing_matrix
        mean = B.dense(B.matmul(mean, h, tr_b=True))
        # TODO: Simplify this if-statement once batched matrices are available.
        if B.rank(var) == 2:
            var = B.dense(B.matmul(var, h ** 2, tr_b=True))
        elif B.rank(var) == 3:
            var = B.dense(B.sum(h * B.matmul(h, var), axis=2))
        else:
            raise RuntimeError(f"Invalid rank {B.rank(var)} of variance.")

        # Add noise.
        var = var + self.noise

        return self.data_transform.untransform((mean, var))

    def sample(self, x):
        # Sample from the latent processes.
        sample = self.latent_processes.sample(x)

        # Pull sample through mixing matrix.
        sample = B.dense(B.matmul(sample, self.mixing_matrix, tr_b=True))

        # Add noise.
        sample = sample + B.sqrt(self.noise) * B.randn(sample)

        return self.data_transform.untransform(sample)


class OILMM(AbstractILMM):
    """Orthogonal ILMM. See :class:`.AbstractILMM`."""


class ILMM(AbstractILMM):
    """ILMM. See :class:`.AbstractILMM`."""


@_dispatch
def _parse_latent_processes(model: AbstractILMM, spec: MultiOutputModel):
    """Parse a specification for the latent processes.

    Args:
        model (:class:`.AbstractILMM`): Instance of the model.
        spec (object): Specification.

    Returns:
        :class:`wbml.model.MultiOutputModel`: Model for the latent processes.
    """
    return spec


@_dispatch
def _parse_latent_processes(model: OILMM, spec: FunctionType):
    return IMOGP(dtype=model.dtype, processes=spec)


@_dispatch
def _parse_latent_processes(model: ILMM, spec: FunctionType):
    return MOGP(dtype=model.dtype, processes=spec)


@_dispatch.abstract
def _parse_mixing_matrix(model: AbstractILMM, spec: str):
    """Parse a specification for the mixing matrix.

    Args:
        model (:class:`.AbstractILMM`): Instance of the model.
        spec (object): Specification.

    Returns:
        function: Appropriate function that instantiates the mixing matrix. See
            :class:`.AbstractILMM`.
    """


@_dispatch
def _parse_mixing_matrix(model: OILMM, spec: None):
    return _parse_mixing_matrix(model, "random")


@_dispatch
def _parse_mixing_matrix(model: OILMM, spec: str):
    if spec == "random":

        def mixing_matrix(params, p, m):
            return params.u.orthogonal(shape=(p, m))

    else:
        raise ValueError(f'Unknown mixing matrix specification "{spec}".')

    return mixing_matrix


@_dispatch
def _parse_mixing_matrix(model: OILMM, spec: B.Numeric):
    def mixing_matrix(params, p, m):
        return params.u.orthogonal(spec, shape=(p, m))

    return mixing_matrix


@_dispatch
def _parse_mixing_matrix(model: ILMM, spec: None):
    return _parse_mixing_matrix(model, "random")


@_dispatch
def _parse_mixing_matrix(model: ILMM, spec: str):
    if spec == "random":

        def mixing_matrix(params, p, m):
            return params.h.unbounded(shape=(p, m))

    else:
        raise ValueError(f'Unknown mixing matrix specification "{spec}".')

    return mixing_matrix


@_dispatch
def _parse_mixing_matrix(model: ILMM, spec: B.Numeric):
    def mixing_matrix(params, p, m):
        return params.h.unbounded(spec, shape=(p, m))

    return mixing_matrix


@_dispatch
def _parse_mixing_matrix(model: AbstractILMM, spec: FunctionType):
    return spec
