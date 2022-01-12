import warnings
from types import FunctionType
from typing import Union

import lab as B
import numpy as np
from matrix import TiledBlocks, AbstractMatrix, Diagonal, Zero
from plum import Dispatcher, convert
from probmods import Model, instancemethod, cast

from .imogp import IMOGP
from .mogp import MOGP

__all__ = ["OILMM", "ILMM"]

_dispatch = Dispatcher()


class AbstractILMM(Model):
    """Instantaneous Linear Mixing Model.

    Args:
        latent_processes (model): Model for the latent processes.
        noise (scalar or vector, optional): Observation noise. Defaults to `1e-2`. If
            the argument is a scalar, thjen the noise will be homogeneous across
            outputs. On the other hand, if the argument is a vector, then the noise will
            be heterogeneous across outputs.
        mixing_matrix (str or tensor or function, optional): Either the string "random",
            an initial value, or a function which takes in a parameter struct, a height,
            and a width and returns the mixing matrix.
        num_outputs (int, optional): Number of outputs.
    """

    def __init__(
        self,
        latent_processes,
        noise=1e-2,
        mixing_matrix=None,
        num_outputs=None,
    ):
        self.latent_processes = latent_processes
        self.noise = noise
        self._mixing_matrix = _parse_mixing_matrix(self, mixing_matrix)
        self.num_outputs = num_outputs

    def __prior__(self):
        self.latent_processes = self.latent_processes(self.ps.latent_processes)
        if self.noise is 0:
            self.noise = 0
        else:
            self.noise = self.ps.noise.positive(self.noise)

    @property
    def noise_matrix(self):
        """matrix: Lazily construct the noise as a matrix: the number of outputs may not
        yet be know at construction time."""
        if self.noise is 0:
            return Zero(self.dtype, self.num_outputs, self.num_outputs)
        elif B.is_scalar(self.noise):
            return B.fill_diag(self.noise, self.num_outputs)
        else:
            return Diagonal(self.noise)

    @property
    def mixing_matrix(self):
        """matrix: Lazily construct the mixing matrix: the number of outputs may not
        yet be known at construction time."""
        h = self._mixing_matrix(
            self.ps.mixing_matrix,
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

    def __noiseless__(self):
        self.latent_processes.__noiseless__()
        self.noise = 0

    def _init(self, y):
        self.num_outputs = B.shape(y, 1)

    @cast
    def __condition__(self, x, y):
        self._init(y)
        proj_x, proj_y, proj_n, _ = self.project(x, y)
        self.latent_processes.__condition__((proj_x, proj_n), proj_y)

    @instancemethod
    @cast
    def logpdf(self, x, y):
        self._init(y)
        proj_x, proj_y, proj_n, reg = self.project(x, y)
        return self.latent_processes.logpdf((proj_x, proj_n), proj_y) - reg

    @instancemethod
    @cast
    def project(self, x, y):
        """Project data.

        Args:
            x (matrix): Locations of data.
            y (matrix): Observations of data.

        Returns:
            tuple: The locations of the projection, the projected data, the projected
                noise, and a regularisation term.
        """
        self._init(y)

        # We convert `available` to NumPy to efficiently compute the available patterns.
        available = B.jit_to_numpy(~B.isnan(y))

        # We will need the mixing matrix multiple times.
        h = self.mixing_matrix

        # Optimise the case where all data is available.
        if B.all(available):
            return self._project_pattern(x, y, np.array([True] * self.num_outputs), h)

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
                B.take(x, pattern_inds), B.take(y, pattern_inds), mask, h
            )
            projs.append((proj_x, proj_y, proj_n))
            total_reg = total_reg + reg

        # Concatenate the projections for all patterns and return.
        proj_xs, proj_ys, proj_ns = zip(*projs)
        return (
            B.concat(*proj_xs, axis=0),
            B.concat(*proj_ys, axis=0),
            B.concat(*proj_ns, axis=0),
            total_reg,
        )

    def _project_pattern(self, x, y, mask, h):
        m = self.latent_processes.num_outputs
        noise = self.noise_matrix

        if not B.all(mask):
            # Data is missing. Pick the available entries.
            y = B.take(y, mask, axis=1)
            h = B.take(h, mask, axis=0)
            noise = B.submatrix(noise, mask)

        # Ensure that `h` is a structured matrix for dispatch.
        h = convert(h, AbstractMatrix)

        # Get number of data points and outputs in this part of the data.
        n = B.shape(x, 0)
        p = sum(mask)

        # Perform projection.
        proj_y = B.matmul(y, B.pinv(h), tr_b=True)

        # Compute projected noise. Carefully handle the case that `self.noise = 0`.
        if isinstance(noise, Zero):
            # Noise is zero, which means that the projected noise is zero too. The
            # regulariser is not defined in this case.
            proj_n = Zero(self.dtype, m, m)
            reg = np.nan
        else:
            # Noise is nonzero. Everything can be computed safely.
            proj_n_inv = B.matmul(h, B.inv(noise), h, tr_a=True)
            proj_n = B.pd_inv(proj_n_inv)

            # Compute Frobenius norm.
            rec_err = B.subtract(y, B.matmul(proj_y, h, tr_b=True))
            frob = B.sum(B.matmul(rec_err, B.inv(noise)) * rec_err)

            # Compute regularising term.
            reg = 0.5 * (
                n * (p - m) * B.log(2 * B.pi)
                + n * (B.logdet(noise) + B.logdet(proj_n_inv))
                + frob
            )

        # Repeat the projected noise for every time stamp.
        proj_n = TiledBlocks(proj_n, n, axis=0)

        return x, proj_y, proj_n, reg

    @instancemethod
    @cast
    def predict(self, x):
        # Make predictions for the latent processes.
        mean, var = self.latent_processes.predict(x)

        # Pull means and variances through mixing matrix.
        h = self.mixing_matrix
        mean = B.matmul(mean, h, tr_b=True)
        # TODO: Simplify this if-statement once batched matrices are available.
        if B.rank(var) == 2:
            var = B.matmul(var, h ** 2, tr_b=True)
        elif B.rank(var) == 3:
            var = B.sum(h * B.matmul(h, var), axis=2)
        else:
            raise RuntimeError(f"Invalid rank {B.rank(var)} of variance.")

        # Add noise.
        var = B.add(var, B.expand_dims(B.diag(self.noise_matrix), axis=0))

        # Return prediction as plain tensors to ease dispatch.
        return B.dense(mean), B.dense(var)

    @instancemethod
    @cast
    def sample(self, x):
        # Sample from the latent processes.
        sample = self.latent_processes.sample(x)

        # Pull sample through mixing matrix.
        sample = B.matmul(sample, self.mixing_matrix, tr_b=True)

        # Add noise.
        noise = self.noise_matrix
        sample = B.add(sample, B.matmul(B.randn(sample), B.chol(noise), tr_b=True))

        # Return sample as plain tensor to ease dispatch.
        return B.dense(sample)


class OILMM(AbstractILMM):
    """Orthogonal ILMM. See :class:`.AbstractILMM`."""


class ILMM(AbstractILMM):
    """ILMM. See :class:`.AbstractILMM`."""


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

        def mixing_matrix(ps, p, m):
            return ps.u.orthogonal(shape=(p, m))

    else:
        raise ValueError(f'Unknown mixing matrix specification "{spec}".')

    return mixing_matrix


@_dispatch
def _parse_mixing_matrix(model: OILMM, spec: B.Numeric):
    def mixing_matrix(ps, p, m):
        return ps.u.orthogonal(spec, shape=(p, m))

    return mixing_matrix


@_dispatch
def _parse_mixing_matrix(model: ILMM, spec: None):
    return _parse_mixing_matrix(model, "random")


@_dispatch
def _parse_mixing_matrix(model: ILMM, spec: str):
    if spec == "random":

        def mixing_matrix(ps, p, m):
            return ps.h.unbounded(shape=(p, m))

    else:
        raise ValueError(f'Unknown mixing matrix specification "{spec}".')

    return mixing_matrix


@_dispatch
def _parse_mixing_matrix(model: ILMM, spec: B.Numeric):
    def mixing_matrix(ps, p, m):
        return ps.h.unbounded(spec, shape=(p, m))

    return mixing_matrix


@_dispatch
def _parse_mixing_matrix(model: AbstractILMM, spec: FunctionType):
    return spec
