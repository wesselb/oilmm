import warnings

import lab as B
from matrix import AbstractMatrix, Dense
from plum import Dispatcher, List
from stheno import GP, Obs, PseudoObs, Kernel

__all__ = ["OILMM", "IGP"]

_dispatch = Dispatcher()


def _per_output(x, y, w):
    p = B.shape(y)[1]

    for i in range(p):
        yi = y[:, i]
        wi = w[:, i]

        # Only return available observations.
        available = ~B.isnan(yi)

        yield x[available], yi[available], wi[available]


def _init_weights(w, y):
    if w is None:
        return B.ones(y)
    else:
        return w


class IGP:
    """Independent GPs.

    Args:
        kernels (list[:class:`stheno.Kernel`]): Kernels.
        noises (vector): Observation noises.
    """

    @_dispatch
    def __init__(self, kernels: List[Kernel], noises: B.Numeric):
        fs = [GP(k) for k in kernels]
        IGP.__init__(self, fs, noises)

    @_dispatch
    def __init__(self, fs: List[GP], noises: B.Numeric):
        self.fs = fs
        self.noises = noises

    def logpdf(self, x, y, w=None, x_ind=None):
        """Compute the logpdf.

        Args:
            x (matrix): Locations of training data.
            y (matrix): Observations of training data.
            w (matrix, optional): Weights of training data.
            x_ind (matrix, optional): Locations of inducing points.

        Returns:
            scalar: Logpdf.
        """
        w = _init_weights(w, y)

        logpdf = 0

        for f, ni, (xi, yi, wi) in zip(self.fs, self.noises, _per_output(x, y, w)):
            if x_ind is None:
                logpdf = logpdf + f(xi, ni / wi).logpdf(yi)
            else:
                obs = PseudoObs(f(x_ind), f(xi, ni / wi), yi)
                logpdf = logpdf + obs.elbo(f.measure)

        return logpdf

    def condition(self, x, y, w=None, x_ind=None):
        """Condition the model.

        Args:
            x (matrix): Locations of training data.
            y (matrix): Observations of training data.
            w (matrix, optional): Weights of training data.
            x_ind (matrix, optional): Locations of inducing points.
        """
        w = _init_weights(w, y)

        fs_post = []
        for f, ni, (xi, yi, wi) in zip(self.fs, self.noises, _per_output(x, y, w)):
            if x_ind is None:
                obs = Obs(f(xi, ni / wi), yi)
            else:
                obs = PseudoObs(f(x_ind), f(xi, ni / wi), yi)
            fs_post.append(f | obs)

        return IGP(fs_post, self.noises)

    def predict(self, x, latent=False, return_variances=False):
        """Predict.

        Args:
            x (matrix): Input locations to predict at.
            latent (bool, optional): Predict noiseless processes. Defaults
                to `False`.
            return_variances (bool, optional): Return means and variances
                instead. Defaults to `False`.

        Returns:
            tuple: Tuple containing means, lower 95% central credible bound,
                and upper 95% central credible bound if `variances` is `False`,
                and means and variances otherwise.
        """
        mean = B.stack(*[B.squeeze(B.dense(f.mean(x))) for f in self.fs], axis=1)
        var = B.stack(*[B.squeeze(f.kernel.elwise(x)) for f in self.fs], axis=1)

        if not latent:
            var = var + self.noises[None, :]

        if return_variances:
            return mean, var
        else:
            error = 1.96 * B.sqrt(var)
            return mean, mean - error, mean + error

    def sample(self, x, latent=False):
        """Sample from the model.

        Args:
            x (matrix): Locations to sample at.
            latent (bool, optional): Sample noiseless processes. Defaults
                to `False`.

        Returns:
            matrix: Sample.
        """
        fdds = [f(x, *(() if latent else (n,))) for f, n in zip(self.fs, self.noises)]
        return B.concat(*[fdd.sample() for fdd in fdds], axis=1)


class OILMM:
    """Orthogonal Instantaneous Linear Mixing model.

    Args:
        kernels (list[:class:`stheno.Kernel`]) Kernels.
        u (matrix): Orthogonal part of the mixing matrix.
        s_sqrt (matrix): Diagonal part of the mixing matrix.
        noise_obs (scalar): Observation noise.

    """

    @_dispatch
    def __init__(
        self, model, u: AbstractMatrix, s_sqrt: AbstractMatrix, noise_obs: B.Numeric
    ):
        self.model = model
        self.u = u
        self.s_sqrt = s_sqrt
        self.h = u @ s_sqrt
        self.noise_obs = noise_obs

        self.p, self.m = B.shape(u)

    @_dispatch
    def __init__(
        self,
        kernels: List[Kernel],
        u: AbstractMatrix,
        s_sqrt: AbstractMatrix,
        noise_obs: B.Numeric,
        noises_latent: B.Numeric,
    ):
        OILMM.__init__(self, IGP(kernels, noises_latent), u, s_sqrt, noise_obs)

    def logpdf(self, x, y, x_ind=None):
        """Compute the logpdf.

        Args:
            x (matrix): Locations of training data.
            y (matrix): Observations of training data.
            x_ind (matrix, optional): Locations of inducing points.

        Returns:
            scalar: Logpdf.
        """
        proj_x, proj_y, proj_w, reg = self.project(x, y)
        return self.model.logpdf(proj_x, proj_y, proj_w, x_ind=x_ind) - reg

    def condition(self, x, y, x_ind=None):
        """Condition the model.

        Args:
            x (matrix): Locations of training data.
            y (matrix): Observations of training data.
            x_ind (matrix, optional): Locations of inducing points.
        """
        self.p = B.shape(y)[1]
        proj_x, proj_y, proj_w, _ = self.project(x, y)
        return OILMM(
            self.model.condition(proj_x, proj_y, proj_w, x_ind=x_ind),
            self.u,
            self.s_sqrt,
            self.noise_obs,
        )

    def project(self, x, y):
        """Project data.

        Args:
            x (matrix): Locations of data.
            y (matrix): Observations of data.

        Returns:
            tuple: Tuple containing the locations of the projection,
                the projection, weights associated with the projection, and
                a regularisation term.
        """
        n = B.shape(x)[0]
        available = ~B.isnan(B.to_numpy(y))

        # Optimise the case where all data is available.
        if B.all(available):
            return self._project_pattern(x, y, (True,) * self.p)

        # Extract patterns.
        patterns = list(set(map(tuple, list(available))))

        if len(patterns) > 30:
            warnings.warn(
                f"Detected {len(patterns)} patterns, which is more "
                f"than 30 and can be slow.",
                category=UserWarning,
            )

        # Per pattern, find data points that belong to it.
        patterns_inds = [[] for _ in range(len(patterns))]
        for i in range(n):
            patterns_inds[patterns.index(tuple(available[i]))].append(i)

        # Per pattern, perform the projection.
        proj_xs = []
        proj_ys = []
        proj_ws = []
        total_reg = 0

        for pattern, pattern_inds in zip(patterns, patterns_inds):
            proj_x, proj_y, proj_w, reg = self._project_pattern(
                B.take(x, pattern_inds), B.take(y, pattern_inds), pattern
            )

            proj_xs.append(proj_x)
            proj_ys.append(proj_y)
            proj_ws.append(proj_w)
            total_reg = total_reg + reg

        return (
            B.concat(*proj_xs, axis=0),
            B.concat(*proj_ys, axis=0),
            B.concat(*proj_ws, axis=0),
            total_reg,
        )

    def _project_pattern(self, x, y, pattern):
        # Check whether all data is available.
        no_missing = all(pattern)

        if no_missing:
            # All data is available. Nothing to be done.
            u = self.u
        else:
            # Data is missing. Pick the available entries.
            y = B.take(y, pattern, axis=1)
            # Ensure that `u` remains a structured matrix.
            u = Dense(B.take(self.u, pattern))

        # Get number of data points and outputs in this part of the data.
        n = B.shape(x)[0]
        p = sum(pattern)

        # Perform projection.
        proj_y_partial = B.matmul(y, B.pinv(u), tr_b=True)
        proj_y = B.matmul(proj_y_partial, B.inv(self.s_sqrt), tr_b=True)

        # Compute projected noise.
        u_square = B.matmul(u, u, tr_a=True)
        proj_noise = (
            self.noise_obs / B.diag(self.s_sqrt) ** 2 * B.diag(B.pd_inv(u_square))
        )

        # Convert projected noise to weights.
        noises = self.model.noises
        weights = noises / (noises + proj_noise)
        proj_w = B.ones(B.dtype(weights), n, self.m) * weights[None, :]

        # Compute Frobenius norm.
        frob = B.sum(y ** 2)
        frob = frob - B.sum(proj_y_partial * B.matmul(proj_y_partial, u_square))

        # Compute regularising term.
        reg = 0.5 * (
            n * (p - self.m) * B.log(2 * B.pi * self.noise_obs)
            + frob / self.noise_obs
            + n * B.logdet(B.matmul(u, u, tr_a=True))
            + n * 2 * B.logdet(self.s_sqrt)
        )

        return x, proj_y, proj_w, reg

    def predict(self, x, latent=False, return_variances=False):
        """Predict.

        Args:
            x (matrix): Input locations to predict at.
            latent (bool, optional): Predict noiseless processes. Defaults
                to `False`.
            return_variances (bool, optional): Return means and variances
                instead. Defaults to `False`.

        Returns:
            tuple[matrix]: Tuple containing means, lower 95% central credible
                bound, and upper 95% central credible bound if `variances` is
                `False`, and means and variances otherwise.
        """
        mean, var = self.model.predict(x, latent=latent, return_variances=True)

        # Pull means and variances through mixing matrix.
        mean = B.dense(B.matmul(mean, self.h, tr_b=True))
        var = B.dense(B.matmul(var, self.h ** 2, tr_b=True))

        if not latent:
            var = var + self.noise_obs

        if return_variances:
            return mean, var
        else:
            error = 1.96 * B.sqrt(var)
            return mean, mean - error, mean + error

    def sample(self, x, latent=False):
        """Sample from the model.

        Args:
            x (matrix): Locations to sample at.
            latent (bool, optional): Sample noiseless processes. Defaults
                to `False`.

        Returns:
            matrix: Sample.
        """
        sample = B.dense(
            B.matmul(self.model.sample(x, latent=latent), self.h, tr_b=True)
        )
        if not latent:
            sample = sample + B.sqrt(self.noise_obs) * B.randn(sample)
        return sample
