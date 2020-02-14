import warnings

import lab as B
from matrix import AbstractMatrix, Dense, Diagonal
from plum import Dispatcher, Self, Referentiable
from stheno import Graph, GP, Delta, WeightedUnique

__all__ = ['OILMM']


def _per_output(x, y, w):
    p = B.shape(y)[1]

    for i in range(p):
        yi = y[:, i]
        wi = w[:, i]

        # Only return available observations.
        available = ~B.isnan(yi)

        yield x[available], yi[available], wi[available]


class IGP(metaclass=Referentiable):
    """Independent GPs.

    Args:
        kernels (list[:class:`stheno.Kernel`]): Kernels.
        noises (vector): Observation noises.
    """
    _dispatch = Dispatcher(in_class=Self)

    @_dispatch(list, B.Numeric)
    def __init__(self, kernels, noises):
        graph = Graph()
        fs = [GP(kernel, graph=graph) for kernel in kernels]
        IGP.__init__(self, graph, fs, noises)

    @_dispatch(Graph, list, B.Numeric)
    def __init__(self, graph, fs, noises):
        self.graph = graph
        self.fs = fs
        self.fs_noisy = [f + GP(noises[i] * Delta(), graph=graph)
                         for i, f in enumerate(self.fs)]
        self.noises = noises

    def logpdf(self, x, y, w):
        """Compute the logpdf.

        Args:
            x (matrix): Locations of training data.
            y (matrix): Observations of training data.
            w (matrix): Weights of training data.

        Returns:
            scalar: Logpdf.
        """
        logpdf = 0
        for f_noisy, (xi, yi, wi) in zip(self.fs_noisy, _per_output(x, y, w)):
            logpdf = logpdf + f_noisy(WeightedUnique(xi, wi)).logpdf(yi)
        return logpdf

    def condition(self, x, y, w):
        """Condition the model.

        Args:
            x (matrix): Locations of training data.
            y (matrix): Observations of training data.
            w (matrix): Weights of training data.
        """
        fs_post = []
        for f, f_noisy, (xi, yi, wi) in zip(self.fs,
                                            self.fs_noisy,
                                            _per_output(x, y, w)):
            fs_post.append(f | (f_noisy(WeightedUnique(xi, wi)), yi))
        return IGP(self.graph, fs_post, self.noises)

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
        if latent:
            ps = self.fs
        else:
            ps = self.fs_noisy

        means = B.stack(*[B.squeeze(B.dense(p.mean(x)))
                          for p in ps],
                        axis=1)
        variances = B.stack(*[B.squeeze(B.dense(p.kernel.elwise(x)))
                              for p in ps],
                            axis=1)

        if return_variances:
            return means, variances
        else:
            error = 2 * B.sqrt(variances)
            return means, means - error, means + error

    def sample(self, x, latent=False):
        """Sample from the model.

        Args:
            x (matrix): Locations to sample at.
            latent (bool, optional): Sample noiseless processes. Defaults
                to `False`.

        Returns:
            matrix: Sample.
        """
        if latent:
            processes = self.fs
        else:
            processes = self.fs_noisy

        return B.concat(*[p(x).sample() for p in processes], axis=1)


class OILMM(metaclass=Referentiable):
    """Orthogonal Instantaneous Linear Mixing model.

    Args:
        kernels (list[:class:`stheno.Kernel`]) Kernels.
        u (matrix): Orthogonal part of the mixing matrix.
        s_sqrt (matrix): Diagonal part of the mixing matrix.
        noise_obs (scalar): Observation noise.

    """
    _dispatch = Dispatcher(in_class=Self)

    @_dispatch(list, AbstractMatrix, Diagonal, B.Numeric, B.Numeric)
    def __init__(self, kernels, u, s_sqrt, noise_obs, noises_latent):
        OILMM.__init__(self, IGP(kernels, noises_latent), u, s_sqrt, noise_obs)

    @_dispatch(object, AbstractMatrix, Diagonal, B.Numeric)
    def __init__(self, model, u, s_sqrt, noise_obs):
        self.model = model
        self.u = u
        self.s_sqrt = s_sqrt
        self.h = u @ s_sqrt
        self.noise_obs = noise_obs

        self.p, self.m = B.shape(u)

    def logpdf(self, x, y):
        """Compute the logpdf.

        Args:
            x (matrix): Locations of training data.
            y (matrix): Observations of training data.

        Returns:
            scalar: Logpdf.
        """
        proj_x, proj_y, proj_w, reg = self._project(x, y)
        return self.model.logpdf(proj_x, proj_y, proj_w) - reg

    def condition(self, x, y):
        """Condition the model.

        Args:
            x (matrix): Locations of training data.
            y (matrix): Observations of training data.
        """
        self.p = B.shape(y)[1]
        proj_x, proj_y, proj_w, _ = self._project(x, y)
        return OILMM(self.model.condition(proj_x, proj_y, proj_w),
                     self.u,
                     self.s_sqrt,
                     self.noise_obs)

    def _project(self, x, y):
        n = B.shape(x)[0]
        available = ~B.isnan(B.to_numpy(y))

        # Extract patterns.
        patterns = list(set(map(tuple, list(available))))

        if len(patterns) > 30:
            warnings.warn(f'Detected {len(patterns)} patterns, which is more '
                          f'than 30 and can be slow.',
                          category=UserWarning)

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
            proj_x, proj_y, proj_w, reg = \
                self._project_pattern(B.take(x, pattern_inds),
                                      B.take(y, pattern_inds),
                                      pattern)

            proj_xs.append(proj_x)
            proj_ys.append(proj_y)
            proj_ws.append(proj_w)
            total_reg = total_reg + reg

        return B.concat(*proj_xs, axis=0), \
               B.concat(*proj_ys, axis=0), \
               B.concat(*proj_ws, axis=0), \
               total_reg

    def _project_pattern(self, x, y, pattern):
        if all(pattern):
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

        # Build mixing matrix and projection.
        h = B.matmul(u, self.s_sqrt)
        proj = B.solve(self.s_sqrt, B.pinv(u))

        # Perform projection.
        proj_y = B.matmul(y, proj, tr_b=True)

        # Compute projected noise.
        proj_noise = self.noise_obs / B.diag(self.s_sqrt) ** 2 * \
                     B.diag(B.pd_inv(B.matmul(u, u, tr_a=True)))

        # Convert projected noise to weights.
        noises = self.model.noises
        weights = noises / (noises + proj_noise)
        proj_w = B.ones(B.dtype(weights), n, self.m) * weights[None, :]

        # Compute regularising term.
        proj_y_orth = B.subtract(y, B.matmul(proj_y, h, tr_b=True))
        reg = 0.5 * (n * (p - self.m) * B.log(2 * B.pi * self.noise_obs) +
                     B.sum(proj_y_orth ** 2) / self.noise_obs +
                     n * B.logdet(B.matmul(u, u, tr_a=True)) +
                     n * 2 * B.logdet(self.s_sqrt))

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
        means, variances = self.model.predict(x,
                                              latent=latent,
                                              return_variances=True)

        # Pull means and variances through mixing matrix.
        means = B.dense(B.matmul(means, self.h, tr_b=True))
        variances = B.dense(B.matmul(variances, self.h ** 2, tr_b=True))

        if not latent:
            variances = variances + self.noise_obs

        if return_variances:
            return means, variances
        else:
            error = 2 * B.sqrt(variances)
            return means, means - error, means + error

    def sample(self, x, latent=False):
        """Sample from the model.

        Args:
            x (matrix): Locations to sample at.
            latent (bool, optional): Sample noiseless processes. Defaults
                to `False`.

        Returns:
            matrix: Sample.
        """
        sample = B.dense(B.matmul(self.model.sample(x, latent=latent),
                                  self.h,
                                  tr_b=True))
        if not latent:
            sample = sample + B.sqrt(self.noise_obs) * B.randn(sample)
        return sample
