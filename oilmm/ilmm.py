from lab import B
from matrix import AbstractMatrix
from plum import Dispatcher, List
from stheno import Measure, GP, Delta, Obs, Kernel

__all__ = ["ILMMPP"]

_dispatch = Dispatcher()


def _per_output(x, y):
    p = B.shape(y)[1]

    for i in range(p):
        yi = y[:, i]

        # Only return available observations.
        available = ~B.isnan(yi)

        yield x[available], i, yi[available]


def _matmul(a, x):
    n, m = B.shape(a)
    out = [0 for _ in range(n)]
    for i in range(n):
        for j in range(m):
            out[i] += a[i, j] * x[j]
    return out


class ILMMPP:
    """Probabilistic programming implementation of the Instantaneous Linear
    Mixing Model.

    Args:
        kernels (list[:class:`stheno.Kernel`]): Kernels.
        h (matrix): Mixing matrix.
        noise_obs (scalar): Observation noise. One.
        noises_latent (vector): Latent noises.
    """

    @_dispatch
    def __init__(
        self,
        measure: Measure,
        xs: List[GP],
        h: AbstractMatrix,
        noise_obs: B.Numeric,
        noises_latent: B.Numeric,
    ):
        self.measure = measure
        self.xs = xs
        self.h = h
        self.noise_obs = noise_obs
        self.noises_latent = noises_latent

        # Create noisy latent processes.
        xs_noisy = [
            x + GP(self.noises_latent[i] * Delta(), measure=self.measure)
            for i, x in enumerate(xs)
        ]

        # Create noiseless observed processes.
        self.fs = _matmul(self.h, self.xs)

        # Create observed processes.
        fs_noisy = _matmul(self.h, xs_noisy)
        self.ys = [
            f + GP(self.noise_obs * Delta(), measure=self.measure) for f in fs_noisy
        ]

    @_dispatch
    def __init__(
        self,
        kernels: List[Kernel],
        h: AbstractMatrix,
        noise_obs: B.Numeric,
        noises_latent: B.Numeric,
    ):
        measure = Measure()

        # Create latent processes.
        xs = [GP(k, measure=measure) for k in kernels]
        ILMMPP.__init__(self, measure, xs, h, noise_obs, noises_latent)

    def logpdf(self, x, y):
        """Compute the logpdf of data.

        Args:
            x (tensor): Input locations.
            y (tensor): Observed values.

        Returns:
            tensor: Logpdf of data.
        """
        obs = Obs(*[(self.ys[i](x), y) for x, i, y in _per_output(x, y)])
        return self.measure.logpdf(obs)

    def condition(self, x, y):
        """Condition on data.

        Args:
            x (tensor): Input locations.
            y (tensor): Observed values.
        """
        obs = Obs(*[(self.ys[i](x), y) for x, i, y in _per_output(x, y)])
        post = self.measure | obs
        return ILMMPP(
            post, [post(x) for x in self.xs], self.h, self.noise_obs, self.noises_latent
        )

    def predict(self, x, latent=False):
        """Compute marginals.

        Args:
            x (tensor): Inputs to construct marginals at.
            latent (bool, optional): Predict noiseless processes. Defaults
                to `False`.

        Returns:
            tuple[matrix]: Tuple containing means, lower 95% central credible
                bound, and upper 95% central credible bound.
        """
        if latent:
            ps = self.fs
        else:
            ps = self.ys

        means, lowers, uppers = zip(*[p(x).marginals() for p in ps])
        return (
            B.stack(*means, axis=1),
            B.stack(*lowers, axis=1),
            B.stack(*uppers, axis=1),
        )

    def sample(self, x, latent=False):
        """Sample data.

        Args:
            x (tensor): Inputs to sample at.
            latent (bool, optional): Sample noiseless processes. Defaults to
                `False`.
        """
        if latent:
            ps = self.fs
        else:
            ps = self.ys

        samples = self.measure.sample(*[p(x) for p in ps])
        return B.concat(*samples, axis=1)
