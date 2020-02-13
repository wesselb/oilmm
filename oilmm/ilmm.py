from lab import B
from plum import Dispatcher, Referentiable, Self
from stheno import GP, Delta, Graph, Obs

__all__ = ['ILMMPP']


def _to_tuples(x, y):
    """Extract tuples with the input locations, output index,
    and observations from a matrix of observations.

    Args:
        x (tensor): Input locations.
        y (tensor): Outputs.

    Returns:
        list[tuple]: List of tuples with the input locations, output index,
            and observations.
    """
    xys = []
    for i in range(B.shape(y)[1]):
        mask = ~B.isnan(y[:, i])
        if B.any(mask):
            xys.append((x[mask], i, y[mask, i]))

    # Ensure that any data was extracted.
    if len(xys) == 0:
        raise ValueError('No data was extracted.')

    return xys


class ILMMPP(metaclass=Referentiable):
    """Probabilistic programming implementation of the Instantaneous Linear
    Mixing Model.

    Args:
        kernels (list[:class:`stheno.Kernel`]) Kernels.
        h (tensor): Mixing matrix.
        noise_obs (tensor): Observation noise. One.
        noises_latent (tensor): Latent noises. One per latent process.
    """
    _dispatch = Dispatcher(in_class=Self)

    @_dispatch(list, B.Numeric, B.Numeric, B.Numeric)
    def __init__(self, kernels, h, noise_obs, noises_latent):
        self.graph = Graph()
        p, m = B.shape(h)

        # Create latent processes.
        xs = [GP(k, graph=self.graph) for k in kernels]

        # Create latent noise.
        es = [GP(noise * Delta(), graph=self.graph)
              for noise in noises_latent]

        # Create noisy latent processes.
        xs_noisy = [x + e for x, e in zip(xs, es)]

        # Multiply with mixing matrix.
        self.fs = [0 for _ in range(p)]
        for i in range(p):
            for j in range(m):
                self.fs[i] += xs_noisy[j] * h[i, j]

        # Create two observed process.
        self.ys = [f + GP(noise_obs * Delta(), graph=self.graph)
                   for f in self.fs]
        self.y = self.graph.cross(*self.ys)

    @_dispatch(Graph, list, list, GP)
    def __init__(self, graph, fs, ys, y):
        self.graph = graph
        self.fs = fs
        self.ys = ys
        self.y = y

    def logpdf(self, x, y):
        """Compute the logpdf of data.

        Args:
            x (tensor): Input locations.
            y (tensor): Observed values.

        Returns:
            tensor: Logpdf of data.
        """
        obs = Obs(*[(self.ys[i](x), y) for x, i, y in _to_tuples(x, y)])
        return self.graph.logpdf(obs)

    def condition(self, x, y):
        """Condition on data.

        Args:
            x (tensor): Input locations.
            y (tensor): Observed values.
        """
        obs = Obs(*[(self.ys[i](x), y) for x, i, y in _to_tuples(x, y)])
        return ILMMPP(self.graph,
                      [p | obs for p in self.fs],
                      [p | obs for p in self.ys],
                      self.y | obs)

    def predict(self, x):
        """Compute marginals.

        Args:
            x (tensor): Inputs to construct marginals at.

        Returns:
            tuple[matrix]: Tuple containing means, lower 95% central credible
                bound, and upper 95% central credible bound.
        """
        means, lowers, uppers = zip(*[y(x).marginals() for y in self.ys])
        return B.stack(*means, axis=1), \
               B.stack(*lowers, axis=1), \
               B.stack(*uppers, axis=1)

    def sample(self, x, latent=False):
        """Sample data.

        Args:
            x (tensor): Inputs to sample at.
            latent (bool, optional): Sample noiseless processes. Defaults to
                `False`.
        """
        ps = self.fs if latent else self.ys
        samples = self.graph.sample(*[p(x) for p in ps])
        return B.concat(*samples, axis=1)
