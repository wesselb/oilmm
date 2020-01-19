import lab as B
from stheno import GP, EQ, Delta, WeightedUnique
from varz import sequential
import wbml.out

__all__ = ['IGP', 'OLMM']


def _eq_constructor(vs):
    return vs.pos(1) * EQ().stretch(vs.pos(1))


def _construct_gps(vs, igp, p):
    fs = []
    es = []

    for i, noise in enumerate(igp.noises(p)):
        fs.append(GP(sequential(f'gp{i}/')(igp.kernel_constructor)(vs)))
        es.append(GP(noise * Delta()))

    return fs, es


class IGP:
    """Independent GPs.

    Args:
        vs (:class:`varz.Vars`): Variable container.
        kernel_constructor (function, optional): Function that takes in a
            variable container and gives back a kernel. Defaults to an
            exponentiated quadratic kernel.
        noise (scalar, optional): Observation noise. Defaults to `1e-2`.
    """

    def __init__(self,
                 vs,
                 kernel_constructor=_eq_constructor,
                 noise=1e-2):
        self.vs = vs
        self.kernel_constructor = kernel_constructor
        self.noise = noise

    def noises(self, p):
        noises = [self.vs.pos(self.noise, name=f'gp{i}/noise')
                  for i in range(p)]
        return B.concat(*[noise[None] for noise in noises])

    def logpdf(self, x, y, w):
        p = B.shape(y)[1]

        # Construct independent GPs.
        fs, es = _construct_gps(self.vs, self, p)

        # Compute logpdf.
        logpdf = 0
        for i in range(p):
            yi = y[:, i]
            wi = w[:, i]

            # Filter missing observations.
            available = ~B.isnan(yi)
            xi = x[available]
            yi = yi[available]
            wi = wi[available]

            f_noisy = fs[i] + es[i]
            logpdf = logpdf + f_noisy(WeightedUnique(xi, wi)).logpdf(yi)

        return logpdf

    def condition(self, x, y, w):
        pass

    def predict(self, x):
        pass


def _pd_inv(a):
    return B.cholsolve(B.chol(B.reg(a)), B.eye(a))


def _pinv(a):
    return B.cholsolve(B.chol(B.reg(B.matmul(a, a, tr_a=True))), B.transpose(a))


class OLMM:
    def __init__(self, vs, model, u, s_sqrt, noise=1e-2):
        self.vs = vs
        self.model = model
        self.u = u
        self.s_sqrt = s_sqrt
        self.noise = noise

        self.p, self.m = B.shape(u)

    def logpdf(self, x, y):
        n = B.shape(x)[0]

        available = ~B.isnan(y)

        # Extract patterns.
        patterns = list(set(map(tuple, list(available))))
        wbml.out.kv('Number of patterns', len(patterns))

        # Per pattern, find data points that belong to it.
        patterns_inds = [[] for _ in range(len(patterns))]
        for i in range(n):
            patterns_inds[patterns.index(tuple(available[i]))].append(i)

        # Per pattern, perform the projection and compute the final logpdf.
        proj_xs = []
        proj_ys = []
        proj_ws = []
        total_reg = 0

        for pattern, pattern_inds in zip(patterns, patterns_inds):
            proj_x, proj_y, proj_w, reg = \
                self._project(B.take(x, pattern_inds),
                              B.take(y, pattern_inds),
                              pattern)

            proj_xs.append(proj_x)
            proj_ys.append(proj_y)
            proj_ws.append(proj_w)
            total_reg = total_reg + reg

        return self.model.logpdf(B.concat(*proj_xs, axis=0),
                                 B.concat(*proj_ys, axis=0),
                                 B.concat(*proj_ws, axis=0)) - total_reg

    def _project(self, x, y, pattern):
        # Filter by the given pattern.
        y = B.take(y, pattern, axis=1)

        # Get number of data points and outputs in this part of the data.
        n = B.shape(x)[0]
        p = sum(pattern)

        # Build mixing matrix and projection.
        u = B.take(self.u, pattern)
        h = u * self.s_sqrt[None, :]
        u_pinv = _pinv(u)
        proj = u_pinv / self.s_sqrt[:, None]

        # Perform projection.
        proj_y = B.matmul(y, proj, tr_b=True)

        # Compute projected noise.
        proj_noise = self.noise / self.s_sqrt ** 2 * \
                     B.diag(_pd_inv(B.matmul(u, u, tr_a=True)))

        # Convert projected noise to weights.
        noises = self.model.noises(self.m)
        weights = noises / (noises + proj_noise)
        proj_w = B.ones(self.vs.dtype, n, self.m) * weights[None, :]

        # Compute regularising term.
        proj_y_orth = y - B.matmul(proj_y, h, tr_b=True)
        reg = 0.5 * (n * (p - self.m) * B.log(2 * B.pi * self.noise) +
                     B.sum(proj_y_orth ** 2) / self.noise +
                     n * B.logdet(B.reg(B.matmul(u, u, tr_a=True))) +
                     n * 2 * B.sum(B.log(self.s_sqrt)))

        return x, proj_y, proj_w, reg
