import warnings

import lab as B
import numpy as np
from stheno import Graph, GP, EQ, Delta, WeightedUnique, Obs
from varz import Vars, sequential

__all__ = ['IGP', 'OLMM']


def _eq_constructor(vs):
    return vs.pos(1) * EQ().stretch(vs.pos(1))


def _construct_gps(vs, igp, p):
    g = Graph()
    fs = []
    es = []

    for i, noise in enumerate(igp.noises(p)):
        kernel = sequential(f'gp{i}/')(igp.kernel_constructor)(vs)
        fs.append(GP(kernel, graph=g))
        es.append(GP(noise * Delta(), graph=g))

    return fs, es


def _per_output(x, y, w):
    p = B.shape(y)[1]

    for i in range(p):
        yi = y[:, i]
        wi = w[:, i]

        # Only return available observations.
        available = ~B.isnan(yi)

        yield x[available], yi[available], wi[available]


class IGP:
    """Independent GPs.

    Args:
        vs (:class:`varz.Vars`, optional): Variable container.
        kernel_constructor (function, optional): Function that takes in a
            variable container and gives back a kernel. Defaults to an
            exponentiated quadratic kernel.
        noise (scalar, optional): Observation noise. Defaults to `1e-2`.
    """

    def __init__(self,
                 vs=None,
                 kernel_constructor=_eq_constructor,
                 noise=1e-2):
        if vs is None:
            vs = Vars(np.float64)

        self.vs = vs
        self.kernel_constructor = kernel_constructor
        self.noise = noise

        self.p = None
        self.x_train = None
        self.y_train = None
        self.w_train = None

    def noises(self, p):
        noises = [self.vs.pos(self.noise, name=f'gp{i}/noise')
                  for i in range(p)]
        return B.concat(*[noise[None] for noise in noises])

    def logpdf(self, x, y, w):
        logpdf = 0
        for (f, e), (xi, yi, wi) in zip(zip(*_construct_gps(self.vs,
                                                            self,
                                                            B.shape(y)[1])),
                                        _per_output(x, y, w)):
            f_noisy = f + e
            logpdf = logpdf + f_noisy(WeightedUnique(xi, wi)).logpdf(yi)
        return logpdf

    def condition(self, x, y, w):
        self.p = B.shape(y)[1]

        self.x_train = x
        self.y_train = y
        self.w_train = w

    def predict(self, x, latent=False):
        means = []
        vars = []
        for (f, e), (xi, yi, wi) in zip(zip(*_construct_gps(self.vs,
                                                            self,
                                                            self.p)),
                                        _per_output(self.x_train,
                                                    self.y_train,
                                                    self.w_train)):
            obs = Obs((f + e)(WeightedUnique(xi, wi)), yi)
            if latent:
                post = f | obs
            else:
                post = (f + e) | obs
            means.append(B.squeeze(B.dense(post.mean(x))))
            vars.append(B.squeeze(B.dense(post.kernel.elwise(x))))
        return B.stack(*means, axis=1), B.stack(*vars, axis=1)

    def sample(self, x, p, latent=False):
        samples = []
        for f, e in zip(*_construct_gps(self.vs, self, p)):
            if latent:
                process = f
            else:
                process = f + e
            samples.append(B.squeeze(process(x).sample()))
        return B.stack(*samples, axis=1)


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
        self.h = u * s_sqrt[None, :]
        self.noise = noise

        self.p, self.m = B.shape(u)

    def logpdf(self, x, y):
        proj_x, proj_y, proj_w, reg = self._project(x, y)
        return self.model.logpdf(proj_x, proj_y, proj_w) - reg

    def _project(self, x, y):
        n = B.shape(x)[0]
        available = ~B.isnan(B.to_numpy(y))

        # Extract patterns.
        patterns = list(set(map(tuple, list(available))))

        if len(patterns) > 30:
            warnings.warn(f'Detected {len(patterns)} patterns, which is more '
                          f'than 30.',
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

    def condition(self, x, y):
        self.p = B.shape(y)[1]
        proj_x, proj_y, proj_w, _ = self._project(x, y)
        self.model.condition(proj_x, proj_y, proj_w)

    def predict(self, x, latent=False):
        means, vars = self.model.predict(x, latent=latent)

        # Pull means and variances through mixing matrix.
        means = B.matmul(means, self.h, tr_b=True)
        vars = B.matmul(vars, self.h ** 2, tr_b=True)

        if not latent:
            vars = vars + self.noise

        return means, vars

    def sample(self, x, latent=False):
        latent_sample = self.model.sample(x, p=self.m, latent=latent)
        observed_sample = B.matmul(latent_sample, self.h, tr_b=True)
        if not latent:
            observed_sample = observed_sample + \
                              B.sqrt(self.noise) * B.randn(observed_sample)
        return observed_sample
