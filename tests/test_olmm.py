import numpy as np
import lab as B
import wbml.lmm
from stheno import EQ
from varz import Vars
from olmm import OLMM, IGP

from .util import approx


def test_logpdf():
    m = 3
    p = 2 * m
    n = 20
    noise = 1e-2
    latent_noise = 2e-2
    kernel = 0.5 * EQ().stretch(0.75)

    x = B.linspace(0, 10, n)

    # Concatenate two orthogonal matrices, to make the missing data
    # approximation exact.
    u1 = B.svd(B.randn(m, m))[0]
    u2 = B.svd(B.randn(m, m))[0]
    u = B.concat(u1, u2, axis=0)

    s_sqrt = B.rand(m)

    # Construct a reference model.
    olmm_pp = wbml.lmm.LMMPP([kernel] * 3,
                             1e-2,
                             latent_noise * B.ones(m),
                             u * s_sqrt[None, :])

    # Sample to generate test data.
    y = olmm_pp.sample(x, latent=False)

    # Throw away data, but retain orthogonality.
    y[5:10, :][:, 3:] = np.nan
    y[10:, :][:, :3] = np.nan

    # Construct OLMM to test.
    vs = Vars(np.float64)
    olmm = OLMM(vs,
                model=IGP(vs=vs,
                          kernel_constructor=lambda vs_: kernel,
                          noise=latent_noise),
                u=u,
                s_sqrt=s_sqrt,
                noise=noise)

    approx(olmm_pp.logpdf(x, y), olmm.logpdf(x, y), decimal=7)
