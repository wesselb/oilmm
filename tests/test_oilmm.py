import lab as B
import matrix
import numpy as np
import probmods.bijection as bijection
from probmods import Transformed
import pytest
from stheno import Measure, GP, EQ

from oilmm import OILMM, ILMM
from oilmm.imogp import IMOGP
from oilmm.mogp import MOGP

# noinspection PyUnresolvedReferences
from .util import approx, increased_regularisation, oilmm


@pytest.mark.parametrize(
    "lmm",
    sum(
        [
            (
                OILMM(
                    np.float64,
                    lambda ps: [
                        (GP(1.0 * EQ().stretch(0.5)), 8e-2),
                        (GP(1.1 * EQ().stretch(0.6)), 7e-2),
                        (GP(1.2 * EQ().stretch(0.7)), 6e-2),
                        (GP(1.3 * EQ().stretch(0.8)), 5e-2),
                    ],
                    noise=noise,
                    # Mixing matrix must be orthogonal. We let it have the right
                    # orthogonal blocks to ensure that the missing data approximation is
                    # exact.
                    mixing_matrix=B.block_diag(
                        B.svd(B.randn(2, 2))[0], B.svd(B.randn(3, 2))[0]
                    ),
                    num_outputs=5,
                    transform="normalise",
                ),
                ILMM(
                    np.float64,
                    lambda ps: [
                        (GP(1.0 * EQ().stretch(0.5)), 8e-2),
                        (GP(1.1 * EQ().stretch(0.6)), 7e-2),
                        (GP(1.2 * EQ().stretch(0.7)), 6e-2),
                        (GP(1.3 * EQ().stretch(0.8)), 5e-2),
                    ],
                    noise=noise,
                    num_outputs=5,
                    transform="normalise",
                ),
            )
            # Test homogeneous and heterogeneous output noise.
            for noise in [5e-1, 5e-1 * B.ones(5)]
        ],
        (),
    ),
)
def test_correctness(lmm, increased_regularisation):
    instance = lmm(lmm.vs)
    # Reduce possible heterogeneous specification to a scalar.
    noise = B.mean(instance.model.noise)
    h = instance.model.mixing_matrix
    lats, noises = zip(*instance.model.latent_processes.processes)

    # Represent the noises on the latent processes as correlated noise in the output
    # space.
    y_noise = matrix.Dense(h) @ matrix.Diagonal(B.stack(*noises)) @ matrix.Dense(h).T

    def build_processes(ps):
        with Measure():
            p, m = B.shape(h)
            xs = [GP(p.mean, p.kernel) for p in lats]  # Copy to current measure.
            fs = [0 for _ in range(p)]
            for i in range(p):
                for j in range(m):
                    fs[i] += h[i, j] * xs[j]
            return [(f, noise) for f in fs]

    mogp = Transformed(np.float64, MOGP(build_processes), transform="normalise")

    x = B.linspace(0, 10, 10)
    x_pred = B.concat(x, 10 * B.rand(10))

    def check_logpdfs(lmm, mogp, mogp_extra_var):
        mogp_extra_var = matrix.TiledBlocks(mogp_extra_var, B.shape(x, 0))
        for y in [lmm.sample(x), mogp.sample(x)]:
            approx(lmm.logpdf(x, y), mogp.logpdf((x, mogp_extra_var), y), rtol=1e-5)

    def check_preds(lmm, mogp, mogp_extra_var):
        # Account for the data transformation.
        mogp_extra_var = B.dense(mogp_extra_var)
        mogp_extra_var = mogp.transform.untransform((0, mogp_extra_var))[1]

        # Make predictions.
        lmm_mean, lmm_var = lmm.predict(x_pred)
        mogp_mean, mogp_var = mogp.predict(x_pred)
        # Make the predictions of `mogp` line up.
        mogp_var = B.diag_extract(mogp_var) + B.diag(mogp_extra_var)[None, :]

        approx(lmm_mean, mogp_mean, rtol=1e-5)
        approx(lmm_var, mogp_var, rtol=1e-5)

    # Check priors.
    check_logpdfs(lmm, mogp, y_noise)
    check_preds(lmm, mogp, y_noise)
    check_preds(lmm.noiseless, mogp.noiseless, 0 * y_noise)

    # Check posteriors.
    y = lmm.sample(x)
    # Drop some data in accordance with the blocks in `H` of the OILMM.
    y[: int(len(x) / 2), -3:] = np.nan
    lmm = lmm.condition(x, y)
    mogp = mogp.condition((x, matrix.TiledBlocks(y_noise, 10)), y)
    check_logpdfs(lmm, mogp, y_noise)
    check_preds(lmm, mogp, y_noise)
    check_preds(lmm.noiseless, mogp.noiseless, 0 * y_noise)


@pytest.mark.parametrize("LMM", [OILMM, ILMM])
@pytest.mark.parametrize(
    "latent_processes",
    [
        lambda _: [(GP(EQ()), 1e-2)] * 3,
    ],
)
@pytest.mark.parametrize(
    "mixing_matrix",
    [
        None,
        B.randn(6, 3),
        lambda ps, p, m: ps.unbounded(shape=(p, m)),
        "random",
    ],
)
@pytest.mark.parametrize("transform", [None, "normalise", bijection.Normaliser()])
def test_contructor(LMM, latent_processes, mixing_matrix, transform):
    lmm = LMM(
        np.float64,
        latent_processes=latent_processes,
        mixing_matrix=mixing_matrix,
        num_outputs=6,
        transform=transform,
    )
    lmm().model.mixing_matrix


@pytest.mark.parametrize("LMM", [OILMM, ILMM])
def test_constructor_invalid_arguments(LMM):
    # Check that the mixing matrix specification must be right.
    with pytest.raises(ValueError):
        LMM(
            np.float64,
            latent_processes=lambda _: [(GP(EQ()), 1e-2)],
            mixing_matrix="invalid",
        )

    # Check that a mixing matrix of the right size must be contructed.
    with pytest.raises(RuntimeError):
        lmm = LMM(
            np.float64,
            latent_processes=lambda _: [(GP(EQ()), 1e-2)],
            mixing_matrix=lambda _, p, m: B.randn(2, 1),
            num_outputs=3,
        )
        lmm().model.mixing_matrix


def test_zero_noise():
    lmm = OILMM(np.float64, lambda _: [(GP(EQ()), 1e-2)], noise=0)
    assert lmm.model.noise is 0
    assert lmm().model.noise is 0


def test_variance_rank_check(mocker, oilmm):
    mocker.patch.object(
        IMOGP,
        "predict",
        return_value=(B.randn(10, 3), 0),
    )
    with pytest.raises(RuntimeError):
        oilmm.predict(B.randn(10))


def test_patterns_warning(oilmm):
    x = B.randn(200)
    y = oilmm.sample(x)

    # Drop data and try to generate more than 30 different patterns.
    y[B.randn(y) > 0] = np.nan
    inds = B.any(~B.isnan(y), axis=1)
    x = x[inds]
    y = y[inds]

    with pytest.warns(UserWarning, match="patterns"):
        oilmm.condition(x, y).predict(x)
