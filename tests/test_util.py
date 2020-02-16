from lab import B
from matrix import Diagonal
from stheno import Normal

from oilmm.util import Normaliser
from .util import allclose


def test_normaliser():
    # Create test data.
    mat = B.randn(3, 3)
    dist = Normal(mat @ mat.T, B.randn(3, 1))
    y = dist.sample(num=10).T

    # Create normaliser.
    norm = Normaliser(y)
    y_norm = norm.normalise(y)

    # Create distribution of normalised data.
    scale = Diagonal(norm.scale[0])
    dist_norm = Normal(B.inv(scale) @ dist.var @ B.inv(scale),
                       B.inv(scale) @ (dist.mean - norm.mean.T))

    allclose(B.sum(dist.logpdf(y.T)),
             B.sum(dist_norm.logpdf(y_norm.T)) + norm.normalise_logdet(y))
