# noinspection PyUnresolvedReferences
import lab.tensorflow
import numpy as np
import pytest
import tensorflow as tf
from oilmm import ILMM, OILMM
from probmods.test import check_model
from stheno import EQ, GP

# noinspection PyUnresolvedReferences
from .util import approx, oilmm


@pytest.mark.parametrize("LMM", [OILMM, ILMM])
def test_model(LMM):
    def build_latent_processes(ps):
        return [
            (
                p.variance.positive(1) * GP(EQ().stretch(p.length_scale.positive(1))),
                p.noise.positive(1e-2),
            )
            for p, _ in zip(ps, range(2))
        ]

    model = LMM(tf.float64, build_latent_processes, num_outputs=3)
    # Train the data transform.
    model.transform(model.sample(np.linspace(0, 5, 5)))
    check_model(model, rtol=5e-4)
