import lab as B
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import wbml.metric as metric
import wbml.out as out
from oilmm.mogp import MOGP
from oilmm.oilmm import ILMM, OILMM
from oilmm.util import count
from plum import Dispatcher
from probmods import Transformed
from stheno import GP, Measure
from wbml.experiment import WorkingDirectory
from wbml.plot import tweak

__all__ = [
    "np",
    "pd",
    "tf",
    "WorkingDirectory",
    "plt",
    "tweak",
    "metric",
    "out",
    "count",
    "ilmm",
]

_dispatch = Dispatcher()

B.epsilon = 1e-10
out.report_time = True


@_dispatch
def ilmm(transformed: Transformed[OILMM]) -> Transformed[ILMM]:
    # Instantiate the model so all parameters are available.
    transformed = transformed()

    def build_correlated_latent_processes(_):
        # Attach them to the same measure space.
        with Measure():
            return [
                (GP(f.mean, f.kernel), noise)
                for f, noise in transformed.model.latent_processes.processes
            ]

    return Transformed(
        transformed.vs,
        ILMM(
            latent_processes=MOGP(build_correlated_latent_processes),
            mixing_matrix=transformed.model.mixing_matrix,
            noise=transformed.model.noise,
            num_outputs=transformed.num_outputs,
        ),
        data_transform=transformed.data_transform,
    )
