from probmods import Transformed

from .imogp import *
from .mogp import *
from .oilmm import OILMM as _OILMM, ILMM as _ILMM
from .util import *

__all__ = ["OILMM", "ILMM"] + util.__all__ + imogp.__all__ + mogp.__all__


def OILMM(
    dtype,
    latent_processes,
    *args,
    transform="normalise",
    learn_transform=False,
    **kw_args
):
    """OILMM.

    Args:
        dtype (dtype): Data type.
        latent_processes (function): Function which takes in a parameter struct
            and which returns a list of tuples of Gaussian processes and noises.
        noise (scalar, optional): Observation noise. Defaults to `1e-2`.
        mixing_matrix (str or tensor or function, optional): Either the string "random",
            an initial value, or a function which takes in a parameter struct, a height,
            and a width and returns the mixing matrix.
        num_outputs (int, optional): Number of outputs.
        transform (object, optional): Transform. See :func:`probmods.bijector.parse`.
            Defaults to normalising the data.
        learn_transform (bool, optional): Learn parameters in the transform. Defaults
            to `False`.

    Returns:
        :class:`probmods.Model`: OILMM.
    """
    return Transformed(
        dtype,
        _OILMM(IMOGP(latent_processes), *args, **kw_args),
        transform=transform,
        learn_transform=learn_transform,
    )


def ILMM(
    dtype,
    latent_processes,
    *args,
    transform="normalise",
    learn_transform=False,
    **kw_args
):
    """ILMM.

    Args:
        dtype (dtype): Data type.
        latent_processes (function): Function which takes in a parameter struct
            and which returns a list of tuples of Gaussian processes and noises.
        noise (scalar, optional): Observation noise. Defaults to `1e-2`.
        mixing_matrix (str or tensor or function, optional): Either the string "random",
            an initial value, or a function which takes in a parameter struct, a height,
            and a width and returns the mixing matrix.
        num_outputs (int, optional): Number of outputs.
        transform (object, optional): Transform. See :func:`probmods.bijector.parse`.
            Defaults to normalising the data.
        learn_transform (bool, optional): Learn parameters in the transform. Defaults
            to `False`.

    Returns:
        :class:`probmods.Model`: ILMM.
    """
    return Transformed(
        dtype,
        _ILMM(MOGP(latent_processes), *args, **kw_args),
        transform=transform,
        learn_transform=learn_transform,
    )
