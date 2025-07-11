"""
Nega Module
=============

Factory class that selects and returns an appropriate matrix completion session
implementation based on the provided parameters.

This class exposes a unified API for matrix completion. When creating an instance,
if the `side_info` and `side_information_reg` parameters are provided, an instance of NegaGenehound
is returned; if `side_info` only is provided, then an instance of NegaIMC is returned;
otherwise, an instance of Nega is instantiated.
"""
from typing import Tuple, Union

import numpy as np

from genepriority.models.nega_genehound import NegaGeneHound
from genepriority.models.nega_imc import NegaIMC
from genepriority.models.nega_standard import Nega

NegaSessionType = Union["Nega", "NegaIMC", "NegaGeneHound"]


class NegaSession:
    """
    Factory class that selects and returns an appropriate matrix completion session
    implementation based on the provided parameters.
    """

    def __new__(
        cls,
        *args,
        side_info: Tuple[np.ndarray, np.ndarray] = None,
        side_information_reg: float = None,
        svd_init: bool = False,
        **kwargs,
    ) -> NegaSessionType:
        """
        Creates a new instance of a matrix completion session.

        Args:
            *args: Positional arguments for the underlying session class.
            side_info (Tuple[np.ndarray, np.ndarray], optional): A tuple containing side information
                for genes and diseases.
            side_information_reg (float, optional): Regularization weight for
                for the side information.
            svd_init (bool, optional): Whether to initialize the latent
                matrices with SVD decomposition. Default to False.
            **kwargs: Keyword arguments for the underlying session class.

        Returns:
            NegaSessionType: An instance of StandardMatrixCompletion or
                SideInfoMatrixCompletion based on the presence of side_info.
        """
        if side_info is None:
            return Nega(*args, svd_init=svd_init, **kwargs)
        if side_information_reg is None:
            return NegaIMC(*args, side_info=side_info, svd_init=svd_init, **kwargs)
        return NegaGeneHound(
            *args,
            side_info=side_info,
            side_information_reg=side_information_reg,
            **kwargs,
        )
