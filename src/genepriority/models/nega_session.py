# pylint: disable=R0903
"""
Nega Module
=============

Factory class that selects and returns an appropriate matrix completion session
implementation based on the provided parameters.

This class exposes a unified API for matrix completion. When creating an instance,
if no side information is passed, an instance of Nega is
is returned; otherwise, depending on the objective function formulation,
IMC of Genehound like, an instance of NegaIMC of an instance of NegaGeneHound is returned.
"""
from typing import Literal, Tuple, Union

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
        formulation: Literal["imc", "GeneHound"] = "imc",
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
            formulation: The type of loss formualtion, either "imc" or "genehound".
                Default to "imc".
            **kwargs: Keyword arguments for the underlying session class.

        Returns:
            NegaSessionType: An instance of StandardMatrixCompletion or
                SideInfoMatrixCompletion based on the presence of side_info.
        """
        if formulation not in ["imc", "genehound"]:
            raise ValueError(
                f"Formulation can only be either 'imc' or 'genehound', got {formulation}"
            )
        if side_info is None:
            model = Nega(*args, **kwargs)
        elif formulation == "imc":
            model = NegaIMC(*args, side_info=side_info, **kwargs)
        else:
            model = NegaGeneHound(
                *args,
                side_info=side_info,
                side_information_reg=side_information_reg,
                **kwargs,
            )
        return model
