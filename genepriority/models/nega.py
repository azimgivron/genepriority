from typing import Union

from genepriority.models.side_information_nega import NegaSi
from genepriority.models.standard_nega import Nega

NegaSessionType = Union["Nega", "NegaSi"]


class NegaSession:
    """
    Factory class that selects and returns an appropriate matrix completion session
    implementation based on the provided parameters.

    This class exposes a unified API for matrix completion. When creating an instance,
    if the `side_info` parameter is provided, an instance of SideInfoMatrixCompletion is returned;
    otherwise, an instance of StandardMatrixCompletion is instantiated.
    """

    def __new__(cls, *args, side_info=None, **kwargs) -> NegaSessionType:
        """
        Creates a new instance of a matrix completion session.

        Args:
            *args: Positional arguments for the underlying session class.
            side_info (tuple or None): A tuple containing side information for genes and diseases.
                If provided, a SideInfoMatrixCompletion instance is created; if None,
                a StandardMatrixCompletion instance is returned.
            **kwargs: Keyword arguments for the underlying session class.

        Returns:
            NegaSessionType: An instance of StandardMatrixCompletion or
                SideInfoMatrixCompletion based on the presence of side_info.
        """
        if side_info is None:
            return Nega(*args, **kwargs)
        return NegaSi(*args, side_info=side_info, **kwargs)
