# pylint: disable=R0903
"""
Nega Module
=============

Factory class that selects and returns an appropriate matrix completion session
implementation based on the provided parameters.

This class exposes a unified API for matrix completion. When creating an instance,
if no side information is passed, an instance of `Nega` is returned; otherwise,
depending on the objective function formulation, an
instance of `NegaFS` or `NegaReg` is returned (wrapped via the `nega` helper).
"""
from typing import Any, Literal, Tuple, Union

import numpy as np
import tensorflow as tf
from negaWsi import Nega as BaseNega, ENegaFS, NegaFS, NegaReg, Result

from genepriority.models.utils import check_save_name, tsne_plot_to_tensor
from genepriority.utils import calculate_auroc_auprc, serialize


def nega(inherit_from: Any):
    """Dynamic Inheritence

    Args:
        inherit_from (Any): Class to inherit from.

    Returns:
        Union[Nega, NegaFS, NegaReg]: The new class.
    """
    name = f"Nega_{inherit_from.__name__}"

    # If we already created it, reuse it (important for stable import path)
    if name in globals():
        return globals()[name]

    class Nega(inherit_from):
        def __init__(
            self,
            *args,
            writer: tf.summary.SummaryWriter | None = None,
            save_name: str | None = None,
            **kwargs,
        ):
            """
            Initializes the BaseNEGA instance with the provided configuration
            parameters for matrix approximation.

            Args:
                writer (tf.summary.SummaryWriter, optional): TensorFlow summary writer for logging
                    training summaries. Defaults to None.
                save_name (str or pathlib.Path, optional): File path where the model will be saved.
                    If set to None, the model will not be saved after training. Defaults to None.
            """
            super().__init__(*args, **kwargs)
            self.save_name = check_save_name(save_name)
            self.writer = writer

        def callback(
            self, ith_iteration: int, testing_loss: np.ndarray, grad_f_W_k: np.ndarray
        ):
            """
            Logs training and evaluation metrics to Tensorboard for the current iteration.

            This method gathers various performance metrics during the optimization process and logs
            them to Tensorboard. It computes the training RMSE, logs the testing loss, and also logs
            histograms and scalar summaries for different evaluation metrics including AUC and average
            precision. In addition, it logs histograms of the predicted values on both
            the training and  testing sets, as well as the flattened gradient values.

            Args:
                ith_iteration (int): The current iteration index at which the logging is performed.
                testing_loss (np.ndarray): The computed loss value for the testing dataset at the
                    current iteration.
                grad_f_W_k (np.ndarray): The gradient of the loss with respect to the model weights
                    at the current iteration.
            """
            if self.writer is not None:
                try:
                    with self.writer.as_default():
                        training_rmse = self.calculate_rmse(self.train_mask)
                        tf.summary.scalar(
                            name="training_loss", data=training_rmse, step=ith_iteration
                        )
                        tf.summary.scalar(
                            name="testing_loss", data=testing_loss, step=ith_iteration
                        )
                        for key, value in self.loss_terms.items():
                            tf.summary.scalar(name=key, data=value, step=ith_iteration)

                        # Extract observed test values and corresponding predictions
                        test_values_actual = self.matrix[self.test_mask]
                        pred = self.predict_all()
                        test_predictions = pred[self.test_mask]
                        if (self.matrix[self.test_mask] == 0).any():
                            auc, avg_precision = calculate_auroc_auprc(
                                test_values_actual, test_predictions
                            )
                            tf.summary.scalar(name="auc", data=auc, step=ith_iteration)
                            tf.summary.scalar(
                                name="average precision",
                                data=avg_precision,
                                step=ith_iteration,
                            )
                        tf.summary.histogram(
                            "Values on test points",
                            test_predictions,
                            step=ith_iteration,
                        )
                        tf.summary.histogram(
                            "Values on training points",
                            pred[self.train_mask],
                            step=ith_iteration,
                        )
                        tf.summary.histogram(
                            "∇f(W^k)", grad_f_W_k.flatten(), step=ith_iteration
                        )
                        if ith_iteration % 1000:
                            fig_h1 = tsne_plot_to_tensor(
                                self.h1, color="#E69F00"
                            )  # shape: (N x rank) or (g x rank)
                            tf.summary.image(
                                "t-SNE: Gene embedding", fig_h1, step=ith_iteration
                            )
                            fig_h2 = tsne_plot_to_tensor(
                                self.h2.T, color="#009E73"
                            )  # shape: (M x rank) or (d x rank)
                            tf.summary.image(
                                "t-SNE: Disease embedding", fig_h2, step=ith_iteration
                            )
                        tf.summary.flush()
                except ValueError as e:
                    self.logger.warning("Tensorboard logging error: %s", e)
                    raise

        def run(self, log_freq: int = 10) -> Result:
            """
            Performs matrix completion using adaptive step size optimization.

            Args:
                log_freq (int, optional): Period at which to log data in Tensorboard.
                    Default to 10 (iterations).

            Returns:
                Result: A dataclass containing:
                    - completed_matrix: The reconstructed matrix (low-rank approximation).
                    - loss_history: List of loss values at each iteration.
                    - rmse_history: List of RMSE values at each iteration.
                    - runtime: Total runtime of the optimization process.
                    - iterations: Total number of iterations performed.
            """
            training_data = super().run(log_freq)
            if self.save_name is not None:
                # Avoid serializing the writer
                serialize(self, self.save_name)
            return training_data

        def __getstate__(self):
            st = self.__dict__.copy()
            st["writer"] = None
            return st

        def __setstate__(self, st):
            self.__dict__.update(st)
            self.writer = None

    # Create the class and register it at module scope
    Nega.__module__ = __name__  # critical: make importable by module path
    Nega.__name__ = name
    Nega.__qualname__ = name
    globals()[name] = Nega  # critical: bind name in module globals
    return Nega


Nega_Nega = nega(BaseNega)
Nega_NegaFS = nega(NegaFS)
Nega_ENegaFS = nega(ENegaFS)
Nega_NegaReg = nega(NegaReg)
NegaSessionType = Union["Nega_BaseNega", "Nega_ENegaFS", "Nega_NegaFS", "Nega_NegaReg"]


class NegaSession:
    """
    Factory class that selects and returns an appropriate matrix completion session
    implementation based on the provided parameters.
    """

    def __new__(
        cls,
        *args,
        side_info: Tuple[np.ndarray, np.ndarray] = None,
        ppi_adjacency: np.ndarray = None,
        formulation: Literal["nega-fs", "nega-reg", "enega"] = "nega-fs",
        **kwargs,
    ) -> NegaSessionType:
        """
        Creates a new instance of a matrix completion session.

        Args:
            *args: Positional arguments for the underlying session class.
            side_info (Tuple[np.ndarray, np.ndarray], optional): A tuple containing side information
                for genes and diseases.
            ppi_adjacency (np.ndarray, optional): PPI graph adjacency matrix. Shape is (n x n).
            svd_init (bool, optional): Whether to initialize the latent
                matrices with SVD decomposition. Default to False.
            formulation: The type of loss formualtion, either "nega-fs", "nega-reg".
                Default to "nega-fs".
            **kwargs: Keyword arguments for the underlying session class.

        Returns:
            NegaSessionType: An instance of StandardMatrixCompletion or
                SideInfoMatrixCompletion based on the presence of side_info.
        """
        if formulation not in ["nega-fs", "nega-reg", "enega"]:
            raise ValueError(
                f"Formulation can only be either 'nega-fs', 'enega' or 'nega-reg', got {formulation}"
            )
        if side_info is None:
            model = nega(BaseNega)(*args, **kwargs)
        elif formulation == "nega-fs":
            model = nega(NegaFS)(*args, side_info=side_info, **kwargs)
        elif formulation == "enega":
            model = nega(ENegaFS)(
                *args, side_info=side_info, ppi_adjacency=ppi_adjacency, **kwargs
            )
        else:
            model = nega(NegaReg)(
                *args,
                side_info=side_info,
                **kwargs,
            )
        return model
