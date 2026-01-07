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
from typing import Literal, Tuple, Union

import negaWsi
import numpy as np
import tensorflow as tf

from genepriority.models.utils import check_save_name, tsne_plot_to_tensor
from genepriority.utils import calculate_auroc_auprc, serialize


class NegaLogger:
    def __init__(
        self,
        writer: tf.summary.SummaryWriter | None = None,
        save_name: str | None = None,
    ):
        """
        Initializes the NegaLogger instance.

        Args:
            writer (tf.summary.SummaryWriter, optional): TensorFlow summary writer for logging
                training summaries. Defaults to None.
            save_name (str or pathlib.Path, optional): File path where the model will be saved.
                If set to None, the model will not be saved after training. Defaults to None.
        """
        self.save_name = check_save_name(save_name)
        self.writer = writer

    def callback(
        self,
        ith_iteration: int,
        training_loss: np.ndarray,
        testing_loss: np.ndarray,
        grad_f_W_k: np.ndarray,
        step_size: float,
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
            training_loss (np.ndarray): The computed loss value for the training dataset at the
                current iteration.
            testing_loss (np.ndarray): The computed loss value for the testing dataset at the
                current iteration.
            grad_f_W_k (np.ndarray): The gradient of the loss with respect to the model weights
                at the current iteration.
            step_size (float): The step size.
        """
        if self.writer is not None:
            try:
                with self.writer.as_default():
                    tf.summary.scalar(
                        name="step_size", data=step_size, step=ith_iteration
                    )
                    tf.summary.scalar(
                        name="training_loss", data=training_loss, step=ith_iteration
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

    def serialize(self):
        """
        Serialize the model.
        """
        if self.save_name is not None:
            # Avoid serializing the writer
            serialize(self, self.save_name)

    def __getstate__(self):
        st = self.__dict__.copy()
        st["writer"] = None
        return st

    def __setstate__(self, st):
        self.__dict__.update(st)
        self.writer = None


class Nega(NegaLogger, negaWsi.Nega):
    def __init__(
        self,
        *args,
        writer: tf.summary.SummaryWriter | None = None,
        save_name: str | None = None,
        **kwargs,
    ):
        NegaLogger.__init__(self, writer, save_name)
        negaWsi.Nega.__init__(self, *args, **kwargs)

    def callback(self, ith_iteration, training_loss, testing_loss, grad_f_W_k, step_size):
        return NegaLogger.callback(
            self, ith_iteration, training_loss, testing_loss, grad_f_W_k, step_size
        )

    def run(self, log_freq: int = 10) -> negaWsi.Result:
        training_res = super().run(log_freq)
        self.serialize()
        return training_res


class NegaFS(NegaLogger, negaWsi.NegaFS):
    def __init__(
        self,
        *args,
        writer: tf.summary.SummaryWriter | None = None,
        save_name: str | None = None,
        **kwargs,
    ):
        NegaLogger.__init__(self, writer, save_name)
        negaWsi.NegaFS.__init__(self, *args, **kwargs)

    def callback(self, ith_iteration, training_loss, testing_loss, grad_f_W_k, step_size):
        return NegaLogger.callback(
            self, ith_iteration, training_loss, testing_loss, grad_f_W_k, step_size
        )

    def run(self, log_freq: int = 10) -> negaWsi.Result:
        training_res = super().run(log_freq)
        self.serialize()
        return training_res


class NegaReg(NegaLogger, negaWsi.NegaReg):
    def __init__(
        self,
        *args,
        writer: tf.summary.SummaryWriter | None = None,
        save_name: str | None = None,
        **kwargs,
    ):
        NegaLogger.__init__(self, writer, save_name)
        negaWsi.NegaReg.__init__(self, *args, **kwargs)

    def callback(self, ith_iteration, training_loss, testing_loss, grad_f_W_k, step_size):
        return NegaLogger.callback(
            self, ith_iteration, training_loss, testing_loss, grad_f_W_k, step_size
        )

    def run(self, log_freq: int = 10) -> negaWsi.Result:
        training_res = super().run(log_freq)
        self.serialize()
        return training_res


class ENegaFS(NegaLogger, negaWsi.ENegaFS):
    def __init__(
        self,
        *args,
        writer: tf.summary.SummaryWriter | None = None,
        save_name: str | None = None,
        **kwargs,
    ):
        NegaLogger.__init__(self, writer, save_name)
        negaWsi.ENegaFS.__init__(self, *args, **kwargs)

    def callback(self, ith_iteration, training_loss, testing_loss, grad_f_W_k, step_size):
        return NegaLogger.callback(
            self, ith_iteration, training_loss, testing_loss, grad_f_W_k, step_size
        )

    def run(self, log_freq: int = 10) -> negaWsi.Result:
        training_res = super().run(log_freq)
        self.serialize()
        return training_res


NegaSessionType = Union["Nega", "ENegaFS", "NegaFS", "NegaReg"]


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
        formulation: Literal["nega-fs", "nega-reg", "enega"] = None,
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
        if formulation is None:
            model = Nega(*args, **kwargs)
        elif formulation == "nega-fs":
            model = NegaFS(*args, side_info=side_info, **kwargs)
        elif formulation == "enega":
            model = ENegaFS(
                *args, side_info=side_info, ppi_adjacency=ppi_adjacency, **kwargs
            )
        elif formulation == "nega-reg":
            model = NegaReg(
                *args,
                side_info=side_info,
                **kwargs,
            )
        else:
            raise ValueError(
                f"Formulation can only be either 'nega-fs', 'enega' or 'nega-reg' with side information, got {formulation}"
            )
        return model
