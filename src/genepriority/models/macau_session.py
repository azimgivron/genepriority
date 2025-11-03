# pylint: disable=R0913,R0903
"""
Macau Module
========================

This module provides an extension of the `smurff.MacauSession` class to include additional
logging capabilities for TensorBoard. The extension enables tracking training and testing
losses during matrix completion. If `smurff` is not installed, the class cannot be instantiated.
"""

import time
from typing import Any

import scipy.sparse as sp
import tensorflow as tf

from negaWsi import Result

try:
    import smurff

    HAS_SMURFF = True
except ImportError:
    smurff = None
    HAS_SMURFF = False


class MacauSession:
    """
    A dynamically defined subclass of `smurff.MacauSession` that integrates TensorBoard logging.

    This wrapper allows import of the class regardless of whether `smurff` is installed.
    However, instantiation will raise an ImportError if `smurff` is missing.

    Attributes:
        writer (tf.summary.SummaryWriter | None): TensorBoard SummaryWriter for logging metrics.
        num_latent (int): Number of latent dimensions.
        nsamples (int): Number of posterior samples.
        burnin (int): Number of burn-in iterations.
        direct (bool): Flag for solver type (Cholesky vs CG).
        univariate (bool): Flag for sampling method.
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        """
        Creates an instance of the internal `_MacauSession` subclass if `smurff` is available.

        Raises:
            ImportError: If the `smurff` library is not installed.

        Returns:
            _MacauSession: A subclass of `smurff.MacauSession` with logging capabilities.
        """
        if not HAS_SMURFF:
            raise ImportError(
                "The 'smurff' library is required to instantiate MacauSession."
            )

        class _MacauSession(smurff.MacauSession):
            """
            Subclass of `smurff.MacauSession` with integrated TensorBoard logging.
            """

            def __init__(
                self,
                num_latent: int,
                nsamples: int,
                burnin: int,
                direct: bool,
                univariate: bool,
                Ytrain: sp.csr_matrix,
                Ytest: sp.csr_matrix,
                writer: tf.summary.SummaryWriter = None,
                center: bool = False,
                **kwargs: Any,
            ) -> None:
                """
                Initialize the MacauSession.

                Args:
                    num_latent (int): Number of latent dimensions in the model.
                    nsamples (int): Number of posterior samples to draw during training.
                    burnin (int): Number of burn-in iterations.
                    direct (bool): Whether to use a Cholesky or CG solver.
                    univariate (bool): Whether to use univariate or multivariate sampling.
                    Ytrain (sp.csr_matrix): Training sparse matrix.
                    Ytest (sp.csr_matrix): Testing sparse matrix.
                    writer (tf.summary.SummaryWriter | None): TensorBoard SummaryWriter instance.
                    center (bool): Whether to center the data before training.
                    **kwargs (Any): Additional keyword arguments for `smurff.MacauSession`.
                """
                if center:
                    Ytrain, global_mean, _ = smurff.center_and_scale(
                        Ytrain, "global", with_mean=True, with_std=False
                    )
                    Ytest.data -= global_mean

                super().__init__(
                    num_latent=num_latent,
                    nsamples=nsamples,
                    burnin=burnin,
                    direct=direct,
                    univariate=univariate,
                    Ytrain=Ytrain,
                    Ytest=Ytest,
                    threshold=0.5,
                    **kwargs,
                )
                self.num_latent = num_latent
                self.nsamples = nsamples
                self.burnin = burnin
                self.direct = direct
                self.univariate = univariate
                self.writer = writer

            def run(self) -> MatrixCompletionResult:
                """
                Executes matrix completion and logs training and testing metrics to TensorBoard.

                Returns:
                    MatrixCompletionResult: Contains training loss history, RMSE history,
                    total iterations, and runtime.
                """
                start_time = time.time()
                self.init()
                rmse = []
                loss = []
                status_item = self.step()
                while status_item is not None:
                    if self.writer is not None and status_item.phase == "Sample":
                        with self.writer.as_default():
                            tf.summary.scalar(
                                "training_loss",
                                status_item.train_rmse,
                                step=status_item.iter,
                            )
                            tf.summary.scalar(
                                "testing_loss",
                                status_item.rmse_avg,
                                step=status_item.iter,
                            )
                            tf.summary.scalar(
                                "auc",
                                status_item.auc_avg,
                                step=status_item.iter,
                            )
                            tf.summary.flush()
                    iterations_count = status_item.iter
                    rmse.append(status_item.rmse_avg)
                    loss.append(status_item.train_rmse)
                    status_item = self.step()

                runtime = time.time() - start_time

                return Result(
                    loss_history=loss,
                    iterations=iterations_count,
                    rmse_history=rmse,
                    runtime=runtime,
                )

        instance = _MacauSession(*args, **kwargs)
        return instance
