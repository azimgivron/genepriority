# pylint: disable=R0913
"""
Macau Module
========================

This module provides an extension of the `smurff.MacauSession` class to include additional
logging capabilities for TensorBoard. The extension enables tracking training and testing
losses during matrix completion. If `smurff` is not installed, the class cannot be instantiated.
"""

import time
from typing import Any, Optional

import scipy.sparse as sp
import tensorflow as tf

from genepriority.models.matrix_completion_result import MatrixCompletionResult

try:
    import smurff
    _has_smurff = True
except ImportError:
    smurff = None
    _has_smurff = False


class MacauSession:
    """
    A dynamically defined subclass of `smurff.MacauSession` that integrates TensorBoard logging.

    This wrapper allows import of the class regardless of whether `smurff` is installed.
    However, instantiation will raise an ImportError if `smurff` is missing.
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        """
        Creates an instance of the internal `_MacauSession` subclass if `smurff` is available.

        Raises:
            ImportError: If the `smurff` library is not installed.

        Returns:
            An instance of a subclass of `smurff.MacauSession` with logging capabilities.
        """
        if not _has_smurff:
            raise ImportError("The 'smurff' library is required to instantiate MacauSession.")

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
                writer: Optional[tf.summary.SummaryWriter] = None,
                center: bool = False,
                **kwargs: Any,
            ) -> None:
                """
                Initialize the MacauSession.

                Args:
                    num_latent (int): Number of latent dimensions in the model.
                    nsamples (int): The number of posterior samples to draw during training.
                    burnin (int): The number of burn-in iterations.
                    direct (bool): Whether to use a Cholesky or CG solver.
                    univariate (bool): Whether to use univariate or multivariate sampling.
                    Ytrain (sp.csr_matrix): Training matrix.
                    Ytest (sp.csr_matrix): Testing matrix.
                    writer (tf.summary.SummaryWriter, optional): TensorBoard writer.
                    center (bool): Whether to center the data before training.
                    **kwargs: Additional keyword arguments for `smurff.MacauSession`.
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
                    **kwargs,
                )
                self.num_latent: int = num_latent
                self.nsamples: int = nsamples
                self.burnin: int = burnin
                self.direct: bool = direct
                self.univariate: bool = univariate
                self.writer: Optional[tf.summary.SummaryWriter] = writer

            def run(self) -> MatrixCompletionResult:
                """
                Executes matrix factorization and logs metrics to TensorBoard if a writer is set.

                Returns:
                    MatrixCompletionResult: Contains training loss, RMSE history,
                    total iterations, and elapsed runtime.
                """
                start_time = time.time()
                self.init()
                rmse: list[float] = []
                loss: list[float] = []
                status_item = self.step()
                while status_item is not None:
                    if self.writer is not None and status_item.phase == "Sample":
                        with self.writer.as_default():
                            tf.summary.scalar("training_loss", status_item.train_rmse, step=status_item.iter)
                            tf.summary.scalar("testing_loss", status_item.rmse_avg, step=status_item.iter)
                            tf.summary.flush()
                    iterations_count = status_item.iter
                    rmse.append(status_item.rmse_avg)
                    loss.append(status_item.train_rmse)
                    status_item = self.step()

                runtime = time.time() - start_time

                return MatrixCompletionResult(
                    loss_history=loss,
                    iterations=iterations_count,
                    rmse_history=rmse,
                    runtime=runtime,
                )

        instance = object.__new__(_MacauSession)
        instance.__init__(*args, **kwargs)
        return instance
