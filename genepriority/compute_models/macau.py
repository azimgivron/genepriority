# pylint: disable=R0913
"""
Macau Module
========================

This module provides an extension of the `smurff.MacauSession` class to include additional
logging capabilities for TensorBoard. The extension enables tracking training and testing
losses during matrix completion.
"""
import time

import scipy.sparse as sp
import smurff
import tensorflow as tf

from genepriority.compute_models.matrix_completion_result import (
    MatrixCompletionResult,
)


class MacauSession(smurff.MacauSession):
    """
    An extended version of the `smurff.MacauSession` class that integrates
    TensorBoard logging.

    Attributes:
        writer (tf.summary.SummaryWriter): A TensorBoard SummaryWriter
            for logging training and testing metrics.
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
        **kwargs,
    ):
        """
        Initialize the MacauSession.

        Args:
            num_latent (int): Number of latent dimensions in the model.
            nsamples (int): The number of posterior samples to draw during training.
            burnin (int): The number of burn-in iterations before collecting
                posterior samples.
            direct (bool): Whether to use a Cholesky or CG solver.
            univariate (bool): Whether to use univariate or multivariate sampling.
            Ytrain (sp.csr_matrix): Train matrix.
            Ytest (sp.csr_matrix): Test matrix.
            writer (tf.summary.SummaryWriter, optional): TensorBoard SummaryWriter
                instance. Defaults to None.
            center (bool): Whether to center the data.
            **kwargs: Additional arguments to be passed to the
                `smurff.MacauSession` constructor.
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
        self.num_latent = num_latent
        self.nsamples = nsamples
        self.burnin = burnin
        self.direct = direct
        self.univariate = univariate
        self.writer = writer

    def run(self) -> MatrixCompletionResult:
        """
        Executes the matrix completion process and logs
        training and testing metrics to TensorBoard if the writer is provided.

        Returns:
            MatrixCompletionResult: Contains the loss history, RMSE history,
                total iterations, and runtime of the process.
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
                        name="training_loss",
                        data=status_item.train_rmse,
                        step=status_item.iter,
                    )
                    tf.summary.scalar(
                        name="testing_loss",
                        data=status_item.rmse_avg,
                        step=status_item.iter,
                    )
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
