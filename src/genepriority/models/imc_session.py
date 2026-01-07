# pylint: disable=R0903
"""
IMC Module
==========

Wrapper around ``negaWsi.IMC`` that adds TensorBoard logging and optional
serialization, mirroring the behavior provided in ``nega_session.py``.
"""
from typing import Union

import negaWsi
import tensorflow as tf

from genepriority.models.utils import check_save_name
from genepriority.utils import calculate_auroc_auprc, serialize


class IMCLogger:
    def __init__(
        self,
        writer: tf.summary.SummaryWriter = None,
        save_name: str = None,
    ) -> None:
        """Initialize logger mixin with optional writer and save path.

        Args:
            writer (tf.summary.SummaryWriter, optional): TensorBoard summary writer for logging.
                If ``None``, disables logging.
            save_name (str, optional): Path to persist the trained model.
        """
        self.save_name = check_save_name(save_name)
        self.writer = writer

    def callback(self) -> None:
        """Log losses and metrics to TensorBoard for the current iteration."""
        if self.writer is not None:
            with self.writer.as_default():
                for d_test, d_train in zip(self.logs["test"], self.logs["training"]):
                    tf.summary.scalar(
                        name="testing_loss", data=d_test, step=self.ith_iteration
                    )
                    tf.summary.scalar(
                        name="training_loss", data=d_train, step=self.ith_iteration
                    )
                    self.ith_iteration += 1

                for key, value in self.loss_terms.items():
                    tf.summary.scalar(name=key, data=value, step=self.ith_iteration)

                test_values_actual = self.matrix[self.test_mask]
                pred = self.predict_all()
                test_predictions = pred[self.test_mask]
                if (self.matrix[self.test_mask] == 0).any():
                    auc, avg_precision = calculate_auroc_auprc(
                        test_values_actual, test_predictions
                    )
                    tf.summary.scalar(name="auc", data=auc, step=self.ith_iteration)
                    tf.summary.scalar(
                        name="average precision",
                        data=avg_precision,
                        step=self.ith_iteration,
                    )
                tf.summary.histogram(
                    "Values on test points", test_predictions, step=self.ith_iteration
                )
                tf.summary.histogram(
                    "Values on training points",
                    pred[self.train_mask],
                    step=self.ith_iteration,
                )
                tf.summary.flush()

    def serialize(self) -> None:
        """Persist model state when a save target is provided."""
        if self.save_name is not None:
            serialize(self, self.save_name)

    def __getstate__(self) -> dict:
        """Remove writer before pickling to keep the object serializable."""
        st = self.__dict__.copy()
        st["writer"] = None
        return st

    def __setstate__(self, st: dict) -> None:
        """Restore serialized state and reset writer."""
        self.__dict__.update(st)
        self.writer = None


class IMC(IMCLogger, negaWsi.IMC):
    def __init__(
        self,
        *args,
        writer: tf.summary.SummaryWriter = None,
        save_name: str = None,
        **kwargs,
    ) -> None:
        """Initialize wrapped IMC with logging/serialization hooks.

        Args:
            writer (tf.summary.SummaryWriter, optional): TensorBoard summary writer for logging.
                If ``None``, disables logging.
            save_name (str, optional): Path to persist the trained model.
            **kwargs: Additional parameters forwarded to ``negaWsi.IMC``.
        """
        IMCLogger.__init__(self, writer, save_name)
        negaWsi.IMC.__init__(self, *args, **kwargs)

    def callback(self) -> None:
        """Delegate logging to ``IMCLogger``."""
        return IMCLogger.callback(self)

    def run(self, log_freq: int = 1) -> negaWsi.Result:
        """Run training, then persist model if requested.

        Args:
            log_freq (int): Logging frequency passed to the base IMC run loop.

        Returns:
            negaWsi.Result: Model training results produced by ``negaWsi.IMC``.
        """
        training_res = super().run(log_freq)
        self.serialize()
        return training_res


IMCSessionType = Union["IMC"]


class IMCSession:
    """
    Simple factory that returns an IMC session with logging/serialization enabled.
    """

    def __new__(cls, *args, **kwargs) -> IMCSessionType:
        """Instantiate the wrapped IMC session.

        Args:
            *args: Positional arguments forwarded to ``IMC``.
            **kwargs: Keyword arguments forwarded to ``IMC``.

        Returns:
            IMC: An instance of the wrapped ``IMC`` class.
        """
        return IMC(*args, **kwargs)
