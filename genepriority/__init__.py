"""GenePriority Module"""
# pylint: disable=R0801
from .evaluation import Evaluation, Results, bedroc_score
from .postprocessing import (
    ModelEvaluationCollection,
    generate_auc_loss_table,
    generate_bedroc_table,
    plot_auc_loss_boxplots,
    plot_bedroc_boxplots,
    plot_roc_curves,
)
from .preprocessing import (
    DataLoader,
    SideInformationLoader,
    TrainTestMasks,
    TrainValTestMasks,
    compute_statistics,
    convert_dataframe_to_sparse_matrix,
    filter_by_number_of_association,
    sample_zeros,
)
from .trainer import MACAUTrainer, NEGTrainer

__all__ = [
    "Evaluation",
    "bedroc_score",
    "Results",
    "generate_auc_loss_table",
    "generate_bedroc_table",
    "plot_auc_loss_boxplots",
    "plot_bedroc_boxplots",
    "plot_roc_curves",
    "ModelEvaluationCollection",
    "DataLoader",
    "SideInformationLoader",
    "TrainTestMasks",
    "TrainValTestMasks",
    "convert_dataframe_to_sparse_matrix",
    "sample_zeros",
    "filter_by_number_of_association",
    "compute_statistics",
    "MACAUTrainer",
    "NEGTrainer",
]
