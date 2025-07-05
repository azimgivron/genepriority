"""GenePriority Module"""
# pylint: disable=R0801
from .evaluation import (
    Evaluation,
    Results,
    auc_per_disease,
    bedroc_per_disease,
    pr_per_disease,
    roc_per_disease,
)
from .postprocessing import (
    ModelEvaluationCollection,
    generate_auc_loss_table,
    generate_bedroc_table,
    plot_auc_boxplots,
    plot_bedroc_boxplots,
    plot_roc_curves,
)
from .preprocessing import (
    DataLoader,
    SideInformationLoader,
    TrainValTestMasks,
    compute_statistics,
    convert_dataframe_to_sparse_matrix,
    filter_by_number_of_association,
    sample_zeros,
)
from .trainer import MACAUTrainer, NEGTrainer

__all__ = [
    "Evaluation",
    "auc_per_disease",
    "bedroc_per_disease",
    "roc_per_disease",
    "pr_per_disease",
    "Results",
    "generate_auc_loss_table",
    "generate_bedroc_table",
    "plot_auc_boxplots",
    "plot_bedroc_boxplots",
    "plot_roc_curves",
    "ModelEvaluationCollection",
    "DataLoader",
    "SideInformationLoader",
    "TrainValTestMasks",
    "convert_dataframe_to_sparse_matrix",
    "sample_zeros",
    "filter_by_number_of_association",
    "compute_statistics",
    "MACAUTrainer",
    "NEGTrainer",
]
