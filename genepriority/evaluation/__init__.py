"""Evaluation Module"""
# pylint disable=R0801
from .evaluation import Evaluation
from .metrics import bedroc_scores, auc_scores
from .results import Results

def average_auc_score():
    return

def average_bedroc_score():
    return

def average_roc_curve():
    return

__all__ = [
    "Evaluation",
    "bedroc_scores",
    "auc_scores",
    "Results",
]
