import argparse
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from tqdm import tqdm

from genepriority import Evaluation, Results
from genepriority.models.neural.neural_cf import NeuralCF
from genepriority.models.neural.neural_cf_routines import (
    GeneDiseaseDataset,
    predict_full_matrix,
    train_epoch,
    validate_epoch,
)
from genepriority.scripts.utils import pre_processing
from genepriority.utils import serialize


def _pick_device() -> str:
    """Prefer Apple Silicon GPU (MPS), then CUDA, else CPU.

    Returns:
        str: Hardware to use.
    """
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def run_fold(
    device,
    writer,
    args: argparse.Namespace,
    mask_train,
    mask_val,
    mask_test,
    gene_disease,
    gene_feats_scaled,
    disease_feats_scaled,
) -> None:
    """
    Train and evaluate a NeuralCF model for a single cross-validation fold, then
    produce a full gene-disease score matrix.

    Args:
        device (str): Target device ("cuda" or "cpu") used for tensors and the model.
        writer : (SummaryWriter) TensorBoard writer used to log hyperparameters and metrics.
        args : (argparse.Namespace)
            Parsed CLI arguments containing optimization, model, and I/O settings.
        mask_train (np.ndarray): Shape (G, D). Boolean mask indicating training entries
            in the gene-disease matrix.
        mask_val (np.ndarray): Shape (G, D). Boolean mask indicating validation entries.
        mask_test (np.ndarray): Shape (G, D). Boolean mask indicating test entries
            (used only for the Results object).
        gene_disease (np.ndarray): Shape (G, D). Observed gene-disease association
            matrix with 0/1 (or scores).
        gene_feats_scaled (np.ndarray): Shape (G, F_g). Standardized gene
            side-information features.
        disease_feats_scaled (np.ndarray): Shape (D, F_d). Standardized disease
            side-information features.
    """
    # Create index arrays
    train_idx = np.vstack(np.where(mask_train)).T
    val_idx = np.vstack(np.where(mask_val)).T

    # DataLoaders
    train_ds = GeneDiseaseDataset(
        train_idx, gene_disease, gene_feats_scaled, disease_feats_scaled
    )
    val_ds = GeneDiseaseDataset(
        val_idx, gene_disease, gene_feats_scaled, disease_feats_scaled
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # Instantiate model per fold
    num_genes, num_diseases = gene_disease.shape
    model = NeuralCF(
        embedding_dim=args.embedding_dim,
        gene_feat_dim=gene_feats_scaled.shape[1],
        disease_feat_dim=disease_feats_scaled.shape[1],
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
    ).to(device)

    # Summary
    batch = next(iter(train_loader))
    gf = batch["g_feat"].to(device)
    df = batch["d_feat"].to(device)
    model_summary = summary(
        model,
        input_data=(gf, df),
        col_names=("output_size", "num_params", "trainable"),
    )
    writer.add_text("hyperparameters", str(model_summary))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=1e-5,
    )
    criterion = nn.MSELoss()

    if args.load_model is not None:
        checkpoint = torch.load(args.load_model, map_location=device)
        model.load_state_dict(checkpoint)
        logging.info(f"Loaded model weights from {args.load_model}")

    # Training loop
    best_val = float("inf")
    no_improve = 0
    patience = args.patience
    best_state: Dict[str, torch.Tensor] = {}
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, auroc, auprc = validate_epoch(model, val_loader, criterion, device)
        # Scheduler step on validation loss
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("learning_rate", current_lr, epoch)
        writer.add_scalar("training_loss", np.sqrt(train_loss), epoch)
        writer.add_scalar("testing_loss", np.sqrt(val_loss), epoch)
        writer.add_scalar("auc", auroc, epoch)
        writer.add_scalar("average precision", auprc, epoch)

        # Early-stopping logic
        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            # Save a copy of the current best weights
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1

        if no_improve >= patience:
            logging.info(
                f"No improvement for {patience} epochs (stopping at epoch {epoch})."
            )
            break

    writer.close()
    if best_state:
        model.load_state_dict(best_state)
        logging.info(
            "Model weights rolled back to best epoch (val_loss = %.4f).", best_val
        )

    # Full-matrix prediction
    y_pred = predict_full_matrix(
        model,
        num_genes,
        num_diseases,
        gene_feats_scaled,
        disease_feats_scaled,
        device,
        batch_size=args.batch_size,
    )
    return model, Results(gene_disease, y_pred, mask_test)


def ncf(args: argparse.Namespace) -> None:
    """
    Run cross-validation over folds.

    Args:
        args: Parsed command-line arguments.
    """
    output_dir = args.output_path
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    dataloader, side_info_loader = pre_processing(
        gene_disease_path=args.gene_disease_path,
        seed=args.seed,
        omim_meta_path=args.omim_meta_path,
        side_info=args.side_info,
        gene_side_info_paths=args.gene_features_paths + [args.gene_graph_path],
        disease_side_info_paths=args.disease_side_info_paths,
        zero_sampling_factor=args.zero_sampling_factor,
        num_folds=args.num_folds,
        validation_size=args.validation_size,
        max_dims=args.max_dims,
    )

    gene_feats, disease_feats = side_info_loader.side_info
    # Feature scaling
    scaler_g = StandardScaler().fit(gene_feats)
    gene_feats_scaled = scaler_g.transform(gene_feats)
    scaler_d = StandardScaler().fit(disease_feats)
    disease_feats_scaled = scaler_d.transform(disease_feats)
    device = _pick_device()
    for fold, (train_mask, test_mask, validation_mask, _) in tqdm(
        enumerate(dataloader.omim_masks), desc="Folds", unit="fold"
    ):
        # Device & TensorBoard

        fold_log_dir: Path = args.log_dir / f"fold{fold+1}-NeuralCF"
        if fold_log_dir.exists():
            for file in fold_log_dir.iterdir():
                file.unlink()
        fold_log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=fold_log_dir)
        model, res = run_fold(
            device,
            writer,
            args,
            train_mask.toarray(),
            validation_mask.toarray(),
            test_mask,
            dataloader.omim.toarray(),
            gene_feats_scaled,
            disease_feats_scaled,
        )
        results.append(res)
        if args.save_model is not None:
            base_path = args.save_model
            numbered_path = base_path.parent / f"{fold}:{base_path.name}"
            torch.save(model.state_dict(), numbered_path)
            logging.info(f"Saved trained model weights to {numbered_path}")
    evaluation = Evaluation(results)
    serialize(evaluation, output_dir / "results.pickle")
