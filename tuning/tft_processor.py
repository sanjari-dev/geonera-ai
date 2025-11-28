# file: geonera-ai/tuning/tft_processor.py

import logging
import optuna
import pandas as pd
import torch
import os
from torch.optim import AdamW

# Import Lightning Modern
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping

# Import Pytorch Forecasting
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

from utils import log_resource_usage

BATCH_SIZE = 64
MAX_PREDICTION_LENGTH = 120
MAX_ENCODER_LENGTH = 240

# Worker CPU
NUM_WORKERS = min(8, os.cpu_count() - 2) if os.cpu_count() > 2 else 0


def _create_dataset(data: pd.DataFrame, feature_cols: list, training_cutoff: int):
    if "time_idx" not in data.columns:
        data = data.sort_values('timestamp').reset_index(drop=True)
        data['time_idx'] = data.index

    training_data = data[data['time_idx'] <= training_cutoff]

    training = TimeSeriesDataSet(
        training_data,
        time_idx="time_idx",
        target="target_close_future_1",
        group_ids=["instrument"],
        min_encoder_length=MAX_ENCODER_LENGTH // 2,
        max_encoder_length=MAX_ENCODER_LENGTH,
        min_prediction_length=1,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        static_categoricals=["instrument"],
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=feature_cols,
        target_normalizer=GroupNormalizer(
            groups=["instrument"], transformation="softplus"
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True
    )

    validation = TimeSeriesDataSet.from_dataset(
        training, data, predict=True, stop_randomization=True
    )

    return training, validation


def objective(trial, data, feature_cols):
    params = {
        "gradient_clip_val": trial.suggest_float("gradient_clip_val", 0.01, 1.0, log=True),
        "hidden_size": trial.suggest_int("hidden_size", 16, 160, log=True),
        "dropout": trial.suggest_float("dropout", 0.1, 0.4),
        "hidden_continuous_size": trial.suggest_int("hidden_continuous_size", 8, 64, log=True),
        "attention_head_size": trial.suggest_categorical("attention_head_size", [1, 2, 4]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
    }

    max_time_idx = data["time_idx"].max()
    training_cutoff = max_time_idx - (MAX_PREDICTION_LENGTH * 20)

    training_ds, validation_ds = _create_dataset(data, feature_cols, training_cutoff)

    train_dataloader = training_ds.to_dataloader(
        train=True,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0)
    )
    val_dataloader = validation_ds.to_dataloader(
        train=False,
        batch_size=BATCH_SIZE * 2,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0)
    )

    # Model Definition
    tft = TemporalFusionTransformer.from_dataset(
        training_ds,
        learning_rate=params["learning_rate"],
        hidden_size=params["hidden_size"],
        attention_head_size=params["attention_head_size"],
        dropout=params["dropout"],
        hidden_continuous_size=params["hidden_continuous_size"],
        loss=QuantileLoss(),
        optimizer=AdamW,
        reduce_on_plateau_patience=4,
    )

    # Trainer Definition (32-bit Default)
    trainer = pl.Trainer(
        max_epochs=15,
        accelerator="gpu",
        devices=1,
        enable_model_summary=False,
        gradient_clip_val=params["gradient_clip_val"],
        callbacks=[
            EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=3, verbose=False, mode="min")
        ],
        logger=False,
        enable_checkpointing=False
    )

    trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    return trainer.callback_metrics["val_loss"].item()


def run_tft_tuning_pipeline(df: pd.DataFrame, n_trials: int = 10, output_path: str = "resources/tft_best_params.csv"):
    logging.info(f"--- Starting Phase 5: TFT Hyperparameter Tuning (Stable Mode: 32-bit, BS={BATCH_SIZE}) ---")
    log_resource_usage("Start TFT Tuning")

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    logging.info(f"Training Device: {device_name}")

    exclude_cols = ['timestamp', 'instrument', 'timeframe', 'time_idx']
    target_cols = [c for c in df.columns if c.startswith('target_')]

    feature_cols = [c for c in df.columns if c not in exclude_cols and c not in target_cols]

    logging.info(f"Tuning using {len(feature_cols)} features...")

    df = df.sort_values('timestamp').reset_index(drop=True)
    df['time_idx'] = df.index

    study = optuna.create_study(
        direction="minimize",
        study_name="TFT_Tuning",
        pruner=optuna.pruners.HyperbandPruner()
    )

    try:
        study.optimize(
            lambda trial: objective(trial, df, feature_cols),
            n_trials=n_trials,
        )

        logging.info("Tuning Complete.")
        logging.info(f"Best Loss: {study.best_value}")
        logging.info(f"Best Params: {study.best_params}")

        pd.DataFrame([study.best_params]).to_csv(output_path, index=False)
        logging.info(f"Best parameters saved to {output_path}")

    except Exception as e:
        logging.exception(f"Tuning failed: {e}")

    log_resource_usage("End TFT Tuning")
