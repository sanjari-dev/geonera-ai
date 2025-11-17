# file: geonera-ai/selection/phase4_stability.py

import logging
import pandas as pd
import catboost as cb
from sklearn.multioutput import MultiOutputRegressor

from .common import (
    _split_data_components,
    _get_volatility_regime,
    _get_aggregated_importances,
    PROTECTED_VOL_FEATURE
)


def run_phase4_selection(df: pd.DataFrame, importance_quantile: float = 0.5) -> pd.DataFrame | None:
    logging.info(f"Phase 4 Input Shape: {df.shape}")
    split_result = _split_data_components(df)
    if split_result is None:
        logging.error("Phase 4: Failed to split data components. Aborting.")
        return None
    id_cols, target_cols, feature_cols, protected_cols = split_result

    if PROTECTED_VOL_FEATURE not in protected_cols:
        logging.error(f"Phase 4: Required feature '{PROTECTED_VOL_FEATURE}' not found in protected columns. Aborting.")
        return None

    vol_feature = PROTECTED_VOL_FEATURE
    X_df = df[feature_cols]
    y_df = df[target_cols]

    if X_df.empty:
        logging.warning("Phase 4: No features found to run stability test on. Skipping.")
        return df

    logging.info(f"Phase 4: Separated {len(feature_cols)} features for stability testing.")
    logging.info(f"Phase 4: Labeling data with volatility regimes based on '{vol_feature}'")

    df['regime'] = _get_volatility_regime(df[vol_feature])
    regimes = ['LOW_VOL', 'MID_VOL', 'HIGH_VOL']
    regime_importances = {}

    for regime in regimes:
        logging.info(f"--- Processing Regime: {regime} ---")
        regime_mask = (df['regime'] == regime)
        X_regime = df.loc[regime_mask, feature_cols]
        y_regime = df.loc[regime_mask, target_cols]

        if len(X_regime) < 100:
            logging.warning(f"Regime '{regime}' has only {len(X_regime)} samples. Skipping.")
            continue

        logging.info(f"Phase 4 ({regime}): Training CatBoost model on {len(X_regime)} samples")
        core_model = cb.CatBoostRegressor(
            iterations=500,
            verbose=100,
            random_state=42,
            task_type='GPU',
            allow_writing_files=False
        )
        multi_model = MultiOutputRegressor(core_model)
        try:
            multi_model.fit(X_regime, y_regime)
            avg_imp = _get_aggregated_importances(multi_model, X_regime.columns.tolist())
            regime_importances[regime] = avg_imp
            logging.info(f"Phase 4 ({regime}): Model training and importance extraction complete.")
        except Exception as e:
            logging.error(f"Phase 4 ({regime}): Failed to train model: {e}")

    if len(regime_importances) < len(regimes):
        logging.error("Phase 4: Failed to train on all regimes. Aborting stability analysis.")
        return None

    logging.info("Phase 4: Analyzing feature stability across all regimes")
    stability_df = pd.DataFrame(regime_importances)
    quantiles = stability_df.quantile(importance_quantile)

    logging.info(f"Phase 4: Stability threshold (quantile={importance_quantile}):")
    logging.info(f"  LOW_VOL:  > {quantiles.get('LOW_VOL', 0):.4f}")
    logging.info(f"  MID_VOL:  > {quantiles.get('MID_VOL', 0):.4f}")
    logging.info(f"  HIGH_VOL: > {quantiles.get('HIGH_VOL', 0):.4f}")

    stable_mask = pd.Series(True, index=stability_df.index)
    for regime in regime_importances.keys():
        stable_mask = stable_mask & (stability_df[regime] > quantiles[regime])

    stable_features = stability_df[stable_mask].index.tolist()
    dropped_count = len(feature_cols) - len(stable_features)

    if not stable_features:
        logging.error(f"Phase 4: No features were found to be stable across all regimes (using quantile={importance_quantile}). Aborting.")
        return None

    logging.info(f"Phase 4: Kept {len(stable_features)} stable features.")
    logging.info(f"Phase 4: Dropped {dropped_count} unstable features.")

    X_selected_df = X_df[stable_features]
    final_df = pd.concat([df[id_cols], y_df, df[protected_cols], X_selected_df], axis=1)

    logging.info(f"Phase 4: Final shape: {final_df.shape}.")
    return final_df