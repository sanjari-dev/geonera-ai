# file: ai/selection/phase3_shap.py

import logging
import pandas as pd
import numpy as np
import catboost as cb
import shap
import optuna
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from .common import _split_data_components


def run_phase3_selection(
        df: pd.DataFrame,
        n_trials: int = 20,
        shap_scores_path: str = "phase3_shap_scores.csv",
        n_top_features: int = 250
) -> pd.DataFrame | None:
    logging.info(f"Phase 3 Input Shape: {df.shape}")

    split_result = _split_data_components(df)
    if split_result is None:
        logging.error("Phase 3: Failed to split data components. Aborting.")
        return None
    id_cols, target_cols, feature_cols, protected_cols = split_result

    X_df = df[feature_cols]
    y_df = df[target_cols]

    if X_df.empty:
        logging.error("Phase 3: No features found to run selection on. Aborting.")
        return df  # Return original df, no features to select

    logging.info(f"Phase 3: Separated {len(feature_cols)} features for relevance testing.")

    X_train, X_val, y_train, y_val = train_test_split(
        X_df, y_df, test_size=0.2, random_state=42, shuffle=False
    )
    logging.info(f"Split data for tuning: {len(X_train)} train, {len(X_val)} validation.")

    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 100, 1000),
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
            'random_state': 42,
            'allow_writing_files': False,
            'task_type': 'GPU',
            'verbose': 200
        }
        core_model = cb.CatBoostRegressor(**params)
        multi_model = MultiOutputRegressor(core_model)
        multi_model.fit(X_train, y_train)
        preds = multi_model.predict(X_val)
        mse_list = [mean_squared_error(y_val.iloc[:, idx], preds[:, idx]) for idx in range(y_val.shape[1])]
        avg_mse = np.mean(mse_list)
        return avg_mse

    logging.info(f"Phase 3: Starting Optuna hyperparameter tuning (n_trials={n_trials}) on GPU")
    optuna.logging.set_verbosity(optuna.logging.INFO)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    logging.info(f"Phase 3: Optuna tuning complete. Best Validation MSE: {study.best_value}")
    logging.info(f"Phase 3: Best parameters found: {best_params}")

    final_core_model = cb.CatBoostRegressor(
        **best_params,
        random_state=42,
        allow_writing_files=False,
        task_type='GPU',
        verbose=200
    )
    final_multi_model = MultiOutputRegressor(final_core_model)

    logging.info("Phase 3: Training final CatBoost model on *all* data with best parameters (on GPU)")
    final_multi_model.fit(X_df, y_df)
    logging.info("Phase 3: Final model training complete.")

    logging.info("Phase 3: Calculating SHAP values for all models (This may take a long time)")
    all_shap_scores = []

    for i, estimator in enumerate(final_multi_model.estimators_):
        target_name = target_cols[i]
        logging.info(f"  Calculating SHAP for target: {target_name}")
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer(X_df)
        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        all_shap_scores.append(mean_abs_shap)

    logging.info("Phase 3: Aggregating SHAP scores")
    final_shap_scores = pd.DataFrame(all_shap_scores, columns=X_df.columns).mean(axis=0)
    final_shap_scores = final_shap_scores.sort_values(ascending=False)

    important_features = final_shap_scores[final_shap_scores > 0]

    if important_features.empty:
        logging.error("Phase 3: No features found with SHAP value > 0. Aborting selection, passing all columns.")
        return df

    kept_features = important_features.head(n_top_features).index.tolist()
    dropped_count = len(feature_cols) - len(kept_features)

    logging.info(f"Phase 3: Kept {len(kept_features)} top features (out of {len(important_features)} with SHAP > 0).")
    logging.info(f"Phase 3: Dropped {dropped_count} features.")

    try:
        final_shap_scores.to_csv(shap_scores_path)
        logging.info(f"Phase 3: SHAP scores saved to {shap_scores_path}")
    except Exception as e:
        logging.warning(f"Phase 3: Could not save SHAP scores: {e}")

    X_selected_df = X_df[kept_features]
    final_df = pd.concat([df[id_cols], y_df, df[protected_cols], X_selected_df], axis=1)

    logging.info(f"Phase 3: Final shape: {final_df.shape}.")
    return final_df