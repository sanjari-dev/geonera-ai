# feature_selector.py

import pandas as pd
import numpy as np
import logging
from sklearn.feature_selection import VarianceThreshold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as spd
import catboost as cb
import shap
import optuna

PROTECTED_VOL_FEATURE = 'atr_20'

def _split_data_components(df: pd.DataFrame) -> tuple[list, list, list, list] | None:
    id_cols = ['timestamp', 'instrument', 'timeframe']
    target_cols = [col for col in df.columns if col.startswith('target_')]
    if PROTECTED_VOL_FEATURE not in df.columns:
        logging.error(f"Core Logic Error: Protected feature '{PROTECTED_VOL_FEATURE}' not found in DataFrame.")
        return None
    protected_cols = [PROTECTED_VOL_FEATURE]
    feature_cols = [col for col in df.columns if
                    col not in id_cols and
                    col not in target_cols and
                    col not in protected_cols]
    if not feature_cols:
        logging.error("No feature columns found after splitting components.")
        return None
    return id_cols, target_cols, feature_cols, protected_cols


def run_phase1_selection(df: pd.DataFrame) -> pd.DataFrame:
    logging.info(f"Phase 1 Input Shape: {df.shape}")
    id_cols = ['timestamp', 'instrument', 'timeframe']
    target_cols = [col for col in df.columns if col.startswith('target_')]
    feature_cols = [col for col in df.columns if col not in id_cols and col not in target_cols]
    if not feature_cols:
        logging.error("Phase 1: No feature columns found. Aborting.")
        return df
    features_df = df[feature_cols]
    logging.info(f"Phase 1: Separated {len(feature_cols)} features for selection.")
    logging.info("Phase 1: Applying Variance Threshold (threshold=0.0)")
    selector = VarianceThreshold(threshold=0.0)
    try:
        selector.fit(features_df)
        kept_cols_variance = selector.get_feature_names_out()
        features_df = features_df[kept_cols_variance]
        dropped_count = len(feature_cols) - len(kept_cols_variance)
        logging.info(f"Phase 1: Dropped {dropped_count} zero-variance features.")
    except ValueError as e:
        logging.error(f"Phase 1: Error during Variance Threshold (skipping): {e}")
    logging.info("Phase 1: Applying Duplicate Column Detection (This may be slow)")
    features_T = features_df.T
    duplicated_mask = features_T.duplicated()
    cols_to_drop_duplicates = features_T[duplicated_mask].index.tolist()
    if cols_to_drop_duplicates:
        features_df = features_df.drop(columns=cols_to_drop_duplicates)
        logging.info(f"Phase 1: Dropped {len(cols_to_drop_duplicates)} duplicate features.")
    else:
        logging.info("Phase 1: No duplicate features found.")
    logging.info("Phase 1: Re-combining ID columns, targets, and selected features")
    final_df = pd.concat([df[id_cols], df[target_cols], features_df], axis=1)
    logging.info(f"Phase 1: Final shape: {final_df.shape}.")
    return final_df


def run_phase2_selection(df: pd.DataFrame, corr_threshold: float = 0.95) -> pd.DataFrame | None:
    logging.info(f"Phase 2 Input Shape: {df.shape}")
    split_result = _split_data_components(df)
    if split_result is None:
        logging.error("Phase 2: Failed to split data components. Aborting.")
        return None
    id_cols, target_cols, feature_cols, protected_cols = split_result
    X_df = df[feature_cols]
    y_df = df[target_cols]
    logging.info(f"Phase 2: Separated {len(feature_cols)} features for clustering.")
    logging.info("Phase 2: Calculating Spearman correlation matrix (This may take a long time)")
    corr_matrix = X_df.corr(method='spearman').abs()
    logging.info("Phase 2: Performing hierarchical clustering")
    dist_matrix = 1 - corr_matrix
    dist_condensed = spd.squareform(dist_matrix, checks=False)
    Z = sch.linkage(dist_condensed, method='average')
    distance_threshold = 1 - corr_threshold
    logging.info(f"Phase 2: Forming flat clusters at distance_threshold={distance_threshold} (corr > {corr_threshold})")
    cluster_labels = sch.fcluster(Z, t=distance_threshold, criterion='distance')
    feature_clusters = pd.Series(cluster_labels, index=X_df.columns, name="cluster_id")
    logging.info(f"Phase 2: Identified {feature_clusters.nunique()} unique clusters from {len(feature_cols)} features.")
    logging.info("Phase 2: Calculating feature-target correlations (Spearman) to select representatives")
    X_y_df = pd.concat([X_df, y_df], axis=1)
    X_y_corr = X_y_df.corr(method='spearman').abs()
    target_correlations = X_y_corr.loc[X_df.columns, y_df.columns]
    avg_target_corr = target_correlations.mean(axis=1)
    df_with_scores = pd.DataFrame({'cluster_id': feature_clusters, 'target_corr': avg_target_corr})
    logging.info("Phase 2: Selecting best representative from each cluster")
    best_features = []
    for cluster_id in df_with_scores['cluster_id'].unique():
        cluster_features = df_with_scores[df_with_scores['cluster_id'] == cluster_id]
        best_feature_in_cluster = cluster_features['target_corr'].idxmax()
        best_features.append(best_feature_in_cluster)
    logging.info(f"Phase 2: Selected {len(best_features)} representative features.")
    X_selected_df = X_df[best_features]
    final_df = pd.concat([df[id_cols], y_df, df[protected_cols], X_selected_df], axis=1)
    logging.info(f"Phase 2: Final shape: {final_df.shape}.")
    return final_df


def run_phase3_selection(
        df: pd.DataFrame,
        n_trials: int = 20,
        shap_scores_path: str = "phase3_shap_scores.csv"
) -> pd.DataFrame | None:
    """
    Runs Phase 3 selection: CatBoost + SHAP.
    'atr_20' is protected from this filter.
    Saves SHAP scores to shap_scores_path.
    """
    logging.info(f"Phase 3 Input Shape: {df.shape}")

    split_result = _split_data_components(df)
    if split_result is None:
        logging.error("Phase 3: Failed to split data components. Aborting.")
        return None
    id_cols, target_cols, feature_cols, protected_cols = split_result

    X_df = df[feature_cols]
    y_df = df[target_cols]

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

    kept_features = final_shap_scores[final_shap_scores > 0].index.tolist()
    dropped_count = len(feature_cols) - len(kept_features)

    if not kept_features:
        logging.error("Phase 3: No features found with SHAP value > 0. Aborting selection, passing all columns.")
        return df

    logging.info(f"Phase 3: Kept {len(kept_features)} features (SHAP > 0).")
    logging.info(f"Phase 3: Dropped {dropped_count} zero-importance features.")

    try:
        final_shap_scores.to_csv(shap_scores_path)
        logging.info(f"Phase 3: SHAP scores saved to {shap_scores_path}")
    except Exception as e:
        logging.warning(f"Phase 3: Could not save SHAP scores: {e}")

    X_selected_df = X_df[kept_features]
    final_df = pd.concat([df[id_cols], y_df, df[protected_cols], X_selected_df], axis=1)

    logging.info(f"Phase 3: Final shape: {final_df.shape}.")
    return final_df


def _get_volatility_regime(atr_series: pd.Series) -> pd.Series:
    low_thresh = atr_series.quantile(0.33)
    high_thresh = atr_series.quantile(0.66)
    logging.info(f"Phase 4: Volatility thresholds calculated: LOW < {low_thresh:.5f}, MID < {high_thresh:.5f}")

    def label_regime(atr_val):
        if atr_val < low_thresh:
            return 'LOW_VOL'
        elif atr_val < high_thresh:
            return 'MID_VOL'
        else:
            return 'HIGH_VOL'

    return atr_series.apply(label_regime)


def _get_aggregated_importances(model: MultiOutputRegressor, feature_names: list) -> pd.Series:
    importances_list = []
    for estimator in model.estimators_:
        importances_list.append(estimator.feature_importances_)
    avg_importances = np.mean(importances_list, axis=0)
    return pd.Series(avg_importances, index=feature_names)


def run_phase4_selection(df: pd.DataFrame, importance_quantile: float = 0.5) -> pd.DataFrame | None:
    logging.info(f"Phase 4 Input Shape: {df.shape}")
    split_result = _split_data_components(df)
    if split_result is None:
        logging.error("Phase 4: Failed to split data components. Aborting.")
        return None
    id_cols, target_cols, feature_cols, protected_cols = split_result
    vol_feature = PROTECTED_VOL_FEATURE
    X_df = df[feature_cols]
    y_df = df[target_cols]
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
            verbose=False,
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
        logging.error("Phase 4: No features were found to be stable across all regimes. Aborting.")
        return None
    logging.info(f"Phase 4: Kept {len(stable_features)} stable features.")
    logging.info(f"Phase 4: Dropped {dropped_count} unstable features.")
    X_selected_df = X_df[stable_features]
    final_df = pd.concat([df[id_cols], y_df, df[protected_cols], X_selected_df], axis=1)
    logging.info(f"Phase 4: Final shape: {final_df.shape}.")
    return final_df