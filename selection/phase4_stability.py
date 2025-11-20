# file: geonera-ai/selection/phase4_stability.py

import logging
import pandas as pd
import catboost as cb
from sklearn.multioutput import MultiOutputRegressor
from tqdm import tqdm

from .common import (
    _split_data_components,
    _get_volatility_regime,
    _get_aggregated_importances
)


def _prune_correlated_features(df: pd.DataFrame, feature_list: list, votes_dict: dict, threshold: float = 0.98) -> list:
    if not feature_list or len(feature_list) < 2:
        return feature_list

    logging.info(f"Phase 4.5: Checking correlation redundancy for {len(feature_list)} winning features...")
    features_df = df[feature_list]
    corr_matrix = features_df.corr().abs()

    to_drop = set()

    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            col_i = corr_matrix.columns[i]
            col_j = corr_matrix.columns[j]

            if col_i in to_drop or col_j in to_drop:
                continue

            if corr_matrix.iloc[i, j] > threshold:
                vote_i = votes_dict.get(col_i, 0)
                vote_j = votes_dict.get(col_j, 0)

                if vote_i < vote_j:
                    to_drop.add(col_i)
                else:
                    to_drop.add(col_j)

    final_features = [f for f in feature_list if f not in to_drop]
    logging.info(f"Phase 4.5: Removed {len(to_drop)} redundant features. Kept {len(final_features)} unique strong features.")
    return final_features


def run_phase4_selection(df: pd.DataFrame, importance_quantile: float = 0.5) -> pd.DataFrame | None:
    logging.info(f"Phase 4 (Ensemble Stability Loop) Input Shape: {df.shape}")

    split_result = _split_data_components(df)
    if split_result is None:
        return None
    id_cols, target_cols, feature_cols, protected_cols = split_result

    X_df = df[feature_cols]
    y_df = df[target_cols]

    if X_df.empty:
        logging.warning("Phase 4: No features found to run stability test on.")
        return df

    START_ATR = 5
    END_ATR = 200
    STEP = 1
    atr_candidates = [f"atr_{i}" for i in range(START_ATR, END_ATR + 1, STEP)]
    valid_atr_cols = [col for col in atr_candidates if col in df.columns]

    if not valid_atr_cols:
        logging.error("Phase 4: No 'atr_XX' columns found. Aborting.")
        return None

    feature_votes = {feat: 0 for feat in feature_cols}
    for atr in valid_atr_cols:
        feature_votes[atr] = 0

    total_iterations = 0
    for atr_col in tqdm(valid_atr_cols, desc="ATR Stability Loop"):
        try:
            current_regimes = _get_volatility_regime(df[atr_col])
        except (ValueError, TypeError, KeyError) as e:
            logging.warning(f"Phase 4: Skipped regime calc for {atr_col}: {e}")
            continue
        except Exception as e:
            logging.error(f"Phase 4: Unexpected error in regime calc for {atr_col}: {e}")
            continue

        regimes = ['LOW_VOL', 'MID_VOL', 'HIGH_VOL']
        regime_importances = {}
        valid_iteration = True

        for regime in regimes:
            regime_mask = (current_regimes == regime)
            X_regime = X_df.loc[regime_mask]
            y_regime = y_df.loc[regime_mask]
            X_regime_with_atr = X_regime.copy()
            if atr_col not in X_regime_with_atr.columns and atr_col in df.columns:
                X_regime_with_atr[atr_col] = df.loc[regime_mask, atr_col]

            if len(X_regime) < 50:
                valid_iteration = False
                break

            core_model = cb.CatBoostRegressor(
                iterations=100,
                depth=4,
                learning_rate=0.1,
                verbose=False,
                random_state=42,
                task_type='GPU',
                allow_writing_files=False
            )
            multi_model = MultiOutputRegressor(core_model)
            try:
                multi_model.fit(X_regime_with_atr, y_regime)
                avg_imp = _get_aggregated_importances(multi_model, X_regime_with_atr.columns.tolist())
                regime_importances[regime] = avg_imp
            except Exception as e:
                logging.warning(f"Phase 4: Training failed for {atr_col} ({regime}). Error: {e}")
                valid_iteration = False
                break

        if not valid_iteration or len(regime_importances) < 3:
            continue

        stability_df = pd.DataFrame(regime_importances)
        quantiles = stability_df.quantile(importance_quantile)
        stable_mask = pd.Series(True, index=stability_df.index)
        for r in regimes:
            stable_mask = stable_mask & (stability_df[r] > quantiles[r])

        stable_feats = stability_df[stable_mask].index.tolist()

        for f in stable_feats:
            if f in feature_votes:
                feature_votes[f] += 1

        total_iterations += 1

    if total_iterations == 0:
        logging.error("Phase 4: No valid iterations completed. Aborting.")
        return None

    CONSENSUS_THRESHOLD = 0.4
    min_votes = int(total_iterations * CONSENSUS_THRESHOLD)

    logging.info(f"Phase 4: Loop complete. Total iterations: {total_iterations}. Vote threshold: {min_votes}")
    voting_winners = [f for f, v in feature_votes.items() if v >= min_votes]

    atr_winners = [f for f in voting_winners if f.startswith('atr_')]
    other_winners = [f for f in voting_winners if not f.startswith('atr_')]

    logging.info(f"Phase 4: Voting complete. Candidates: {len(atr_winners)} ATRs, {len(other_winners)} others.")

    final_atrs = _prune_correlated_features(df, atr_winners, feature_votes, threshold=0.95)
    final_others = other_winners

    final_features = final_atrs + final_others

    logging.info(f"Phase 4 Final: Selected {len(final_atrs)} Best ATRs: {final_atrs}")

    core_protected = [c for c in protected_cols if not c.startswith('atr_')]
    cols_to_concat = [
        df[id_cols],
        y_df,
        df[core_protected],
        df[final_features]
    ]

    final_df = pd.concat(cols_to_concat, axis=1)
    final_df = final_df.loc[:, ~final_df.columns.duplicated()]

    logging.info(f"Phase 4: Final shape: {final_df.shape}.")
    return final_df