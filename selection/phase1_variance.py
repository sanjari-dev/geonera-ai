# file: ai/selection/phase1_variance.py

import logging
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from .common import _split_data_components


def run_phase1_selection(df: pd.DataFrame) -> pd.DataFrame | None:
    logging.info(f"Phase 1 Input Shape: {df.shape}")

    split_result = _split_data_components(df)
    if split_result is None:
        logging.error("Phase 1: Failed to split data components. Aborting.")
        return None

    id_cols, target_cols, feature_cols, protected_cols = split_result

    features_df = df[feature_cols]

    if not feature_cols:
        logging.error("Phase 1: No feature columns found to select from. Aborting.")
        return df  # Return original df as there is nothing to select

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

    logging.info("Phase 1: Re-combining ID, targets, protected, and selected features")
    final_df = pd.concat([df[id_cols], df[target_cols], df[protected_cols], features_df], axis=1)

    logging.info(f"Phase 1: Final shape: {final_df.shape}.")
    return final_df