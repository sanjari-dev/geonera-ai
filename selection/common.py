# file: geonera-ai/selection/common.py

import logging
import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputRegressor


PROTECTED_CORE_FEATURES = [
    'open', 'high', 'low', 'close', 'tick_count',
    'min_spread', 'max_spread', 'avg_spread',
    'total_bid_volume', 'total_ask_volume', 'vwap',
    'time_minute_of_hour', 'time_hour_of_day', 'time_day_of_week'
]


def _split_data_components(df: pd.DataFrame) -> tuple[list, list, list, list] | None:
    id_cols = ['timestamp', 'instrument', 'timeframe']
    target_cols = [col for col in df.columns if col.startswith('target_')]
    atr_cols = [col for col in df.columns if col.startswith('atr_')]
    protected_cols = PROTECTED_CORE_FEATURES.copy()
    protected_cols.extend(atr_cols)
    protected_cols = list(set(protected_cols))

    existing_protected_cols = [col for col in protected_cols if col in df.columns]

    feature_cols = [col for col in df.columns if col not in id_cols and col not in target_cols and col not in protected_cols]

    if not feature_cols:
        logging.error("No feature columns left after protecting core features.")

    logging.info(f"Split components: {len(id_cols)} IDs, {len(target_cols)} targets, {len(existing_protected_cols)} protected (incl. ATRs), {len(feature_cols)} features for selection.")
    return id_cols, target_cols, feature_cols, existing_protected_cols


def _get_volatility_regime(atr_series: pd.Series) -> pd.Series:
    low_thresh = atr_series.quantile(0.33)
    high_thresh = atr_series.quantile(0.66)
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