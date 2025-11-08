# data_exporter.py

import pandas as pd
import logging

from indicators import (
    add_sma_calculations, add_ema_calculations, add_wma_calculations,
    add_vwma_calculations, add_hma_calculations, add_smma_calculations,
    add_dema_calculations, add_tema_calculations, add_rsi_calculations,
    add_cci_calculations, add_momentum_calculations, add_roc_calculations,
    add_wpr_calculations, add_atr_calculations, add_adx_calculations,
    add_trix_calculations, add_smoothed_rsi_calculations,
    add_donchian_channel_calculations, add_envelope_calculations,
    add_fractal_chaos_bands_calculations, add_obv_calculations,
    add_chaikin_money_flow_calculations, add_force_index_calculations,
    add_money_flow_index_calculations, add_vortex_indicator_calculations,
    add_aroon_indicator_calculations, add_chande_momentum_oscillator_calculations,
    add_standard_deviation_calculations, add_variance_calculations,
    add_median_filter_calculations
)


def create_raw_features(candles_data: list) -> pd.DataFrame | None:
    """
    Converts raw candle data into a massive, CLEANED feature DataFrame.
    1. Converts types
    2. Calculates all 30+ indicator sets
    3. Creates 1-block-future targets
    4. Cleans all NaN rows
    Returns a single, clean DataFrame.
    """
    try:
        column_names = [
            'timestamp', 'instrument', 'timeframe',
            'open', 'high', 'low', 'close',
            'tick_count', 'min_spread', 'max_spread', 'avg_spread',
            'total_bid_volume', 'total_ask_volume',
            'vwap'
        ]
        df = pd.DataFrame(candles_data, columns=column_names)
        decimal_cols = [
            'open', 'high', 'low', 'close',
            'min_spread', 'max_spread', 'avg_spread', 'vwap',
            'total_bid_volume', 'total_ask_volume'
        ]
        for col in decimal_cols:
            if col in df.columns:
                df[col] = df[col].astype(float)

        logging.info(f"Converted {len(decimal_cols)} Decimal columns to float64.")
        logging.info("Starting all indicator calculations")

        sma_df = add_sma_calculations(df)
        ema_df = add_ema_calculations(df)
        wma_df = add_wma_calculations(df)
        vwma_df = add_vwma_calculations(df)
        hma_df = add_hma_calculations(df)
        smma_df = add_smma_calculations(df)
        dema_df = add_dema_calculations(df)
        tema_df = add_tema_calculations(df)
        rsi_df = add_rsi_calculations(df)
        cci_df = add_cci_calculations(df)
        mom_df = add_momentum_calculations(df)
        roc_df = add_roc_calculations(df)
        wpr_df = add_wpr_calculations(df)
        atr_df = add_atr_calculations(df)
        adx_df = add_adx_calculations(df)
        trix_df = add_trix_calculations(df)
        sm_rsi_df = add_smoothed_rsi_calculations(df)
        dc_df = add_donchian_channel_calculations(df)
        env_df = add_envelope_calculations(df)
        fcb_df = add_fractal_chaos_bands_calculations(df)
        obv_df = add_obv_calculations(df)
        cmf_df = add_chaikin_money_flow_calculations(df)
        fi_df = add_force_index_calculations(df)
        mfi_df = add_money_flow_index_calculations(df)
        vi_df = add_vortex_indicator_calculations(df)
        aroon_df = add_aroon_indicator_calculations(df)
        cmo_df = add_chande_momentum_oscillator_calculations(df)
        std_df = add_standard_deviation_calculations(df)
        var_df = add_variance_calculations(df)
        med_df = add_median_filter_calculations(df)

        logging.info("Concatenating all indicator DataFrames")
        df = pd.concat([
            df, sma_df, ema_df, wma_df, vwma_df, hma_df, smma_df, dema_df, tema_df,
            rsi_df, cci_df, mom_df, roc_df, wpr_df, atr_df, adx_df, trix_df, sm_rsi_df,
            dc_df, env_df, fcb_df,
            obv_df, cmf_df, fi_df, mfi_df,
            vi_df, aroon_df, cmo_df,
            std_df, var_df, med_df
        ], axis=1)

        logging.info("Creating 1-block-future target variables (y)")
        df['target_open_future_1'] = df['open'].shift(-1)
        df['target_high_future_1'] = df['high'].shift(-1)
        df['target_low_future_1'] = df['low'].shift(-1)
        df['target_close_future_1'] = df['close'].shift(-1)

        logging.info(f"Raw feature DataFrame created with shape: {df.shape}")

        logging.info(f"Original size before NaN cleaning: {len(df)} rows")

        id_cols = ['timestamp', 'instrument', 'timeframe']
        cols_to_clean = [col for col in df.columns if col not in id_cols]

        df = df.dropna(subset=cols_to_clean)
        logging.info(f"Base file clean size after dropping NaN rows: {len(df)} rows")

        if df.empty:
            logging.warning("No data remaining after NaN cleaning. Returning empty DataFrame.")

        return df

    except (ValueError, TypeError) as e:
        logging.error(f"Data-related error (e.g., type conversion, calculation mismatch) during export: {e}")
        return None
    except Exception as e:
        logging.exception(f"An unexpected error occurred in create_raw_features: {e}")
        return None