# file: ai/data_exporter.py

import pandas as pd
import logging
import time

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
    Converts raw candle data into a massive feature DataFrame.
    NOTE: This function returns a "dirty" DataFrame with NaNs
    (in the indicator warmup period and targets)
    The caller (e.g., pipeline.py) is responsible for filtering and dropping NaNs.
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
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['time_minute_of_hour'] = df['timestamp'].dt.minute.astype(float)
        df['time_hour_of_day'] = df['timestamp'].dt.hour.astype(float)
        df['time_day_of_week'] = df['timestamp'].dt.dayofweek.astype(float)

        logging.info("Successfully created time-based features (minute, hour, day_of_week).")
        decimal_cols = [
            'open', 'high', 'low', 'close',
            'min_spread', 'max_spread', 'avg_spread', 'vwap',
            'total_bid_volume', 'total_ask_volume'
        ]
        for col in decimal_cols:
            if col in df.columns:
                df[col] = df[col].astype(float)

        logging.info(f"Converted {len(decimal_cols)} Decimal columns to float64.")
        logging.info("Starting all indicator calculations...")
        total_start_time = time.perf_counter()

        start_time = time.perf_counter()
        sma_df = add_sma_calculations(df)
        logging.info(f"-> SMA Calculations finished. Duration: {time.perf_counter() - start_time:.2f} seconds.")

        start_time = time.perf_counter()
        ema_df = add_ema_calculations(df)
        logging.info(f"-> EMA Calculations finished. Duration: {time.perf_counter() - start_time:.2f} seconds.")

        start_time = time.perf_counter()
        wma_df = add_wma_calculations(df)
        logging.info(f"-> WMA Calculations finished. Duration: {time.perf_counter() - start_time:.2f} seconds.")

        start_time = time.perf_counter()
        vwma_df = add_vwma_calculations(df)
        logging.info(f"-> VWMA Calculations finished. Duration: {time.perf_counter() - start_time:.2f} seconds.")

        start_time = time.perf_counter()
        hma_df = add_hma_calculations(df)
        logging.info(f"-> HMA Calculations finished. Duration: {time.perf_counter() - start_time:.2f} seconds.")

        start_time = time.perf_counter()
        smma_df = add_smma_calculations(df)
        logging.info(f"-> SMMA Calculations finished. Duration: {time.perf_counter() - start_time:.2f} seconds.")

        start_time = time.perf_counter()
        dema_df = add_dema_calculations(df)
        logging.info(f"-> DEMA Calculations finished. Duration: {time.perf_counter() - start_time:.2f} seconds.")

        start_time = time.perf_counter()
        tema_df = add_tema_calculations(df)
        logging.info(f"-> TEMA Calculations finished. Duration: {time.perf_counter() - start_time:.2f} seconds.")

        start_time = time.perf_counter()
        rsi_df = add_rsi_calculations(df)
        logging.info(f"-> RSI Calculations finished. Duration: {time.perf_counter() - start_time:.2f} seconds.")

        start_time = time.perf_counter()
        cci_df = add_cci_calculations(df)
        logging.info(f"-> CCI Calculations finished. Duration: {time.perf_counter() - start_time:.2f} seconds.")

        start_time = time.perf_counter()
        mom_df = add_momentum_calculations(df)
        logging.info(f"-> Momentum Calculations finished. Duration: {time.perf_counter() - start_time:.2f} seconds.")

        start_time = time.perf_counter()
        roc_df = add_roc_calculations(df)
        logging.info(f"-> ROC Calculations finished. Duration: {time.perf_counter() - start_time:.2f} seconds.")

        start_time = time.perf_counter()
        wpr_df = add_wpr_calculations(df)
        logging.info(f"-> WPR Calculations finished. Duration: {time.perf_counter() - start_time:.2f} seconds.")

        start_time = time.perf_counter()
        atr_df = add_atr_calculations(df)
        logging.info(f"-> ATR Calculations finished. Duration: {time.perf_counter() - start_time:.2f} seconds.")

        start_time = time.perf_counter()
        adx_df = add_adx_calculations(df)
        logging.info(f"-> ADX Calculations finished. Duration: {time.perf_counter() - start_time:.2f} seconds.")

        start_time = time.perf_counter()
        trix_df = add_trix_calculations(df)
        logging.info(f"-> TRIX Calculations finished. Duration: {time.perf_counter() - start_time:.2f} seconds.")

        start_time = time.perf_counter()
        sm_rsi_df = add_smoothed_rsi_calculations(df)
        logging.info(f"-> Smoothed RSI Calculations finished. Duration: {time.perf_counter() - start_time:.2f} seconds.")

        start_time = time.perf_counter()
        dc_df = add_donchian_channel_calculations(df)
        logging.info(f"-> Donchian Channel Calculations finished. Duration: {time.perf_counter() - start_time:.2f} seconds.")

        start_time = time.perf_counter()
        env_df = add_envelope_calculations(df)
        logging.info(f"-> Envelope Calculations finished. Duration: {time.perf_counter() - start_time:.2f} seconds.")

        start_time = time.perf_counter()
        fcb_df = add_fractal_chaos_bands_calculations(df)
        logging.info(f"-> Fractal Chaos Bands Calculations finished. Duration: {time.perf_counter() - start_time:.2f} seconds.")

        start_time = time.perf_counter()
        obv_df = add_obv_calculations(df)
        logging.info(f"-> OBV Calculations finished. Duration: {time.perf_counter() - start_time:.2f} seconds.")

        start_time = time.perf_counter()
        cmf_df = add_chaikin_money_flow_calculations(df)
        logging.info(f"-> Chaikin Money Flow Calculations finished. Duration: {time.perf_counter() - start_time:.2f} seconds.")

        start_time = time.perf_counter()
        fi_df = add_force_index_calculations(df)
        logging.info(f"-> Force Index Calculations finished. Duration: {time.perf_counter() - start_time:.2f} seconds.")

        start_time = time.perf_counter()
        mfi_df = add_money_flow_index_calculations(df)
        logging.info(f"-> Money Flow Index Calculations finished. Duration: {time.perf_counter() - start_time:.2f} seconds.")

        start_time = time.perf_counter()
        vi_df = add_vortex_indicator_calculations(df)
        logging.info(f"-> Vortex Indicator Calculations finished. Duration: {time.perf_counter() - start_time:.2f} seconds.")

        start_time = time.perf_counter()
        aroon_df = add_aroon_indicator_calculations(df)
        logging.info(f"-> Aroon Indicator Calculations finished. Duration: {time.perf_counter() - start_time:.2f} seconds.")

        start_time = time.perf_counter()
        cmo_df = add_chande_momentum_oscillator_calculations(df)
        logging.info(f"-> Chande Momentum Oscillator Calculations finished. Duration: {time.perf_counter() - start_time:.2f} seconds.")

        start_time = time.perf_counter()
        std_df = add_standard_deviation_calculations(df)
        logging.info(f"-> Standard Deviation Calculations finished. Duration: {time.perf_counter() - start_time:.2f} seconds.")

        start_time = time.perf_counter()
        var_df = add_variance_calculations(df)
        logging.info(f"-> Variance Calculations finished. Duration: {time.perf_counter() - start_time:.2f} seconds.")

        start_time = time.perf_counter()
        med_df = add_median_filter_calculations(df)
        logging.info(f"-> Median Filter Calculations finished. Duration: {time.perf_counter() - start_time:.2f} seconds.")

        total_end_time = time.perf_counter()
        logging.info(f"All indicator calculations complete. Total duration: {total_end_time - total_start_time:.2f} seconds.")

        logging.info("Concatenating all indicator DataFrames")
        df = pd.concat([
            df, sma_df, ema_df, wma_df, vwma_df, hma_df, smma_df, dema_df, tema_df,
            rsi_df, cci_df, mom_df, roc_df, wpr_df, atr_df, adx_df, trix_df, sm_rsi_df,
            dc_df, env_df, fcb_df,
            obv_df, cmf_df, fi_df, mfi_df,
            vi_df, aroon_df, cmo_df,
            std_df, var_df, med_df
        ], axis=1)

        logging.info(f"Raw feature DataFrame created with shape: {df.shape}")
        return df

    except (ValueError, TypeError) as e:
        logging.error(f"Data-related error (e.g., type conversion, calculation mismatch) during export: {e}")
        return None
    except Exception as e:
        logging.exception(f"An unexpected error occurred in create_raw_features: {e}")
        return None