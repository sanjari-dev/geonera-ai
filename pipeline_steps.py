# file: ai/pipeline_steps.py

import logging
import os
import pandas as pd
import utils as utils
from config import get_db_connection
from repository import get_candles_data
from data_exporter import create_raw_features


def generate_base_data_full(params: dict, base_parquet_file: str) -> str:
    """
    Generates the base feature data by loading ALL data into memory at once (non-streaming).
    SAVES "DIRTY" DATA (including NaNs from the initial warmup) directly to Parquet.
    """
    logging.info(f"--- STARTING FULL (NON-STREAMING) BASE DATA GENERATOR (DIRTY) ---")

    instrument = params['instrument']
    timeframe = params['timeframe']

    if os.path.exists(base_parquet_file):
        logging.warning(f"({instrument}) Old base file found. Deleting: {base_parquet_file}")
        os.remove(base_parquet_file)

    client = None
    try:
        client = get_db_connection()
        if not client:
            return f"FAILED ({instrument}): Could not connect to ClickHouse."

        logging.info(f"({instrument}) Fetching FULL data range: {params['start_time']} to {params['end_time']}")

        candles = get_candles_data(
            client=client,
            instrument=instrument,
            timeframe=timeframe,
            start_time=params['start_time'],
            end_time=params['end_time']
        )

        if not candles:
            logging.error(f"({instrument}) No data found for the entire range. Aborting.")
            return f"FAILED ({instrument}): 0 rows processed. Check data source and timeframe."

        logging.info(f"({instrument}) Data fetched. Found {len(candles)} rows. Creating raw features...")
        df_base_dirty = create_raw_features(candles)

        if df_base_dirty is None or df_base_dirty.empty:
            logging.error(f"({instrument}) create_raw_features failed or returned empty data for the full dataset.")
            return f"FAILED ({instrument}): create_raw_features returned empty data."

        logging.info(
            f"({instrument}) Features created. Saving (DIRTY) file with shape {df_base_dirty.shape} to {base_parquet_file}")

        df_base_dirty.to_parquet(base_parquet_file, index=False)

        total_rows_processed = len(df_base_dirty)
        logging.info(f"--- FULL LOAD COMPLETE. DIRTY file saved. Total: {total_rows_processed} rows. ---")

        utils.save_column_list(df=df_base_dirty, parquet_file_path=base_parquet_file, instrument=instrument)

        return f"SUCCESS ({instrument}): Base file generated ({total_rows_processed} rows)."

    except Exception as e:
        logging.exception(f"({instrument}) An error occurred during full data generation: {e}")
        return f"FAILED ({instrument}): Full generator error. {e}"
    finally:
        if client:
            client.disconnect()
            logging.info(f"({instrument}) ClickHouse connection closed.")


def clean_base_data(df_base: pd.DataFrame, instrument: str) -> pd.DataFrame | None:
    """
    Runs "Stage 0" cleaning: Creates targets and drops all NaNs
    (from indicator warm-up and target shifting).
    """
    try:
        logging.info(f"({instrument}) Creating 1-block-future target variables (y)")
        df_base['target_open_future_1'] = df_base['open'].shift(-1)
        df_base['target_high_future_1'] = df_base['high'].shift(-1)
        df_base['target_low_future_1'] = df_base['low'].shift(-1)
        df_base['target_close_future_1'] = df_base['close'].shift(-1)

        logging.info(f"({instrument}) Base file loaded. Cleaning NaNs from indicators and targets...")
        id_cols = ['timestamp', 'instrument', 'timeframe']

        # cols_to_clean will contain ALL feature and target columns
        cols_to_clean = [col for col in df_base.columns if col not in id_cols]

        initial_rows = len(df_base)
        df_base = df_base.dropna(subset=cols_to_clean)
        final_rows = len(df_base)

        logging.info(
            f"({instrument}) NaNs cleaned. Rows removed: {initial_rows - final_rows}. Final base shape: {df_base.shape}")

        if df_base.empty:
            logging.error(f"({instrument}) No data left after cleaning NaNs from the full base file. Aborting.")
            return None

        return df_base

    except Exception as e:
        logging.exception(f"({instrument}) An error occurred during 'Stage 0' data cleaning: {e}")
        return None