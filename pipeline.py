# file: ai/pipeline.py

import logging
import os
import pandas as pd
from datetime import timedelta
from app_config import get_query_parameters
from config import get_db_connection
from repository import get_candles_data
from data_exporter import create_raw_features
from feature_selector import (
    run_phase1_selection,
    run_phase2_selection,
    run_phase3_selection,
    run_phase4_selection
)

MAX_INDICATOR_PERIOD = 200


def _save_column_list_to_txt(df: pd.DataFrame, parquet_file_path: str, instrument: str):
    """
    Saves the DataFrame's column list to a .txt file.
    """
    column_file_path = parquet_file_path.replace(".parquet", "_columns.txt")
    columns_list = df.columns.tolist()
    try:
        with open(column_file_path, 'w') as f:
            for column_name in columns_list:
                f.write(f"{column_name}\n")
        logging.info(f"({instrument}) Successfully saved column list ({len(columns_list)} columns) to {column_file_path}")
    except Exception as e:
        logging.warning(f"({instrument}) Could not save column list to {column_file_path}: {e}")


def _define_file_paths(params: dict, output_dir: str) -> dict:
    """Creates a dictionary of all file paths for the pipeline."""
    base_filename = (
        f"{params['instrument']}_{params['timeframe']}_"
        f"{params['start_date_str']}_to_{params['end_date_str']}"
    )
    paths = {
        "base": os.path.join(output_dir, f"{base_filename}_base.parquet"),
        "phase_1": os.path.join(output_dir, f"{base_filename}_phase_1.parquet"),
        "phase_2": os.path.join(output_dir, f"{base_filename}_phase_2.parquet"),
        "phase_3": os.path.join(output_dir, f"{base_filename}_phase_3.parquet"),
        "phase_4": os.path.join(output_dir, f"{base_filename}_phase_4.parquet"),
        "shap_scores": os.path.join(output_dir, f"{base_filename}_phase_3_shap_scores.csv")
    }
    return paths


def _get_timedelta_for_timeframe(timeframe: str) -> timedelta:
    """Returns the timedelta duration for one timeframe unit."""
    if timeframe.startswith('m'):
        try:
            minutes = int(timeframe[1:])
            return timedelta(minutes=minutes)
        except ValueError:
            logging.warning(f"Unknown timeframe {timeframe}, defaulting to m1.")
            return timedelta(minutes=1)
    elif timeframe.startswith('h'):
        try:
            hours = int(timeframe[1:])
            return timedelta(hours=hours)
        except ValueError:
            logging.warning(f"Unknown timeframe {timeframe}, defaulting to h1.")
            return timedelta(hours=1)
    elif timeframe.startswith('D'):
        try:
            days = int(timeframe[1:])
            return timedelta(days=days)
        except ValueError:
            logging.warning(f"Unknown timeframe {timeframe}, defaulting to D1.")
            return timedelta(days=1)

    logging.warning(f"Undefined timeframe {timeframe}, defaulting to m1.")
    return timedelta(minutes=1)


def _generate_base_data_full(params: dict, base_parquet_file: str) -> str:
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

        _save_column_list_to_txt(df=df_base_dirty, parquet_file_path=base_parquet_file, instrument=instrument)

        return f"SUCCESS ({instrument}): Base file generated ({total_rows_processed} rows)."

    except Exception as e:
        logging.exception(f"({instrument}) An error occurred during full data generation: {e}")
        return f"FAILED ({instrument}): Full generator error. {e}"
    finally:
        if client:
            client.disconnect()
            logging.info(f"({instrument}) ClickHouse connection closed.")


def _run_selection_phase(
        phase_name: str,
        input_df: pd.DataFrame,
        output_file: str,
        selection_func,
        instrument: str,
        **kwargs
) -> pd.DataFrame | None:
    try:
        logging.info(f"({instrument}) --- Starting Feature Selection: {phase_name} ---")
        df_out = selection_func(input_df, **kwargs)
        if df_out is None or df_out.empty:
            logging.warning(f"({instrument}) {phase_name} selection resulted in an empty or None DataFrame. No file saved.")
            return None
        logging.info(f"({instrument}) Saving {phase_name} file with shape: {df_out.shape}")
        df_out.to_parquet(output_file, index=False)
        logging.info(f"({instrument}) {phase_name} selected features saved to {output_file}")
        _save_column_list_to_txt(df=df_out, parquet_file_path=output_file, instrument=instrument)
        logging.info(f"({instrument}) --- Finished Feature Selection: {phase_name} ---")
        return df_out
    except Exception as e:
        logging.exception(f"({instrument}) An unexpected error occurred during {phase_name} selection: {e}")
        return None


def run_pipeline_for_instrument(output_dir: str) -> str:
    """
    Runs the complete 4-stage feature selection pipeline.
    It will intelligently generate the base file via streaming if it's missing,
    or process the existing base file if it's found.
    """
    params = get_query_parameters()
    instrument = params['instrument']
    logging.info(f"========== STARTING PIPELINE (SELECTOR) FOR: {instrument} ==========")

    df_base, df_phase1, df_phase2, df_phase3 = None, None, None, None
    base_filename_str = f"{instrument}_UNKNOWN"

    try:
        # 1. Load Configuration
        logging.info(f"({instrument}) Loading query parameters")
        if not params:
            return f"FAILED ({instrument}): Could not load query parameters."

        # 2. Define File Names
        files = _define_file_paths(params, output_dir)
        base_filename_str = files['base'].replace("_base.parquet", "")

        # 3. Check Phase 4 (Final)
        if os.path.exists(files['phase_4']):
            logging.info(f"({instrument}) Final file '{files['phase_4']}' already exists.")
            return f"SKIPPED ({instrument}): Final file already exists."

        # 4. Check/Get Phase 3
        if os.path.exists(files['phase_3']):
            logging.info(f"({instrument}) Loading existing Phase 3 file: {files['phase_3']}")
            df_phase3 = pd.read_parquet(files['phase_3'])

        if df_phase3 is None:
            # 5. Check/Get Phase 2
            if os.path.exists(files['phase_2']):
                logging.info(f"({instrument}) Loading existing Phase 2 file: {files['phase_2']}")
                df_phase2 = pd.read_parquet(files['phase_2'])

            if df_phase2 is None:
                # 6. Check/Get Phase 1
                if os.path.exists(files['phase_1']):
                    logging.info(f"({instrument}) Loading existing Phase 1 file: {files['phase_1']}")
                    df_phase1 = pd.read_parquet(files['phase_1'])

                if df_phase1 is None:
                    # 7. Check/Get Base Data
                    logging.info(f"({instrument}) Checking for base feature file: {files['base']}")
                    if not os.path.exists(files['base']):
                        logging.warning(f"({instrument}) Base file not found. Starting full (non-streaming) generator...")
                        status = _generate_base_data_full(params, files['base'])
                        if not status.startswith("SUCCESS"):
                            return status

                        logging.info(f"({instrument}) Base file generation complete. Automatically proceeding to load and clean...")

                    try:
                        logging.info(f"({instrument}) Loading DIRTY base feature file: {files['base']}")
                        df_base = pd.read_parquet(files['base'])
                        logging.info(f"({instrument}) Successfully loaded DIRTY base file with shape: {df_base.shape}")
                    except Exception as e:
                        logging.error(f"({instrument}) Failed to read base file {files['base']}: {e}")
                        logging.error(
                            f"({instrument}) File might be corrupt. Try deleting it and re-running to regenerate.")
                        return f"FAILED ({instrument}): Could not read existing base file."

                    logging.info(f"({instrument}) Creating 1-block-future target variables (y)")
                    df_base['target_open_future_1'] = df_base['open'].shift(-1)
                    df_base['target_high_future_1'] = df_base['high'].shift(-1)
                    df_base['target_low_future_1'] = df_base['low'].shift(-1)
                    df_base['target_close_future_1'] = df_base['close'].shift(-1)

                    logging.info(f"({instrument}) Base file loaded. Cleaning NaNs from indicators and targets...")
                    id_cols = ['timestamp', 'instrument', 'timeframe']
                    cols_to_clean = [col for col in df_base.columns if col not in id_cols]

                    initial_rows = len(df_base)
                    df_base = df_base.dropna(subset=cols_to_clean)
                    final_rows = len(df_base)

                    logging.info(
                        f"({instrument}) NaNs cleaned. Rows removed: {initial_rows - final_rows}. Final base shape: {df_base.shape}")

                    if df_base.empty:
                        logging.error(f"({instrument}) No data left after cleaning NaNs from the full base file. Aborting.")
                        return f"FAILED ({instrument}): No data left after full base file NaN cleaning."

                    # 8. Run Phase 1
                    df_phase1 = _run_selection_phase("Phase 1", df_base, files['phase_1'], run_phase1_selection, instrument)
                    del df_base
                    if df_phase1 is None:
                        return f"FAILED ({instrument}): Phase 1 returned no data."

                # 9. Run Phase 2
                df_phase2 = _run_selection_phase("Phase 2", df_phase1, files['phase_2'], run_phase2_selection, instrument, corr_threshold=0.95)
                del df_phase1
                if df_phase2 is None:
                    return f"FAILED ({instrument}): Phase 2 returned no data."

            # 10. Run Phase 3
            df_phase3 = _run_selection_phase(
                "Phase 3",
                df_phase2,
                files['phase_3'],
                run_phase3_selection,
                instrument,
                n_trials=20,
                shap_scores_path=files['shap_scores']
            )
            del df_phase2
            if df_phase3 is None:
                return f"FAILED ({instrument}): Phase 3 returned no data."

        # 11. Run Phase 4
        df_phase4 = _run_selection_phase("Phase 4", df_phase3, files['phase_4'], run_phase4_selection, instrument, importance_quantile=0.5)
        del df_phase3

        if df_phase4 is None:
            return f"WARNING ({instrument}): Phase 4 failed or returned empty data."

        status_message = f"SUCCESS ({instrument}): Pipeline complete. Final file: {files['phase_4']}"

    except Exception as e:
        logging.exception(f"An unexpected error occurred in pipeline for {instrument}: {e}")
        status_message = f"CRITICAL FAILURE ({base_filename_str}): Pipeline crashed. Error: {e}"

    logging.info(f"========== PIPELINE FINISHED FOR: {instrument} ==========")
    return status_message