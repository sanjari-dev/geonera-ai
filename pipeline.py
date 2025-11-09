# file: ai/pipeline.py

import logging
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
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

CHUNK_SIZE_DAYS = 90
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
        logging.info(
            f"({instrument}) Successfully saved column list ({len(columns_list)} columns) to {column_file_path}")
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

    logging.warning(f"Undefined timeframe {timeframe}, defaulting to m1.")
    return timedelta(minutes=1)


def _generate_base_data_streaming(params: dict, base_parquet_file: str) -> str:
    """
    Generates the base feature data using streaming (chunking)
    to avoid RAM exhaustion on large datasets.
    """
    logging.info(f"--- STARTING STREAMING BASE DATA GENERATOR ---")

    instrument = params['instrument']
    timeframe = params['timeframe']

    if os.path.exists(base_parquet_file):
        logging.warning(f"({instrument}) Old base file found. Deleting: {base_parquet_file}")
        os.remove(base_parquet_file)

    overlap_delta = timedelta(days=10)
    logging.info(f"({instrument}) Using overlap of {overlap_delta} to cover indicator warmup.")

    client = None
    pq_writer = None
    schema = None

    try:
        client = get_db_connection()
        if not client:
            return f"FAILED ({instrument}): Could not connect to ClickHouse."

        current_chunk_start_time = params['start_time']
        total_rows_processed = 0

        logging.info(f"({instrument}) Starting streaming process from {params['start_time']} to {params['end_time']}")

        while current_chunk_start_time < params['end_time']:
            current_chunk_end_time = current_chunk_start_time + timedelta(days=CHUNK_SIZE_DAYS)
            if current_chunk_end_time > params['end_time']:
                current_chunk_end_time = params['end_time']

            query_start_time = current_chunk_start_time - overlap_delta
            query_end_time = current_chunk_end_time

            logging.info(f"({instrument}) Processing chunk: {current_chunk_start_time} to {current_chunk_end_time}")
            logging.info(f"   -> Fetching data (including overlap): {query_start_time} to {query_end_time}")

            candles = get_candles_data(
                client=client,
                instrument=instrument,
                timeframe=timeframe,
                start_time=query_start_time,
                end_time=query_end_time
            )
            if not candles:
                logging.warning(f"({instrument}) No data found for range {current_chunk_start_time}. Skipping.")
                current_chunk_start_time = current_chunk_end_time
                continue

            df_chunk_dirty = create_raw_features(candles)

            if df_chunk_dirty is None or df_chunk_dirty.empty:
                logging.error(f"({instrument}) create_raw_features failed or returned empty data for this chunk.")
                current_chunk_start_time = current_chunk_end_time
                continue

            # noinspection PyTypeChecker
            df_chunk_filtered = df_chunk_dirty[df_chunk_dirty['timestamp'] >= current_chunk_start_time].copy()
            df_chunk_filtered: pd.DataFrame = df_chunk_filtered

            if df_chunk_filtered.empty:
                logging.warning(
                    f"({instrument}) No data left after timestamp filtering in chunk {current_chunk_start_time}.")
                current_chunk_start_time = current_chunk_end_time
                continue

            if pq_writer is None:
                # noinspection PyArgumentList
                table = pa.Table.from_pandas(df_chunk_filtered)
                schema = table.schema
                pq_writer = pq.ParquetWriter(base_parquet_file, schema)
                pq_writer.write_table(table)
            else:
                # noinspection PyArgumentList
                table = pa.Table.from_pandas(df_chunk_filtered, schema=schema)
                pq_writer.write_table(table)

            rows_in_chunk = len(df_chunk_filtered)
            total_rows_processed += rows_in_chunk
            logging.info(f"   -> Chunk processed. {rows_in_chunk} (uncleaned) rows saved. Total: {total_rows_processed} rows.")
            current_chunk_start_time = current_chunk_end_time

        logging.info(f"--- STREAMING COMPLETE. File saved to: {base_parquet_file} ---")

        if schema:
            df_schema = pd.DataFrame(columns=schema.names)
            _save_column_list_to_txt(df=df_schema, parquet_file_path=base_parquet_file, instrument=instrument)

        if total_rows_processed == 0:
            logging.error(f"({instrument}) Streaming finished but 0 total rows were processed.")
            return f"FAILED ({instrument}): 0 rows processed. Check data source and timeframe."

        return f"SUCCESS ({instrument}): Base file generated ({total_rows_processed} rows). Please re-run to start Phase 1."

    except Exception as e:
        logging.exception(f"({instrument}) An error occurred during streaming generation: {e}")
        return f"FAILED ({instrument}): Streaming generator error. {e}"
    finally:
        if pq_writer:
            pq_writer.close()
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
                        logging.warning(f"({instrument}) Base file not found. Starting streaming generator...")
                        status = _generate_base_data_streaming(params, files['base'])
                        return status

                    try:
                        logging.info(f"({instrument}) Loading existing base feature file: {files['base']}")
                        df_base = pd.read_parquet(files['base'])
                        logging.info(f"({instrument}) Successfully loaded base file with shape: {df_base.shape}")
                    except Exception as e:
                        logging.error(f"({instrument}) Failed to read base file {files['base']}: {e}")
                        logging.error(f"({instrument}) File might be corrupt. Try deleting it and re-running to regenerate.")
                        return f"FAILED ({instrument}): Could not read existing base file."

                    logging.info(f"({instrument}) Base file loaded. Cleaning NaNs from indicators and targets...")
                    id_cols = ['timestamp', 'instrument', 'timeframe']
                    cols_to_clean = [col for col in df_base.columns if col not in id_cols]

                    initial_rows = len(df_base)
                    df_base = df_base.dropna(subset=cols_to_clean)
                    final_rows = len(df_base)

                    logging.info(f"({instrument}) NaNs cleaned. Rows removed: {initial_rows - final_rows}. Final base shape: {df_base.shape}")

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