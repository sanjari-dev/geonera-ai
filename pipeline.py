# pipeline.py

import logging
import os
import pandas as pd
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


def _generate_base_data(file_name: str, params: dict) -> pd.DataFrame | None:
    logging.info(f"({params['instrument']}) No valid base file found. Generating from database")
    client = None
    try:
        client = get_db_connection()
        if not client:
            logging.error(f"({params['instrument']}) Failed to connect to database.")
            return None
        candles = get_candles_data(
            client=client,
            instrument=params['instrument'],
            timeframe=params['timeframe'],
            start_time=params['start_time'],
            end_time=params['end_time']
        )
        if not candles:
            logging.warning(f"({params['instrument']}) No candle data retrieved from database.")
            return None
        logging.info(f"({params['instrument']}) Successfully retrieved {len(candles)} candle records.")
        df_raw = create_raw_features(candles)
        if df_raw is None or df_raw.empty:
            logging.error(f"({params['instrument']}) Failed to create raw features or DataFrame is empty.")
            return None
        logging.info(f"({params['instrument']}) Saving base file with shape: {df_raw.shape}")
        df_raw.to_parquet(file_name, index=False)
        logging.info(f"({params['instrument']}) Raw feature (base) file saved to {file_name}")
        return df_raw
    except Exception as e:
        logging.exception(f"({params['instrument']}) An unexpected error occurred during data generation: {e}")
        return None
    finally:
        if client:
            client.disconnect()
            logging.info(f"({params['instrument']}) Connection to ClickHouse closed.")


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
        logging.info(f"({instrument}) --- Finished Feature Selection: {phase_name} ---")
        return df_out
    except Exception as e:
        logging.exception(f"({instrument}) An unexpected error occurred during {phase_name} selection: {e}")
        return None


def run_pipeline_for_instrument(output_dir: str) -> str:
    """
    Runs the complete 4-stage feature selection pipeline for a single instrument.
    Returns a status message.
    """
    params = get_query_parameters()
    instrument = params['instrument']
    logging.info(f"========== STARTING PIPELINE FOR: {instrument} ==========")

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
                    if os.path.exists(files['base']):
                        logging.info(f"({instrument}) Loading existing base feature file: {files['base']}")
                        df_base = pd.read_parquet(files['base'])

                    if df_base is None:
                        df_base = _generate_base_data(files['base'], params)
                        if df_base is None:
                            return f"FAILED ({instrument}): Could not generate base data."

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