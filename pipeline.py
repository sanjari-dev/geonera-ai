# file: geonera-ai/pipeline.py

import logging
import os
import pandas as pd
from app_config import get_query_parameters
from utils import define_file_paths, save_column_list
from pipeline_steps import generate_base_data_full, clean_base_data
from selection import (
    run_phase1_selection,
    run_phase2_clustering,
    run_phase3_selection,
    run_phase4_selection
)


def _run_selection_phase(
        phase_name: str,
        input_df: pd.DataFrame,
        output_file: str,
        selection_func,
        instrument: str,
        **kwargs
) -> pd.DataFrame | None:
    """
    Helper function to run a selection phase, log, and save the results.
    """
    try:
        logging.info(f"({instrument}) --- Starting Feature Selection: {phase_name} ---")
        df_out = selection_func(input_df, **kwargs)
        if df_out is None or df_out.empty:
            logging.warning(f"({instrument}) {phase_name} selection resulted in an empty or None DataFrame. No file saved.")
            return None
        logging.info(f"({instrument}) Saving {phase_name} file with shape: {df_out.shape}")
        df_out.to_parquet(output_file, index=False)
        logging.info(f"({instrument}) {phase_name} selected features saved to {output_file}")
        save_column_list(df=df_out, parquet_file_path=output_file, instrument=instrument)
        logging.info(f"({instrument}) --- Finished Feature Selection: {phase_name} ---")
        return df_out
    except Exception as e:
        logging.exception(f"({instrument}) An unexpected error occurred during {phase_name} selection: {e}")
        return None


def run_pipeline_for_instrument(output_dir: str) -> str:
    """
    Runs the complete 4-stage feature selection pipeline.
    This is the main orchestrator function.
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
        files = define_file_paths(params, output_dir)
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
                        status = generate_base_data_full(params, files['base'])
                        if not status.startswith("SUCCESS"):
                            return status
                        logging.info(f"({instrument}) Base file generation complete. Automatically proceeding...")

                    try:
                        logging.info(f"({instrument}) Loading DIRTY base feature file: {files['base']}")
                        df_base = pd.read_parquet(files['base'])
                        logging.info(f"({instrument}) Successfully loaded DIRTY base file with shape: {df_base.shape}")
                    except Exception as e:
                        logging.error(f"({instrument}) Failed to read base file {files['base']}: {e}")
                        return f"FAILED ({instrument}): Could not read existing base file."

                    # 7a. Run "Stage 0" Cleaning
                    df_base_clean = clean_base_data(df_base, instrument)
                    del df_base  # Free memory

                    if df_base_clean is None:
                        return f"FAILED ({instrument}): No data left after full base file NaN cleaning."

                    # 8. Run Phase 1
                    df_phase1 = _run_selection_phase("Phase 1", df_base_clean, files['phase_1'], run_phase1_selection, instrument)
                    del df_base_clean  # Free memory
                    if df_phase1 is None:
                        return f"FAILED ({instrument}): Phase 1 returned no data."

                # 9. Run Phase 2
                df_phase2 = _run_selection_phase(
                    "Phase 2",
                    df_phase1,
                    files['phase_2'],
                    run_phase2_clustering,
                    instrument,
                    correlation_threshold=0.95
                )
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
                shap_scores_path=files['shap_scores'],
                n_top_features=750
            )
            del df_phase2
            if df_phase3 is None:
                return f"FAILED ({instrument}): Phase 3 returned no data."

        # 11. Run Phase 4 (Manual Call for Final Cleanup)
        logging.info(f"({instrument}) --- Starting Feature Selection: Phase 4 ---")
        try:
            df_phase4 = run_phase4_selection(
                df_phase3,
                importance_quantile=0.1
            )
            del df_phase3
        except Exception as e:
            logging.exception(f"({instrument}) An unexpected error occurred during Phase 4 selection: {e}")
            return f"FAILED ({instrument}): Phase 4 crashed."

        if df_phase4 is None or df_phase4.empty:
            logging.warning(f"({instrument}) Phase 4 selection resulted in an empty or None DataFrame. No file saved.")
            return f"WARNING ({instrument}): Phase 4 failed or returned."

        logging.info(f"({instrument}) --- Finished Feature Selection: Phase 4 ---")

        # 12. Final Cleanup
        logging.info(f"({instrument}) Cleaning final DataFrame. Removing temporary target and utility columns...")

        columns_to_drop = [
            'target_open_future_1',
            'target_high_future_1',
            'target_low_future_1',
            'target_close_future_1',
            'atr_20',
            'regime'
        ]

        existing_cols_to_drop = [col for col in columns_to_drop if col in df_phase4.columns]

        df_phase4 = df_phase4.drop(columns=existing_cols_to_drop)

        logging.info(f"({instrument}) Final cleanup complete. Final shape for training: {df_phase4.shape}")

        # 13. Manual Save of Final File
        try:
            logging.info(f"({instrument}) Saving final (Phase 4) file with shape: {df_phase4.shape}")
            df_phase4.to_parquet(files['phase_4'], index=False)
            logging.info(f"({instrument}) Phase 4 selected features saved to {files['phase_4']}")
            save_column_list(df=df_phase4, parquet_file_path=files['phase_4'], instrument=instrument)
        except Exception as e:
            logging.exception(f"({instrument}) Could not save final Phase 4 file: {e}")
            return f"FAILED ({instrument}): Could not save final file."

        status_message = f"SUCCESS ({instrument}): Pipeline complete. Final file: {files['phase_4']}"

    except Exception as e:
        logging.exception(f"An unexpected error occurred in pipeline for {instrument}: {e}")
        status_message = f"CRITICAL FAILURE ({base_filename_str}): Pipeline crashed. Error: {e}"

    logging.info(f"========== PIPELINE FINISHED FOR: {instrument} ==========")
    return status_message