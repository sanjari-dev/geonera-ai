# file: geonera-ai/utils.py

import logging
import os
import pandas as pd
import subprocess
import psutil
from datetime import timedelta

def save_column_list(df: pd.DataFrame, parquet_file_path: str, instrument: str):
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


def define_file_paths(params: dict, output_dir: str) -> dict:
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


def get_timedelta_for_timeframe(timeframe: str) -> timedelta:
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


def log_resource_usage(step_name: str, file_path: str = None):
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss
    mem_mb = mem_bytes / (1024 * 1024)
    mem_gb = mem_mb / 1024

    log_msg = f"--- [RESOURCE MONITOR] Step: {step_name} | RAM: {mem_mb:.2f} MB ({mem_gb:.2f} GB)"
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,nounits,noheader'],
            encoding='utf-8',
            stderr=subprocess.DEVNULL
        )
        rows = result.strip().split('\n')
        for row in rows:
            idx, used, total = row.split(',')
            log_msg += f" | GPU {idx.strip()} VRAM: {used.strip()}MB / {total.strip()}MB"
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    except Exception as e:
        logging.warning(f"Failed to query GPU VRAM: {e}")

    if file_path:
        if os.path.exists(file_path):
            file_bytes = os.path.getsize(file_path)
            file_mb = file_bytes / (1024 * 1024)
            log_msg += f" | File: {file_mb:.2f} MB"
        else:
            log_msg += f" | File: NOT FOUND"

    logging.info(log_msg)