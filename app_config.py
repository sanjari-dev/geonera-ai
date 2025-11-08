# file: ai/app_config.py

import os
import logging
from datetime import datetime, UTC, timedelta


def get_query_parameters() -> dict | None:
    """
    Reads and parses query parameters from env vars and constants.
    """
    try:
        instrument = os.getenv('QUERY_INSTRUMENT', 'EURUSD')
        timeframe = os.getenv('QUERY_TIMEFRAME', 'm1')

        start_date_str = "2003-05-01"
        end_date_str = os.getenv('QUERY_END_DATE')

        if not end_date_str:
            logging.error("QUERY_END_DATE must be set in the .env file.")
            return None

        start_date_naive = datetime.strptime(start_date_str, '%Y-%m-%d')
        end_date_naive = datetime.strptime(end_date_str, '%Y-%m-%d')

        start_time = start_date_naive.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=UTC)
        end_date_exclusive = end_date_naive + timedelta(days=1)
        end_time = end_date_exclusive.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=UTC)

        return {
            "instrument": instrument,
            "timeframe": timeframe,
            "start_time": start_time,
            "end_time": end_time,
            "start_date_str": start_date_str,
            "end_date_str": end_date_str
        }

    except ValueError:
        logging.error("Date format in .env must be YYYY-MM-DD.")
        return None
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        return None