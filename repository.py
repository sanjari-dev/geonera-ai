# repository.py

import logging
from datetime import datetime
from clickhouse_driver import Client


def get_candles_data(client: Client, instrument: str, timeframe: str, start_time: datetime, end_time: datetime) -> list | None:
    """
    Fetches complete candle data from the geonera.candles table.
    """

    query = """
            SELECT
                timestamp, instrument, timeframe, open, high, low, close, tick_count, min_spread, max_spread, avg_spread, total_bid_volume, total_ask_volume, vwap
            FROM
                geonera.candles
            WHERE
                instrument = %(instrument)s
              AND
                timeframe = %(timeframe)s
              AND
                timestamp >= %(start_time)s
              AND
                timestamp <= %(end_time)s
            ORDER BY
                timestamp
            """

    params = {
        'instrument': instrument,
        'timeframe': timeframe,
        'start_time': start_time,
        'end_time': end_time
    }

    try:
        logging.info(f"Fetching data for {instrument} ({timeframe}) from {start_time} to {end_time}...")
        results = client.execute(query, params)
        return results

    except Exception as e:
        logging.error(f"Error fetching data: {e}")
        return None