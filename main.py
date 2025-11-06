# main.py

import logging
from dotenv import load_dotenv
from app_config import get_query_parameters
from data_exporter import save_to_parquet
from config import get_db_connection
from repository import get_candles_data


def main():
    """Main function to run the program."""

    # 1. Load Configuration
    logging.info("Loading query parameters...")
    params = get_query_parameters()
    if not params:
        logging.error("Failed to load parameters. Exiting program.")
        return

    client = None
    try:
        # 2. Create Connection
        logging.info("Attempting to connect to ClickHouse...")
        client = get_db_connection()
        if not client:
            # Error already logged by get_db_connection()
            return

        logging.info(f"Query parameters set: Instrument={params['instrument']}, Timeframe={params['timeframe']}")
        logging.info(f"Querying from {params['start_time']} to {params['end_time']}")

        # 3. Fetch Data
        candles = get_candles_data(
            client=client,
            instrument=params['instrument'],
            timeframe=params['timeframe'],
            start_time=params['start_time'],
            end_time=params['end_time']
        )

        # 4. Process and Export Data
        if candles:
            logging.info(f"Successfully retrieved {len(candles)} candle records.")
            # Delegate all export logic to its module
            save_to_parquet(candles, params)

        elif candles == []:
            logging.warning("Query successful, but no data found for this time range.")
        else:
            logging.error("Failed to fetch data.")

    except Exception as e:
        logging.exception(f"An unexpected error occurred in main execution {e}")

    finally:
        # 5. Close Connection
        if client:
            client.disconnect()
            logging.info("Connection to ClickHouse closed.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,  # Set the minimum level to log
        format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    logging.info("Application starting...")
    load_dotenv()
    main()
    logging.info("Application finished.")