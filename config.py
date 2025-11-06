# config.py

import os
import logging
from clickhouse_driver import Client


def get_db_connection() -> Client | None:
    """
    Reads .env configuration and creates a connection to ClickHouse.
    Assumes load_dotenv() has already been called.
    """
    try:
        host = os.getenv('CLICKHOUSE_HOST')
        port = int(os.getenv('CLICKHOUSE_PORT', 9000))
        user = os.getenv('CLICKHOUSE_USER')
        password = os.getenv('CLICKHOUSE_PASSWORD')
        database = os.getenv('CLICKHOUSE_DB')

        if not all([host, user, password, database]):
            logging.error("Ensure all .env variables (HOST, USER, PASSWORD, DB) are set.")
            return None

        client = Client(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            compression=True
        )

        client.execute('SELECT 1')
        logging.info(f"Successfully connected to ClickHouse at {host}:{port}")
        return client

    except Exception as e:
        logging.error(f"Error connecting to ClickHouse: {e}")
        return None