# geonera-ai/main.py

import logging
import os
from dotenv import load_dotenv
import asyncio
from pipeline import run_pipeline_for_instrument
from send_telegram import send_telegram_message

OUTPUT_DIR = "resources"

async def main():
    logging.info(f"--- Main Application Starting ---")
    all_statuses = []

    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        logging.info(f"Ensured output directory exists: ./{OUTPUT_DIR}")
    except OSError as e:
        logging.critical(f"Could not create directory {OUTPUT_DIR}: {e}. Exiting.")
        return

    try:
        status = run_pipeline_for_instrument(output_dir=OUTPUT_DIR)
        all_statuses.append(status)
    except Exception as e:
        error_msg = f"CRITICAL FAILURE: Main loop crash: {e}"
        logging.exception(error_msg)
        all_statuses.append(error_msg)

    logging.info("All instrument pipelines complete.")

    success_count = sum(1 for s in all_statuses if s.startswith("SUCCESS"))
    skipped_count = sum(1 for s in all_statuses if s.startswith("SKIPPED"))
    failed_count = sum(1 for s in all_statuses if s.startswith("FAILED") or s.startswith("WARNING") or s.startswith("CRITICAL"))

    summary_message = f"*Geonera Feature Pipeline Summary*\n"
    summary_message += f"Total: {len(all_statuses)} | Success: {success_count} | Skipped: {skipped_count} | Failed: {failed_count}\n"
    summary_message += "---"

    for status in all_statuses:
        if not status.startswith("SKIPPED"):
            if status.startswith("SUCCESS"):
                status_message_short = status.split("Final file:")[0]
                summary_message += f"\n- {status_message_short}"
            else:
                summary_message += f"\n- {status}"

    logging.info("Sending summary to Telegram...")
    await send_telegram_message(summary_message)
    logging.info("--- Main Application Finished ---")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    load_dotenv()
    asyncio.run(main())