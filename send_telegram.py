# send_telegram.py

import os
import logging
from telegram import Bot
from telegram.constants import ParseMode


async def send_telegram_message(message: str):
    """
    Sends a message to a predefined Telegram chat.
    Reads token and chat_id from .env variables.
    """
    try:
        token = os.getenv('TELEGRAM_BOT_TOKEN')
        chat_id = os.getenv('TELEGRAM_CHAT_ID')

        if not token or not chat_id:
            logging.warning("Telegram token or chat_id not found in .env. Skipping message.")
            return

        bot = Bot(token=token)
        await bot.send_message(
            chat_id=chat_id,
            text=message,
            parse_mode=ParseMode.MARKDOWN
        )
        logging.info("Telegram notification sent successfully.")
    except Exception as e:
        logging.error(f"Failed to send Telegram message: {e}")