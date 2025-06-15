import os
import asyncio
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from telegram import Update
from .handlers import start, help_command, handle_message
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_WEBHOOK_URL = os.getenv("TELEGRAM_WEBHOOK_URL")

if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN environment variable not set.")
if not TELEGRAM_WEBHOOK_URL:
    raise ValueError("TELEGRAM_WEBHOOK_URL environment variable not set. For webhook deployment.")

async def setup_telegram_bot(application: Application) -> None:
    """Sets up the handlers for the Telegram bot."""
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

async def main() -> None:
    """Runs the Telegram bot with webhook."""
    if not TELEGRAM_BOT_TOKEN:
        print("Error: TELEGRAM_BOT_TOKEN not found in environment variables.")
        return
    if not TELEGRAM_WEBHOOK_URL:
        print("Error: TELEGRAM_WEBHOOK_URL not found in environment variables for webhook setup.")
        return

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    await setup_telegram_bot(application)

    print(f"Setting webhook to: {TELEGRAM_WEBHOOK_URL}")
    await application.bot.set_webhook(url=TELEGRAM_WEBHOOK_URL)
    print("Webhook set successfully.")

async def process_telegram_update(update_json: dict, bot_token: str) -> None:
    """
    Processes a single update from Telegram.
    This function will be called by your FastAPI webhook endpoint.
    """
    if not bot_token:
        raise ValueError("Bot token not provided for processing update.")

    application = Application.builder().token(bot_token).build()
    await setup_telegram_bot(application)
    
    # Initialize the application (this creates the bot instance)
    await application.initialize()
    
    try:
        update = Update.de_json(update_json, application.bot)
        if update:
            await application.process_update(update)
    finally:
        # Clean up
        await application.shutdown()


if __name__ == "__main__":
    print("To set the webhook, you would typically run a function like main() once.")
    print("The bot will then listen for updates via the FastAPI endpoint.")