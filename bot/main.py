import logging
from aiogram import Bot, Dispatcher
from aiogram.utils import executor
from config import API_TOKEN
from handlers import register_handlers

logging.basicConfig(level=logging.INFO)

bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

register_handlers(dp)

if __name__ == "__main__":
    logging.info("Bot is executing")
    executor.start_polling(dp, skip_updates=True)
