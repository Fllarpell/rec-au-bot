import logging
import os
from aiogram import types
from utils.audio_processing import convert_to_wav, transcribe_audio


async def handle_voice(message: types.Message):
    logging.info("Voice message is received")

    voice_file = await message.voice.download()
    input_file_path = voice_file.name
    output_file_path = input_file_path.replace(".oga", ".wav")

    try:
        convert_to_wav(input_file_path, output_file_path)

        transcription = transcribe_audio(output_file_path)

        await message.reply(f"Recognition: {transcription}")
    except Exception as e:
        logging.error(f"Error with processing voice message: {e}")
        await message.reply("Error with processing voice message.")
    finally:
        if os.path.exists(input_file_path):
            os.remove(input_file_path)
        if os.path.exists(output_file_path):
            os.remove(output_file_path)


def register_handlers(dp):
    dp.register_message_handler(handle_voice, content_types=types.ContentType.VOICE)
