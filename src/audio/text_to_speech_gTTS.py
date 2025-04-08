"""
Module: text_to_speech_gTTS.py
Uses Google Text-to-Speech + pygame to speak text.
Requires:
    pip install gTTS pygame
    An active internet connection for Google TTS.
"""

import os
import pygame
from gtts import gTTS
import logging
import time
import datetime

logger = logging.getLogger("AIVOL.audio.gtts_speak")

class GTTSSpeaker:
    def __init__(self, 
                 language: str = "en",
                 slow: bool = False,
                 output_folder: str = "audio_outputs"):
        """
        language: e.g. "en" for English
        slow: speak slower if True
        output_folder: folder path where mp3 files will be saved
        """
        self.language = language
        self.slow = slow
        self.output_folder = output_folder

        # Initialize pygame mixer if not already
        if not pygame.mixer.get_init():
            pygame.mixer.init()
            logger.info("Pygame mixer initialized.")

        # Ensure the output folder exists
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
            logger.info(f"Created audio output folder: {self.output_folder}")

        # We'll keep a small counter in case multiple calls happen in the same second
        self.file_counter = 0

    def speak(self, text: str):
        """
        Convert text to speech using Google TTS, then play via pygame.
        Blocks until playback finishes.
        Also saves each spoken message as a unique MP3 in self.output_folder.
        """
        if not text:
            return
        try:
            # Create a unique filename based on timestamp + a small counter
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.file_counter += 1
            filename = f"tts_{timestamp}_{self.file_counter}.mp3"
            filepath = os.path.join(self.output_folder, filename)

            tts_obj = gTTS(text=text, lang=self.language, slow=self.slow)
            tts_obj.save(filepath)
            logger.info(f"Saved TTS audio to: {filepath}")

            pygame.mixer.music.load(filepath)
            pygame.mixer.music.play()

            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)

        except Exception as e:
            logger.error(f"Error in gTTS speak: {e}")
