#!/usr/bin/env python3
"""
Voice to Object Integration Module (Single-shot)
Optionally used if you want a one-time "listen & parse" approach
instead of the new continuous loop in main.py.
"""

import logging
import time
import sys
import os
from typing import Optional, Tuple

# Make sure your imports align with your folder structure
try:
    from audio.voice_recognition import get_voice_command
    from text.text_processor import process_user_prompt, ObjectInfo
except ImportError:
    # Adjust if needed
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from audio.voice_recognition import get_voice_command
    from text.text_processor import process_user_prompt, ObjectInfo

logger = logging.getLogger("AIVOL.integration.voice_to_object")


class VoiceObjectExtractor:
    """
    A class that integrates voice recognition with text processing to extract
    object information from a single voice command.
    """

    def __init__(
        self,
        timeout: int = 5,
        phrase_time_limit: int = 30,
        ambient_noise_duration: float = 0.5,
        retry_attempts: int = 2
    ):
        self.timeout = timeout
        self.phrase_time_limit = phrase_time_limit
        self.ambient_noise_duration = ambient_noise_duration
        self.retry_attempts = retry_attempts
        logger.info("VoiceObjectExtractor initialized: timeout=%d, phrase_time_limit=%d",
                    timeout, phrase_time_limit)

    def listen_and_extract(self) -> Tuple[Optional[ObjectInfo], Optional[str]]:
        """
        Listen for a voice command and extract object information (one shot).
        Returns the ObjectInfo + the raw voice command text.
        """
        logger.info("Listening for one-shot voice command...")

        voice_command = None
        for attempt in range(self.retry_attempts + 1):
            if attempt > 0:
                logger.info("Retry attempt %d of %d", attempt, self.retry_attempts)
                time.sleep(1)

            voice_command = get_voice_command(
                timeout=self.timeout,
                phrase_time_limit=self.phrase_time_limit,
                ambient_noise_duration=self.ambient_noise_duration
            )

            if voice_command:
                break

        if not voice_command:
            logger.warning("Failed to capture voice command after %d attempts", self.retry_attempts + 1)
            return None, None

        logger.info("Voice command captured: '%s'", voice_command)

        # Parse out object info
        try:
            object_info = process_user_prompt(voice_command)
            logger.info("Extracted object info: %s", object_info)
            return object_info, voice_command
        except Exception as e:
            logger.error("Error processing voice command: %s", e)
            return None, voice_command


def extract_object_from_voice() -> Tuple[Optional[ObjectInfo], Optional[str]]:
    """
    Convenience function to do a quick one-shot extraction.
    """
    extractor = VoiceObjectExtractor()
    return extractor.listen_and_extract()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    obj_info, voice_cmd = extract_object_from_voice()
    if voice_cmd:
        print(f"Voice command: {voice_cmd}")
    if obj_info and obj_info.name:
        print(f"Object info: {obj_info}")
    else:
        print("No object extracted.")
