#!/usr/bin/env python3
"""
Voice to Object Integration Module

This module integrates voice recognition and text processing to extract
object information from voice commands. It provides a seamless flow from
voice input to structured object data that can be used by object detection models.
"""

import logging
import time
import sys
import os
from typing import Optional, Dict, Any, Tuple

# Add the parent directory to the path to make imports work when running directly
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..')))

# Now import the modules using relative imports when running as part of the package
# or absolute imports when running directly
try:
    # Try relative imports first (when imported as part of the package)
    from ..audio.voice_recognition import get_voice_command
    from ..text.text_processor import process_user_prompt, ObjectInfo
except ImportError:
    # Fall back to absolute imports (when run directly)
    from src.audio.voice_recognition import get_voice_command
    from src.text.text_processor import process_user_prompt, ObjectInfo

# Set up logging
logger = logging.getLogger("AIVOL.integration.voice_to_object")


class VoiceObjectExtractor:
    """
    A class that integrates voice recognition with text processing to extract
    object information from voice commands.
    """

    def __init__(self,
                 timeout: int = 5,
                 phrase_time_limit: int = 30,
                 ambient_noise_duration: float = 0.5,
                 retry_attempts: int = 2):
        """
        Initialize the VoiceObjectExtractor.

        Parameters:
            timeout (int): Timeout in seconds for waiting for a voice command to start
            phrase_time_limit (int): Maximum duration in seconds to listen for a phrase
            ambient_noise_duration (float): Duration in seconds to adjust for ambient noise
            retry_attempts (int): Number of retry attempts if voice recognition fails
        """
        self.timeout = timeout
        self.phrase_time_limit = phrase_time_limit
        self.ambient_noise_duration = ambient_noise_duration
        self.retry_attempts = retry_attempts
        logger.info("VoiceObjectExtractor initialized with timeout=%d, phrase_time_limit=%d",
                    timeout, phrase_time_limit)

    def listen_and_extract(self) -> Tuple[Optional[ObjectInfo], Optional[str]]:
        """
        Listen for a voice command and extract object information.

        Returns:
            Tuple[Optional[ObjectInfo], Optional[str]]: 
                - The extracted object information
                - The original voice command text (for reference)
        """
        logger.info("Listening for voice command...")

        # Try to get a voice command with retries
        voice_command = None
        for attempt in range(self.retry_attempts + 1):
            if attempt > 0:
                logger.info("Retry attempt %d of %d",
                            attempt, self.retry_attempts)
                time.sleep(1)  # Brief pause before retry

            voice_command = get_voice_command(
                timeout=self.timeout,
                phrase_time_limit=self.phrase_time_limit,
                ambient_noise_duration=self.ambient_noise_duration
            )

            if voice_command:
                break

        if not voice_command:
            logger.warning("Failed to capture voice command after %d attempts",
                           self.retry_attempts + 1)
            return None, None

        logger.info("Voice command captured: '%s'", voice_command)

        # Process the voice command to extract object information
        try:
            object_info = process_user_prompt(voice_command)
            logger.info("Extracted object info: %s", object_info)
            return object_info, voice_command
        except Exception as e:
            logger.error("Error processing voice command: %s", e)
            return None, voice_command


def extract_object_from_voice() -> Tuple[Optional[ObjectInfo], Optional[str]]:
    """
    Convenience function to extract object information from a voice command
    using default parameters.

    Returns:
        Tuple[Optional[ObjectInfo], Optional[str]]: 
            - The extracted object information
            - The original voice command text (for reference)
    """
    extractor = VoiceObjectExtractor()
    return extractor.listen_and_extract()


if __name__ == "__main__":
    # Configure logging for standalone testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=== Voice to Object Integration Test ===")
    print("Please speak a command to find an object (e.g., 'Where is my red cup?')")

    object_info, voice_command = extract_object_from_voice()

    if voice_command:
        print(f"\nRecognized voice command: '{voice_command}'")
    else:
        print("\nNo voice command was recognized.")

    if object_info and object_info.name:
        print(f"Extracted object: {object_info}")
        print("Object details:")
        for key, value in object_info.to_dict().items():
            if value:
                print(f"  {key}: {value}")
    else:
        print("Could not extract object information from the voice command.")
