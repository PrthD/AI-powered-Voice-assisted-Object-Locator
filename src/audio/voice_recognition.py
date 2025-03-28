"""
Module: voice_recognition.py
Purpose: Capture audio from the default microphone (or a specified device index) and convert it to text.
"""

import speech_recognition as sr
import logging
from typing import Optional

# Set up module-level logger
logger = logging.getLogger("AIVOL.audio.voice_recognition")


def get_voice_command(
    timeout: int = 5,
    phrase_time_limit: int = 30,
    ambient_noise_duration: float = 0.5
) -> Optional[str]:
    """
    Capture audio from the microphone (device_index=2 by default for RPi) 
    and convert it to text via Google Speech Recognition.

    Args:
        timeout (int): Maximum number of seconds to wait for a phrase to start.
        phrase_time_limit (int): Maximum duration (in seconds) to record once a phrase starts.
        ambient_noise_duration (float): Duration (in seconds) to adjust for ambient noise.

    Returns:
        Optional[str]: The recognized text command, or None if recognition fails.
    """

    recognizer = sr.Recognizer()
    try:
        # --- If your USB mic is on index=2, uncomment below. ---
        with sr.Microphone(device_index=2) as source:
            logger.info("Adjusting for ambient noise for %.2f seconds...", ambient_noise_duration)
            recognizer.adjust_for_ambient_noise(source, duration=ambient_noise_duration)
            logger.info("Listening for a voice command...")
            audio_data = recognizer.listen(
                source, timeout=timeout, phrase_time_limit=phrase_time_limit
            )
    except sr.WaitTimeoutError:
        logger.error("Listening timed out while waiting for a phrase to start.")
        return None
    except Exception as e:
        logger.error("An error occurred while capturing audio: %s", e)
        return None

    try:
        logger.info("Processing audio via Google Speech Recognition...")
        command = recognizer.recognize_google(audio_data)
        logger.info("Voice command recognized: %s", command)
        return command
    except sr.UnknownValueError:
        logger.error("Google Speech Recognition could not understand the audio.")
    except sr.RequestError as e:
        logger.error("Could not request results from Google Speech Recognition service; %s", e)
    except Exception as e:
        logger.error("An unexpected error occurred during speech recognition: %s", e)

    return None


if __name__ == "__main__":
    # Test capturing and printing a command
    command = get_voice_command()
    if command:
        print("Recognized command:", command)
    else:
        print("No command recognized.")
