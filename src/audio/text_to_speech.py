'''NOTE: There are some issues I found when I started working on text_to_speech.py on WSL:
        1. pysttsx3 hasn't been installed
            Relatively quick fix. Just 'pip install pysttsx3'
        
        2. Need eSpeak or eSpeak-ng (RECOMMENDED) installed
            Do:
                sudo apt update
                sudo apt install eSpeak or eSpeak-ng
        
        3. Need to install alsa-ultils
            Do:
                sudo apt update
                sudo apt install alsa-ultils

    *** ALL STEPS AND TESTS MENTIONED ABOVE APPLIED ON WSL ONLY ***

    Also, the audio quality is shit in WSL comparing to running on Windows. I don't really know why.
    Also, for some reasons, this doesn't work in venv and I don't know why.

    A better version can be seen in text_to_speech2.py, where gTTS is used instead of pyttsx3. It provides
    a better audio quality but produce an extra .mp3 file and therefor maybe extra delay. Also it may
    require internet connection.
'''

import pyttsx3

def init():
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)  # Adjust speech rate if needed
        engine.setProperty('volume', 1)  # Set volume to maximum

        # Check available voices and set the default one
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[1].id)

        return engine
    except Exception as e:
        print(f"Error: {e}")

def Text2Speech(text:str, engine):
    try:
        engine.say(text)

        # Loop indefinitely with block for now
        engine.runAndWait()
    except Exception as e:
        print(f"Error: {e}")

    print(Text)

# Debug and Test
if __name__ == "__main__":
    engine = init()
    while True:
        Text = input("Enter an input prompt: ")
        Text2Speech(Text, engine)