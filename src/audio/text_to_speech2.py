'''
    NOTE: Need gTTS and pygame installed
    Do:
        pip install gTTS
        pip install pygame
'''

from gtts import gTTS
import pygame
import os

class Text2Speech:
    '''
        Convert text to speech

        Attributes:
            className -> Name of the class, can be whatever the flip ypu want
            language  -> Language to speak the text in. The default is en (English)
            slow      -> True or False: Determine if the speech should be speak more slowly
            saveFile  -> Name of the sound file to saved, usually .mp3 type. The default is out.mp3
    '''
    def __init__(self,
                className: str = "Text2Sppech",
                language : str = "en", 
                slow     : bool= False, 
                saveFile : str = "out.mp3"):
        
        self.Name = className
        self.lang = language
        self.slow = slow
        self.saveFile = saveFile

        # Initialize pygame.mixer if it hasnt been initialized
        if pygame.mixer.get_init() is None:
            pygame.mixer.init()


    def Speak(self, text:str):
        '''
            Convert text to speech

            Args:
                text -> The text to convert into sppech
        '''
        try:
            obj = gTTS(text=text, lang=self.lang, slow=self.slow)
            obj.save(self.saveFile)
            pygame.mixer.music.load(self.saveFile)
            pygame.mixer.music.play()
            pygame.time.delay(1000)
        except Exception as e:
            print(f"Speak Error: \'{e}\'")
    
    def clean(self):
        '''Clean up all ouput files'''
        
        os.system(f"rm {self.fileName}")
        os.system("rm *.mp3 *.wav")

# Debug and test
if __name__ == "__main__":
    t2p = Text2Speech("Text2Speech01", "en", False, "output.mp3")
    while True:
        text = input("Enter Text: ")
        t2p.Speak(text)
        print(text)