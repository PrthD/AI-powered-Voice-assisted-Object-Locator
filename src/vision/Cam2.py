'''Backup model'''

import cv2

class Camera:
    '''
        Specify the recording device or the video used

        Args & Attributes:
            device    -> 0, 1 or -1: specify the recording device. The default is 0
            videoName -> If device = -1, then switch to video analyze mode with videoName
                         being the name of the video
            cap       -> The video/device object. Can be returned via return_cap()
    '''

    def __init__(self, device: int = 0, videoName: str = ""):
        self.device = device
        self.videoName = videoName
        self.cap = cv2.VideoCapture(device) if device >= 0 else cv2.VideoCapture(videoName)

        if self.cap.isOpened() == False:
            print("Error opening device or video")
    
    def return_cap(self) -> cv2.VideoCapture:
        '''Return the recording device'''
        return self.cap
    
    def cleanup(self):
        '''When finish, clean up the environment and its associated attributes'''
        self.cap.release()
        cv2.destroyAllWindows()

# Debug and test
if __name__ == "__main__":
    device = Camera(0)
    cap = device.return_cap()

    while cap.isOpened():
        ret, frame = cap.read()
       
        if ret == True:
            cv2.imshow('Frame', frame)
        
        key = cv2.waitKey(1)
        
        # Press 's' to stop
        if key == 27:
            break

    device.cleanup()