'''File Name: Cam.py
Description: This program is a script that acts as the interface between the camera and main program for sampling.

Author: Chinmoy, Parth, Het, Dang
Dependency: OpenCV'''

import cv2


'''Function Name: sample()
Input Arguments: None
Output Arguments: Dict

Description: Function samples the camera and returns the error value and picture sample to calling program'''

def sample():
    return_dict = {"ret_val": 0}
    camera = cv2.VideoCapture(0)

    #Condition to check if the camera was set up
    if not camera.isOpened():
        #If camera could not be opened, ret_val = -10
        return_dict["ret_val"] = -10
        return_dict["frame"] = None
        camera.release()
        return return_dict
    
    ret, return_dict["frame"] = camera.read()
    if not ret:
        #If image could not be sampled, ret_val = -20
        return_dict["ret_val"] = -20
        return_dict["frame"] = None
    
    camera.release()
    return return_dict

if __name__ == "__main__":
    cv2.namesWindow("test")
    return_dict = sample()
    cv2.imshow("test", return_dict["frame"])