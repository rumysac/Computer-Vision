# -*- coding: utf-8 -*-
"""
Created on Sat Nov 1 01:54:05 2021

@author: Rumeysa CELIK
"""
import cv2
import numpy as np
import time 

#PART A: Write a Python script to take live video stream from the webcam of your computer, compute the gradient magnitude on the grayscale version of the image, and display the input and the gradient magnitude on the screen. (The video stream should stop when you hit “q” on your keyboard. You should also display the fps [frame per second] on the video frames.)


# define a video capture object
vid = cv2.VideoCapture(0)
# used to record the time when we processed last frame
prev_frame_time = 0

# used to record the time at which we processed current frame
new_frame_time = 0

# font which we will be using to display FPS
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
      
    # Capture the video frame by frame
    ret, frame = vid.read()
    
    # Display the resulting frame
    # putting the FPS count on the frame
 
    # time when we finish processing for this frame
    new_frame_time = time.time()
    # Calculating the fps
    
    # fps will be number of frame processed in given time frame
    # since their will be most of time error of 0.001 second
    # we will be subtracting it to get more accurate result
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
       
    # converting the fps into integer
    fps = int(fps)
       
    # converting the fps to string so that we can display it on frame
       	# by using putText function
    fps = str(fps)
    cv2.putText(frame, "FPS:"+fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
    cv2.imshow('frame', frame)
    

    # # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #converts BGR to gray
   
    
    # # Apply sobel filter in x and y directions
    grad_x = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=5) #x gradient
    grad_y = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=5) #y gradient
    


    # # scale the gradient to -1 to 1 range
    grad_x = grad_x / np.absolute(grad_x).max()
    grad_x = np.uint8(128+127*grad_x)
    grad_y = grad_y / np.absolute(grad_y).max()
    grad_y = np.uint8(128+127*grad_y)

  
    # computing the gradient magnitude and orientation
    magnitude = np.sqrt((grad_x ** 2) + (grad_y ** 2))
    orientation = np.arctan2(grad_y, grad_x) * (180 / np.pi) % 180 #converting to degree
    
    #converting the arrays from float16 to uint8 to display
    magnitude =magnitude.astype(np.uint8)
    orientation =orientation.astype(np.uint8)
    
    # time when we finish processing for this frame
    new_frame_time = time.time()
    # Calculating the fps
    
    # fps will be number of frame processed in given time frame
    # since their will be most of time error of 0.001 second
    # we will be subtracting it to get more accurate result
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
       
    # converting the fps into integer
    fps = int(fps)
       
    # converting the fps to string so that we can display it on frame
       	# by using putText function
    fps = str(fps)
    cv2.putText(magnitude, "FPS:"+fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
    cv2.imshow('Gradient magnitude',magnitude) 
       
  
    # Wait key for 1ms, if it is 'q' quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the video capture object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
     
 
