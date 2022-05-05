# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 19:11:12 2021

@author: Rumeysa CELIK
"""
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np


#PART A: Write a Python scriptto crop out a region from the center; the region size      should be half the size of the input image. Save the image as a pngfile.

img= cv2.imread("/Users/rumeysacelik/Desktop/ps1_CELIK/ps1_Input/huser.jpeg",cv2.COLOR_BGR2GRAY)

cv2.imshow('original image',img)

print("ORIGINAL IMAGE SHAPE:",img.shape) # Print image shape
#Cropping an image

width, height = img.shape[1], img.shape[0]
c_x,c_y=int(width/2),int(height/2) 
mid_x, mid_y = int(width/2), int(height/2)


cropped_img = img[c_y-int(c_y/2):c_y+int(c_y/2),c_x-int(c_x/2):c_x+int(c_x/2)]
print("CROPPED IMAGE SHAPE:",cropped_img.shape)  

# Display cropped image
cv2.imshow("cropped image", cropped_img)

# Save the cropped image
cv2.imwrite("output/Cropped_Image.png", cropped_img)



#PART B: Extract the red channelof the imageand display it.

red_img= img[:,:,2]
cv2.imshow("RED_image", red_img)
# Save the red image
cv2.imwrite("output/RED_image.png", red_img)
 

#PART C: Convert the image to grayscaleand display it.

gray_img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #converts BGR to gray
cv2.imshow('GRAY_img',gray_img)
# Save the red image
cv2.imwrite("output/GRAY_img.png", gray_img)

#PART D: Define the Sobel filters in x and y directionsin your code. Apply thesefilters to the grayscale image and displaythe results.Using these gradients, obtain the gradient magnitude and gradient orientation. Find out a way to display the gradient orientation; display the gradient magnitude and gradient orientation.


gX = cv2.Sobel(gray_img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
gY = cv2.Sobel(gray_img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)

 # scaling the gradients
gX = cv2.convertScaleAbs(gX)
gY = cv2.convertScaleAbs(gY)


# combining the gradient representations into a single image with addWeighted by giving them equal weight
combined = cv2.addWeighted(gX, 0.5, gY, 0.5, 0)
# show our output images
cv2.imshow("Sobel_X", gX)
cv2.imshow("Sobel_Y", gY)
cv2.imshow("Sobel_Combined", combined)
# Save the sobel images
cv2.imwrite("output/Sobel_X.png", gX)
cv2.imwrite("output/Sobel_Y.png", gY)
cv2.imwrite("output/Sobel_Combined.png", combined)

# computing the gradient magnitude and orientation
magnitude = np.sqrt((gX ** 2) + (gY ** 2))
orientation = np.arctan2(gY, gX) * (180 / np.pi) % 180 #converting to degree


print("MAGNITUDE\n",magnitude)
print("ORIENTATION\n",orientation)

#converting the arrays from float16 to uint8 to display
magnitude =magnitude.astype(np.uint8)
orientation =orientation.astype(np.uint8)

print("MAGNITUDE\n",magnitude)
print("ORIENTATION\n",orientation)
# Save the gradient magnitude and orientation 
cv2.imwrite("output/magnitude.png", magnitude)
cv2.imwrite("output/orientation.png", orientation)


# initializing a [1,3] figure to display the input grayscale image along with
# the gradient magnitude and orientation representations, respectively
(fig, axs) = plt.subplots(nrows=1, ncols=3, figsize=(16, 12))
# plot each of the images
axs[0].imshow(gray_img, cmap="gray")
axs[1].imshow(magnitude, cmap="jet")
axs[2].imshow(orientation, cmap="jet")
# set the titles of each axes
axs[0].set_title("Grayscale")
axs[1].set_title("Gradient Magnitude")
axs[2].set_title("Gradient Orientation [0, 180]")
# loop over each of the axes and turn off the x and y ticks
for i in range(0, 3):
	axs[i].get_xaxis().set_ticks([])
	axs[i].get_yaxis().set_ticks([])
# show the plots
plt.tight_layout()
plt.savefig('/Users/rumeysacelik/Desktop/ComputerVisionHW1/gradient.png')
plt.show()

#PART E: Obtain Laplacian of Gaussian imagefor different sigma values, and displaythe results.

"""Laplacian of Gaussian (LoG) is same as applying gaussian smooothing filter first and then 
computing laplacian of the result"""
# Removing noise by blurring with a Gaussian filter
sigma=0####"
src = cv2.GaussianBlur(img, (3, 3),sigma)
# Converting the image to grayscale
src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
# [laplacian]
# Applying Laplace function
dst = cv2.Laplacian(src_gray, cv2.CV_16S, ksize=3)
# converting back to uint8
log = cv2.convertScaleAbs(dst)
# Save the log image
cv2.imwrite("output/Laplacian of Gaussian(sigma=0).png", log)

cv2.imshow("Laplace of Gaussian-sigma=0", log)


sigma=15####
# Removing noise by blurring with a Gaussian filter
src = cv2.GaussianBlur(img, (3, 3),sigma)
# Converting the image to grayscale
src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
# [laplacian]
# Applying Laplace function
dst = cv2.Laplacian(src_gray, cv2.CV_16S, ksize=3)
# converting back to uint8
log = cv2.convertScaleAbs(dst)
# Save the log image
cv2.imwrite("output/Laplacian of Gaussian(sigma=15).png", log)

cv2.imshow("Laplace of Gaussian-sigma=15", log)

cv2.waitKey(0)
cv2.destroyAllWindows()
