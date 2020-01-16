#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import matplotlib.pyplot as plt
import cv2 
print("OpenCV Version: ", cv2.__version__)


# In[9]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[19]:


# Load a JPG image as an array
my_image = cv2.imread('monalisa.jpg')
# convert the image from BGR to RGB color space 
my_image = cv2.cvtColor(my_image, cv2.COLOR_BGR2RGB)


# In[12]:


# Show size of the array
print("Original image array shape: ", my_image.shape)


# In[13]:


# Show values for pixel (100,100)
print ("Pixel (100,100) values: ", my_image[100][100][:])


# In[15]:


# Resize the image
my_image = cv2.resize(my_image, (400,600)) 
plt.imshow(my_image)
plt.show()


# In[16]:


# Show size of the array
print("Resized image array shape: ", my_image.shape)


# In[17]:


# convert the image from RGB to BGR color space 
my_image = cv2.cvtColor(my_image, cv2.COLOR_RGB2BGR) 
# Save the new image
cv2.imwrite('new_monalisa.jpg', my_image)


# In[18]:


# convert the image to greyscale
my_grey = cv2.cvtColor(my_image, cv2.COLOR_RGB2GRAY) 
print('Image converted to grayscale.') 
plt.imshow(my_grey,cmap='gray')
plt.show()


# In[20]:


# Load a JPG image as an array
my_image = cv2.imread('new_monalisa.jpg')
# convert the image from BGR to RGB color space 
my_image = cv2.cvtColor(my_image, cv2.COLOR_BGR2RGB)


# In[21]:


# draw a black filled rectangle at top left 
my_image[10:100,10:100,:] = 0 
plt.imshow(my_image)


# In[23]:


# draw a red filled rectangle at top right 
my_image[10:100,300:390,:] = 0
# fill in the red channel with maximum value (255) 
my_image[10:100,300:390,0] = 255 
plt.imshow(my_image)


# In[25]:


# get the face as region of interest - roi 
roi = my_image[50:250,125:250,:]
# resize the roi
roi = cv2.resize(roi,(300,300))
# draw the roi pixels elsewhere in image 
my_image[300:600,50:350,:] = roi 
plt.imshow(my_image)


# In[26]:


# Load a JPG image as an array
my_image = cv2.imread('new_monalisa.jpg')
# convert the image from BGR to RGB color space 
my_image = cv2.cvtColor(my_image, cv2.COLOR_BGR2RGB)


# In[27]:


# define a function to show image
# takes parameters p_image and p_title 
def show_image(p_image, p_title):
    plt.figure(figsize=(5,10)) 
    plt.axis('off') 
    plt.title(p_title) 
    plt.imshow(p_image)


# In[28]:


# make a copy of the image 
temp_image = my_image.copy()


# In[29]:


# draw a line of blue color = (0,0,255) in RGB colorspace - line width is 5px
cv2.line(temp_image, (10,100), (390,100), (0,0,255), 5)


# In[30]:


# draw a rectangle at coordinates of line 5px 
cv2.rectangle(temp_image, (200,200), (300,400), (0,255,255), 5)


# In[31]:


# draw a circle - for filled option set linewidth -1 
cv2.circle(temp_image,(100,200), 50, (255,0,0), -1)


# In[32]:


# draw some text on the image
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(temp_image,'Mona Lisa',(10,500), font, 1.5, (255,255,255), 2, cv2.LINE_AA)


# In[33]:


# call our function to display image
show_image(temp_image,'Result 1: Draw geometry and text')


# In[35]:


#Perform Computer Vision Thresholding Operation on the Image
# make a copy of the original image
temp_image = my_image.copy()

# convert to grayscale
gray = cv2.cvtColor(temp_image, cv2.COLOR_RGB2GRAY)

# create threshold for the image using different algorithms
# last parameter here is the algorithm - we will check for pixel intensity > 100
ret,thresh1 = cv2.threshold(gray,100,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(gray,100,255,cv2.THRESH_BINARY_INV) 
ret,thresh3 = cv2.threshold(gray,100,255,cv2.THRESH_TRUNC) 
ret,thresh4 = cv2.threshold(gray,100,255,cv2.THRESH_TOZERO) 
ret,thresh5 = cv2.threshold(gray,100,255,cv2.THRESH_TOZERO_INV)


# In[36]:


# set an array of titles for above algorithm results
titles = ['Original Image','BINARY Threshold','BINARY_INV Threshold','TRUNC Threshold','TOZERO Threshold','TOZERO_INV Threshold'] 
# create an array of results images
images = [gray, thresh1, thresh2, thresh3, thresh4, thresh5]


# In[37]:


# now we will plot these images as an array 
plt.figure(figsize=(15,15))
for i in np.arange(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray') 
    plt.title(titles[i])
    plt.axis('off')
plt.show()


# In[38]:


#use a process called convolution to run a filter or kernel over the image
#Run Kernel/Filters on the Image to Blur and Sharpen
# make a copy of the original image 
temp_image = my_image.copy() 
show_image(temp_image,'Original image')

# first apply the kernel for smoothing or blurring 
kernel = np.ones((3,3),np.float32)/9
result = cv2.filter2D(temp_image,-1,kernel)

# apply burring twice to see better effect 
result = cv2.filter2D(result,-1,kernel) 
result = cv2.filter2D(result,-1,kernel) 
show_image(result,'Result: Blurring filter')

# apply sharpening filter 
kernel_sharpening = np.array([[-1,-1,-1],[-1, 9,-1],[-1,-1,-1]])
result = cv2.filter2D(temp_image,-1,kernel_sharpening)
show_image(result,'Result: Sharpening filter')


# In[40]:


#Another very useful technique that is often used is to extract geometry information from images
# Run a Canny Edge Detector Algorithm to Detect Edges
#make a copy of the original image
temp_image = my_image.copy()

# convert to grayscale
gray = cv2.cvtColor(temp_image,cv2.COLOR_RGB2GRAY)

# run the Canny algorithm to detect edges 
edges = cv2.Canny(gray,100,255)
plt.figure(figsize=(5,10)) 
plt.axis('off')
plt.title('Result: Canny Edge detection') 
plt.imshow(edges, cmap='gray')


# In[49]:


#OpenCV comes with an algorithm that can look at images and automatically detect 
#faces in them. This algorithm is called Haar Cascades. The idea here is that it 
#tries to use some knowledge of how a face looks in a big array of pixels. It tries to 
#capture knowledge like the fact that our eyes are usually darker than the rest of our face, 
#the region between the eyes is bright, etc. Then, using a cascade of learning units or classifiers, 
#it identifies the coordinates of a face in an image. These classifiers for detecting faces, eyes, ears, etc. 
#are already trained for you and made available on the OpenCV GitHub

#Use Haar Cascades to Detect Face in Image
# make a copy of the original image
temp_image = my_image.copy()

# convert to grayscale
gray = cv2.cvtColor(temp_image,cv2.COLOR_RGB2GRAY)

# load the face cascade model from xml file
face_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml.html')

# find faces and draw green rectangle for each face found 
faces = face_cascade.detectMultiScale(gray,1.3,5)
for (x,y,w,h) in faces:
    roi_color = temp_image[y:y+h, x:x+w]
    # show the roi detected
    show_image(roi_color, 'Result: ROI of face detected by Haar Cascade Classifier') 
    cv2.rectangle(temp_image,(x,y),(x+w,y+h),(0,255,0),2)

# show the image with face detected
show_image(temp_image, 'Result: Face detection using Haar Cascade Classifier')
#NOTE:  NEED TO HAVE XML FILE OR FIGURE OUT HOW TO CLONE FROM URL??

