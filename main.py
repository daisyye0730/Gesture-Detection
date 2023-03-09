import cv2
import matplotlib.pyplot as plt
import imutils
import math

# INSTRUCTIONS: 
# open the library and see which image you want to test it on 
# do not include ".png"  
name = input('what is the name of the image? \n')
original_img = cv2.imread('./library/' + name + '.png')

# convert the input image to grayscale
gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

# apply thresholding to convert grayscale to binary image
ret,thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)

# convert BGR to RGB to display using matplotlib
imgRGB = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=2)

# display Original and Binary Images
plt.subplot(131),plt.imshow(imgRGB,cmap = 'gray'),plt.title('Original Image'), plt.axis('off')
plt.subplot(132),plt.imshow(thresh,cmap = 'gray'),plt.title('Binary Image'),plt.axis('off')
plt.show()

# blue the image
thresh = cv2.blur(thresh,(10,10)) 

# find contours in thresholded image, then grab the largest one
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
c = max(cnts, key=cv2.contourArea)

# compute the topmost point of the contour, it will be used to locate the hand
extLeft = tuple(c[c[:, :, 0].argmin()][0])
extTop = tuple(c[c[:, :, 1].argmin()][0])
cX = extTop[0]
cY = extTop[1]
height_divided_3 = imgRGB.shape[0] / 3
width_divided_3 = imgRGB.shape[1] / 3

# compute the location of the contour 
location = ''
if int(cY//height_divided_3) == 1 and int(cX//width_divided_3) == 1:
    location = 'center'
elif int(cY//height_divided_3) == 0 and int(cX//height_divided_3) == 0:
    location = 'upper left'
elif int(cY//height_divided_3) == 0 and int(cX//width_divided_3) == 2:
    location = 'upper right'
elif int(cY//height_divided_3) == 2 and int(cX//width_divided_3) == 0:
    location = 'bottom left'
elif int(cY//height_divided_3) == 2 and int(cX//width_divided_3) == 2:
    location = 'bottom right'
else:
    location = 'unknown'

# identify if it is a fist or a splay by finding the defects in convex hull with respect to hand
hull = cv2.convexHull(c, returnPoints=False)
defects = cv2.convexityDefects(c, hull)
n = 0
for i in range(defects.shape[0]):
    s,e,f,d = defects[i,0]
    start = tuple(c[s][0])
    end = tuple(c[e][0])
    far = tuple(c[f][0])
    
    # find the length of all sides of triangle
    x = math.dist(end, start)
    y = math.dist(far, start)
    z = math.dist(end, far)
    s = (x+y+z)/2
    ar = math.sqrt(s*(s-x)*(s-y)*(s-z))
    
    #distance between point and convex hull
    d=(2*ar)/x
    
    # apply cosine rule
    angle = math.acos((y**2 + z**2 - x**2)/(2*y*z)) * 57

    # ignore angles > 90 and ignore points very close to convex hull
    # this will clean up the points not affected by the hand 
    if angle <= 90 and d>40:
        n += 1
        #cv2.circle(thresh, far, 5, (255,0,0), 5)
hand = ''    
if n == 4:
    hand = 'splay'
elif n == 0:
    #print(math.dist(extLeft, extTop))
    if math.dist(extLeft, extTop) > 200:
        hand = 'palm'
    else:
        hand = 'fist'
else:
    hand = 'unrecognized'
if hand == 'palm':
    cv2.putText(thresh, hand, (int(width_divided_3), int(height_divided_3)), cv2.FONT_HERSHEY_SIMPLEX, 1, (105,105,105), 2)
else:
    cv2.putText(thresh, location + ', ' + hand, (int(width_divided_3), int(height_divided_3)), cv2.FONT_HERSHEY_SIMPLEX, 1, (105,105,105), 2)

# show the output image
cv2.imshow("Image", thresh)
cv2.waitKey(0)