import numpy as np
import cv2
from matplotlib import pyplot as plt

#Extracting the SIFT features for the images

def drawMatches(img1, kp1, img2, kp2, matches):
    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)


    # Also return the image if you'd like a copy
    return out

def ComputeOrbMatches(img):
	orb = cv2.ORB()
	kp,desc = orb.detectAndCompute(img,None)
	return kp,desc

def matchdescriptors(img1,img2):
	kp1,des1 = ComputeOrbMatches(img1)
	kp2,des2 = ComputeOrbMatches(img2)
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	matches = bf.match(des1,des2)
	# Sort them in the order of their distance.
	matches = sorted(matches, key = lambda x:x.distance)
	img3 = drawMatches(img1,kp1,img2,kp2,matches)
	return img3


img1 = cv2.imread('frames/frame0.png',0)
img2 = cv2.imread('frames/frame1.png',0)
img3 = matchdescriptors(img1,img2)
#print img3
plt.imshow(img3),plt.show()