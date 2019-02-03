__author__ = 'deepika'

import cv2 #import OpenCv. (Version 3)
import numpy as np

def meanShift():
    for img in ["1", "2", "3"]: #this is the list of images saved on my local as mentioned in asisgnment.
        for _param in [ (1, 1), (2, 30), (1, 40), (1, 80), (10, 30) ]: #These are the list of parameters i.e. ( spatial window radius, color window radius)
            im = cv2.imread(img + ".jpg")
            im = cv2.cvtColor(im, cv2.COLOR_RGB2LAB) #From RGB to LAB space
            cv2.pyrMeanShiftFiltering(im, _param[0], _param[1], im, 1) #Fixing Pyramid param to 1, vary the rest of parameters.
            cv2.imwrite("Img_"+img + "_" + str(_param[0]) + "_" + str(_param[1]) +".jpg",im) #Write the image on local


def watershed():
    for im in ["1", "2", "3"]:
        count = 1
        for val in [0.1, 0.5, 0.7, 0.9]:

            img = cv2.imread(im + ".jpg")
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

            # noise removal
            kernel = np.ones((3,3),np.uint8)
            opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

            # sure background area
            sure_bg = cv2.dilate(opening,kernel,iterations=3)

            # Finding sure foreground area
            dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
            ret, sure_fg = cv2.threshold(dist_transform,val*dist_transform.max(),255,0)

            # Finding unknown region
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(sure_bg,sure_fg)

            # Marker labelling
            ret, markers = cv2.connectedComponents(sure_fg)

            # Add one to all labels so that sure background is not 0, but 1
            markers = markers+1

            # Now, mark the region of unknown with zero
            markers[unknown==255] = 0

            markers = cv2.watershed(img,markers)
            img[markers == -1] = [255,0,0]

            cv2.imwrite(im + "_" + str(count) + "_result.jpg", img)
            count = count + 1

if __name__ == '__main__':
    meanShift()
    watershed()