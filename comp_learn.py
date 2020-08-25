# pushing to git is 
# git status
# git add filename.ext
# git commit -m "Some Message"
# git push origin master

#run script using python3 

#SOM implementation for automated image segmentation by Daniel Han 24/8/2020

import cv2
import numpy as np
import sys

def show_image(winName,image,width,height):
    cv2.namedWindow(winName,cv2.WINDOW_NORMAL)      #create window
    cv2.resizeWindow(winName,width,height)       #resize window
    cv2.imshow(winName,image)       #show image
    cv2.waitKey(0)      #delete window if button pressed
    cv2.destroyAllWindows()     #free all windows

def main(argv):
    im = cv2.imread(argv[1],-1)     #read in image
    im_scaled = np.array(cv2.normalize(im,dst=None,alpha=0,beta=65535,norm_type=cv2.NORM_MINMAX),dtype='uint16')     #scale image
   
    #print values of im
    # print(im_scaled)
    #get shape of im
    r,c = im.shape
    print(r,c)

    #display image on screen
    show_image("test",im_scaled,600,600)

    #create a kernel for looking at a local area in image
    ksize = 3
    kernel = np.ones((ksize,ksize))
    gap = int(np.ceil(ksize/2 -1))
    somim = np.zeros((r-gap,c-gap))
    #initialize reference vectors
    

if __name__=="__main__":
    main(sys.argv)