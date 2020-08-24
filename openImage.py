# pushing to git is 
# git status
# git add filename.ext
# git commit -m "Some Message"
# git push origin master

#run script using python3 

#SOM implementation by Daniel Han 24/8/2020

import cv2
import numpy as np
import sys

def main(argv):
    
    im = cv2.imread(argv[1],-1)     #read in image
    im_scaled = np.array(cv2.normalize(im,dst=None,alpha=0,beta=65535,norm_type=cv2.NORM_MINMAX),dtype='uint16')     #scale image
    #print values of im
    print(im_scaled)
    #display image on screen
    winName = "test"        #display window name
    cv2.namedWindow(winName,cv2.WINDOW_NORMAL)      #create window
    cv2.resizeWindow(winName,600,600)       #resize window
    cv2.imshow(winName,im_scaled)       #show image
    cv2.waitKey(0)      #delete window if button pressed
    cv2.destroyAllWindows()     #free all windows

if __name__=="__main__":
    main(sys.argv)