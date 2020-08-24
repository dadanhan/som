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
    
    im = cv2.imread(argv[1],-1)

    print(im)

    cv2.imshow('test',im)
    cv2.waitKey(0)

if __name__=="__main__":
    main(sys.argv)