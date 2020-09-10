# pushing to git is: "git status", "git add filename.ext", "git commit -m "Some Message"", "git push origin master"
#run script using python3 
#SOM implementation for automated image segmentation by Daniel Han 24/8/2020
import matplotlib.pyplot as plt
import som_functions as sf
import sys
import cv2
import numpy as np 

def main(argv):
    im = cv2.imread(argv[1],-1)     #read in image
    im_scaled = np.array(cv2.normalize(im,dst=None,alpha=0,beta=65535,norm_type=cv2.NORM_MINMAX),dtype='uint16')     #scale image
    sf.show_image("test",im_scaled,600,600)    #display image on screen
    #competitive learning algorithm for an image
    ksize = 5
    nrv = 4
    reps = 1000
    somim,refv,means,stds,iters = sf.competitive_learn_image('self organizing',im,ksize,nrv,reps,kernel=np.ones((ksize,ksize)))        # perform competitive learning
    print(somim.shape,im_scaled.shape)
    plt.figure()
    plt.subplot(121)
    plt.imshow(im_scaled)
    plt.subplot(122)
    plt.imshow(somim)
    plt.colorbar()
    plt.show()

if __name__=="__main__":
    main(sys.argv)