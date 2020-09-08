import numpy as np
import cv2
import som_functions as sf
import sys
import matplotlib.pyplot as plt

def main(argv):
    im = cv2.imread(argv[1],-1)     #read in image
    print(im)
    # sf.show_image("test",im,600,600)    #show image
    print(im.shape)
    #competitive learning algorithm
    ksizes = []
    for i in range(1,20):
        ksizes.append(2*i+1)
    nrv=4
    reps=1000
    for ksize in ksizes:
        print(ksize)
        somim = sf.competitive_learn_image('self organizing',im,ksize,nrv,reps,kernel=np.ones((ksize,ksize)))
        print(somim.shape,im.shape)
        plt.figure()
        plt.subplot(121)
        plt.imshow(im)
        plt.subplot(122)
        plt.imshow(somim)
        plt.colorbar()
        # plt.show()
        plt.savefig('som_k'+str(ksize)+'.png',dpi=300)

if __name__=="__main__":
    main(sys.argv)