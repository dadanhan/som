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
    for i in range(1,10):
        ksizes.append(2*i+1)
    nrv=4
    reps=1000
    for ksize in ksizes:
        print('ksize',ksize)
        somim,refv,means,stds,iters = sf.competitive_learn_image('self organizing',im,ksize,nrv,reps,kernel=np.ones((ksize,ksize)))
        print(somim.shape,im.shape)
        #plot and save results
        plt.figure()
        plt.subplot(221)
        plt.imshow(im)
        plt.subplot(222)
        plt.imshow(somim)
        plt.colorbar()
        plt.subplot(223)
        plt.plot(iters,means)
        plt.xlabel('t')
        plt.ylabel('mean int.')
        plt.subplot(224)
        plt.plot(iters,stds)
        plt.xlabel('t')
        plt.ylabel('std int.')
        plt.tight_layout()
        plt.show()
        # plt.savefig('./results/som_k'+str(ksize)+'.png',dpi=300)

if __name__=="__main__":
    main(sys.argv)