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
import random
import matplotlib.pyplot as plt

#function to show image in a rescaled opencv window
def show_image(winName,image,width,height):
    cv2.namedWindow(winName,cv2.WINDOW_NORMAL)      #create window
    cv2.resizeWindow(winName,width,height)       #resize window
    cv2.imshow(winName,image)       #show image
    cv2.waitKey(0)      #delete window if button pressed
    cv2.destroyAllWindows()     #free all windows

#function to sample a point in image that is ksize by ksize pixels big
def sample_image(image,gap,kernel):
    #brighter parts are sampled with higher probability using image histogram
    #make image co-ordinates to sample and image pixel intensities
    r,c = image.shape
    coords = []
    weights = []
    for i in range(gap,r-gap):
        for j in range(gap,c-gap):
            coords.append((i,j))
            weights.append(float(image[i][j]))
    #sample using weights obtained from pixel intensities
    [(sel_r,sel_c)] = random.choices(coords,weights)
    return sel_r,sel_c,kernel*image[(sel_r-gap):(sel_r+gap+1),(sel_c-gap):(sel_c+gap+1)]

#function to measure distance between random sample and reference vectors
def find_distances(sample,refv,ksize,gap):
    #put sample into 1D vector
    sample_vec = []
    for i in range(0,ksize):
        for j in range(0,ksize):
            sample_vec.append(sample[i][j])
    #find the euclidean distance between sample and reference vectors
    diff = np.power(sample_vec-refv,2)
    dist = np.sqrt(np.sum(diff,axis=1))
    #the middle selected pixel is located
    # print(sample[gap,gap])
    # print(sample_vec[ksize*gap+gap])
    return dist

#function for running competitive learning
def competitive_learn_image(image,ksize,nrv,reps,kernel):
    if ksize % 2 == 0:
        raise Exception("Variable ksize must be odd")
    #get shape of image
    r,c = image.shape
    #create a kernel for looking at a local area in image
    gap = int(np.ceil(ksize/2 -1))
    # print(gap)
    somim = np.zeros((r-gap,c-gap))
    #initialize reference vectors with random numbers in refv between lowest and highest values in image
    refv = np.array(np.random.randint(np.amin(image),np.amax(image),size=(nrv,ksize*ksize)),dtype='float')
    #loop as many times as reps through different points in image to move weights and learn competitively
    for t in range(0,reps):
        #sample a random point in image 
        s_r,s_c,sample = sample_image(image,gap,kernel)
        #take this random point and measure distance with reference vectors
        dists = find_distances(sample,refv,ksize,gap)
        mindist_index = np.where(dists == min(dists))[0][0]
        #select best refv and reward
        
        #punish others refv
        
    return somim

def main(argv):
    im = cv2.imread(argv[1],-1)     #read in image
    im_scaled = np.array(cv2.normalize(im,dst=None,alpha=0,beta=65535,norm_type=cv2.NORM_MINMAX),dtype='uint16')     #scale image

    #display image on screen
    show_image("test",im_scaled,600,600)

    #competitive learning algorithm for an image
    ksize = 3
    nrv = 4
    # somim = competitive_learn_image(im,ksize,nrv,reps=10,kernel=np.ones((ksize,ksize)))
    somim = competitive_learn_image(im,ksize,nrv,reps=1,kernel=np.ones((ksize,ksize)))
    
if __name__=="__main__":
    main(sys.argv)