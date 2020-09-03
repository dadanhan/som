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
def find_distances(sample,refv,ksize,gap,dist_type='euclidean'):
    #put sample into 1D vector
    sample_vec = []
    for i in range(0,ksize):
        for j in range(0,ksize):
            sample_vec.append(sample[i][j])
    #find the euclidean distance between sample and reference vectors
    if dist_type == 'euclidean':
        diff = np.power(sample_vec-refv,2)
        dist = np.sqrt(np.sum(diff,axis=1))
    #the middle selected pixel is located
    # print(sample[gap,gap])
    # print(sample_vec[ksize*gap+gap])
    #find minimum
    mindist_index = np.where(dist == min(dist))[0][0]
    return dist,sample_vec,mindist_index

#function to reward the closest reference vector
def reward_punish(samv,refv,mindist_index,alpha,stype):
    #do something with closest refv
    if stype == 'simple competitive':    #the simplest version of competitive learning
        refv[mindist_index] = refv[mindist_index]+alpha*(samv-refv[mindist_index])
    if stype == 'abrasive competitive':     #extension to simple competitive
        for i in range(0,len(refv)):
            if i == mindist_index:
                refv[i] = refv[i]+alpha*(samv-refv[i])
            else:
                refv[i] = refv[i]-alpha*(samv-refv[i])
    if stype == 'self organizing':      #self organizing map
        for i in range(0,len(refv)):
            refv[i] = refv[i]+alpha*(samv-refv[i])


#function to sample all pixels and determine which neuron it belongs to
def neuron_image(image,refv,ksize,gap):
    r,c = image.shape
    somim = np.zeros((r-gap-1,c-gap-1))
    for i in range(gap,r-gap):
        for j in range(gap,c-gap):
            dist,samv,mindist_index = find_distances(image[(i-gap):(i+gap+1),(j-gap):(j+gap+1)],refv,ksize,gap)
            somim[i-gap,j-gap]=mindist_index
    return somim

#function for running competitive learning
def competitive_learn_image(image,ksize,nrv,reps,kernel):
    if ksize % 2 == 0:
        raise Exception("Variable ksize must be odd")
    #get shape of image
    r,c = image.shape
    #create a kernel for looking at a local area in image
    gap = int(np.ceil(ksize/2 -1))
    #initialize reference vectors with random numbers in refv between lowest and highest values in image
    refv = np.array(np.random.randint(np.amin(image),np.amax(image),size=(nrv,ksize*ksize)),dtype='float')
    print(refv)
    #loop as many times as reps through different points in image to move weights and learn competitively
    for t in range(0,reps):
        alpha = 1/(t+1)     #set the gain co-efficient
        #print t for every multiple of 10
        if t % 100 == 0:
            print(t)
        #sample a random point in image 
        s_r,s_c,sample = sample_image(image,gap,kernel)
        #take this random point and measure distance with reference vectors
        dists,samv,mindist_index = find_distances(sample,refv,ksize,gap)
        #update refvs
        reward_punish(samv,refv,mindist_index,alpha,stype='self organizing')
    #at the end of training
    print(refv)

    return neuron_image(image,refv,ksize,gap)

def main(argv):
    im = cv2.imread(argv[1],-1)     #read in image
    im_scaled = np.array(cv2.normalize(im,dst=None,alpha=0,beta=65535,norm_type=cv2.NORM_MINMAX),dtype='uint16')     #scale image

    #display image on screen
    show_image("test",im_scaled,600,600)

    #competitive learning algorithm for an image
    ksize = 3
    nrv = 4
    reps = 1000
    # somim = competitive_learn_image(im,ksize,nrv,reps=10,kernel=np.ones((ksize,ksize)))
    somim = competitive_learn_image(im,ksize,nrv,reps,kernel=np.ones((ksize,ksize)))

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