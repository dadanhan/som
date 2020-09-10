import cv2 
import numpy as np
import random

#function to show image in a rescaled opencv window
def show_image(winName,image,width,height): 
    cv2.namedWindow(winName,cv2.WINDOW_NORMAL)      #create window
    cv2.resizeWindow(winName,width,height)       #resize window
    cv2.imshow(winName,image)       #show image
    cv2.waitKey(0)      #delete window if button pressed
    cv2.destroyAllWindows()     #free all windows

#function to sample a point in image that is ksize by ksize pixels big
def sample_image(image,gap,kernel,coords,weights):
    [(sel_r,sel_c)] = random.choices(coords,weights)    #sample using weights obtained from pixel intensities
    return sel_r,sel_c,kernel*image[(sel_r-gap):(sel_r+gap+1),(sel_c-gap):(sel_c+gap+1)]

#function to measure distance between random sample and reference vectors
def find_distances(sample,refv,ksize,gap,dist_type='euclidean'):
    sample_vec = [] #put sample into 1D vector
    for i in range(0,ksize):
        for j in range(0,ksize):
            sample_vec.append(sample[i][j])
    if dist_type == 'euclidean':    #find the euclidean distance between sample and reference vectors
        diff = np.power(sample_vec-refv,2)
        dist = np.sqrt(np.sum(diff,axis=1))
    #the middle selected pixel is located print(sample[gap,gap]) print(sample_vec[ksize*gap+gap])
    mindist_index = np.where(dist == min(dist))[0][0] #find minimum
    return dist,sample_vec,mindist_index

#function to reward the closest reference vector
def reward_punish(samv,refv,mindist_index,alpha,stype):
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
            refv[i] = refv[i] + alpha*(samv-refv[i])

#function to sample all pixels and determine which neuron it belongs to
def neuron_image(image,refv,ksize,gap):
    r,c = image.shape
    somim = np.zeros((r-2*gap,c-2*gap))
    for i in range(gap,r-gap):
        for j in range(gap,c-gap):
            dist,samv,mindist_index = find_distances(image[(i-gap):(i+gap+1),(j-gap):(j+gap+1)],refv,ksize,gap)
            somim[i-gap,j-gap]=mindist_index
    return somim

#function for ordering the reference vectors
def get_intensity_order(refv,somim,image,gap):
    mean_ints = []
    std_ints = []
    r,c = image.shape
    tempimage = image[gap:(r-gap),gap:(c-gap)]
    print(tempimage.shape,somim.shape)
    for i in range(0,len(refv)):
        mean_ints.append(np.mean(tempimage[somim==i]))
        std_ints.append(np.std(tempimage[somim==i]))
    print('mean_ints',mean_ints)
    print('std_ints',std_ints)
    index = sorted(range(len(mean_ints)),key=lambda k: mean_ints[k])
    print(index)
    new_somim = np.zeros_like(somim)
    for i in range(0,len(refv)):
        new_somim[somim==index[i]] = i
    return index,mean_ints,std_ints,new_somim

#function for running competitive learning
def competitive_learn_image(stype,image,ksize,nrv,reps,kernel):
    if ksize % 2 == 0:
        raise Exception("Variable ksize must be odd")
    r,c = image.shape   #get shape of image
    gap = int(np.ceil(ksize/2 -1))      #create a kernel for looking at a local area in image
    # print('gap',gap)
    refv = np.array(np.random.randint(np.amin(image),np.amax(image),size=(nrv,ksize*ksize)),dtype='float')      #initialize reference vectors with random numbers in refv between lowest and highest values in image
    #create vector to sample from image
    coords = []
    weights = []
    for i in range(gap,r-gap):
        for j in range(gap,c-gap):
            coords.append((i,j))
            weights.append(float(image[i][j]))
    #metrics to plot later
    means = []
    stds = []
    iters = []
    #loop as many times as reps through different points in image to move weights and learn competitively
    for t in range(0,reps):     
        alpha = 0.5/(t+1)     #set the gain co-efficient
        if t % 100 == 0:    #print t for every multiples
            print(t)
            somim = neuron_image(image,refv,ksize,gap)
            refvorder,mean,std,somim = get_intensity_order(refv,somim,image,gap)
            means.append(mean)
            stds.append(std)
            iters.append(t)
        s_r,s_c,sample = sample_image(image,gap,kernel,coords,weights)     #sample a random point in image 
        dists,samv,mindist_index = find_distances(sample,refv,ksize,gap)    #take this random point and measure distance with reference vectors
        # compete_type = 'self organizing'
        # compete_type = 'abrasive competitive'
        # compete_type = 'simple competitive'
        reward_punish(samv,refv,mindist_index,alpha,stype)    #update refvs
    #get som image and order in intensity of reference vectors
    somim = neuron_image(image,refv,ksize,gap)
    refvorder,mean_final,std_final,somim = get_intensity_order(refv,somim,image,gap)
    means.append(mean_final)
    stds.append(std_final)
    iters.append(t)
    print(refv[refvorder])
    return somim,refv,means,stds,iters