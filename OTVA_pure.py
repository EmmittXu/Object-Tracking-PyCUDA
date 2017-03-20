###################################################################
# This is the GPU version code for the final project of EECSE4750,
# Hybrid Computing, Columbia University , 2016 Fall
# Auther: Guowei Xu (gx2127), Jianqiao Zhao (jz2778)
# Created on 12/17/2016
###################################################################


# We use several opencv library function
import cv2
import matplotlib as mpl
mpl.use('agg')
import numpy as np
import time
from pycuda import driver, compiler, gpuarray, tools
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('Agg')
import pycuda.autoinit


#########################################
# This function is used to validate the results
# of the GPU back projection and the CPU one
#########################################
def back_proj_validate(g, c):
    if g.shape != c.shape:
        return False
    for i, j in zip(g.flatten(), c.flatten()):
        if abs(i - j) > 1:
            return False
    return True


################################################
#### CPU HISTOGRAM CODE
#################################################
# This is a cpu code for calculating image histograms
# Compute histogram in Python:
def hist(x):
    bins = np.zeros(180, np.float32)
    for v in x.flat:
        bins[v] += 1
    return bins.reshape(180, 1)


#############################################################################
# Kernel code starts here
# We implement two kernel, one is histogram, the other is back projection
##############################################################################
kernel_code = """
# define TILE_SIZE %(TILE_SIZE)s
# define MATRIX_LENGTH %(MATRIX_LENGTH)s
# define MATRIX_WIDTH %(MATRIX_WIDTH)s

//This is the kernel for calculating histogram
//Shared memory, thread syncronizing and atomic operation are used
__global__ void func1(unsigned char *img, unsigned int *bins){
    const unsigned int P = %(P)s;
    unsigned int k;
    volatile __shared__ unsigned int bins_loc[256];
    unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;
    for (k=0; k<256; k++)
        bins_loc[k] = 0.0;
    for (k=0; k<P; k++)
        ++bins_loc[img[i*P+k]];
    // Set the barrier for all the threads
    __syncthreads();
    for (k=0; k<256; k++)
        // Use atomic addition.
        atomicAdd(&bins[k],bins_loc[k]);
}


//This is the kernel code for calculating back projection
//We use tiling and check boundary condition

__global__ void backproj(unsigned int *frame, unsigned int *bin, unsigned int *back){
	int frame_hue;
	int Pvalue;
    int x = blockIdx.y * TILE_SIZE + threadIdx.x;
    int y = blockIdx.x * TILE_SIZE + threadIdx.y;
    if (x < MATRIX_LENGTH && y < MATRIX_WIDTH)
	    frame_hue = frame[y*(MATRIX_LENGTH)+x];
    __syncthreads();

	if (x < MATRIX_LENGTH && y < MATRIX_WIDTH)
	    Pvalue = bin[frame_hue];
	__syncthreads();

	if (x < MATRIX_LENGTH && y < MATRIX_WIDTH)
	    back[y*(MATRIX_LENGTH)+x] = Pvalue;
	__syncthreads();

}

"""
kernel_code = kernel_code % {'N': 160 * 96, 'P': 96, 'TILE_SIZE': 32, 'MATRIX_LENGTH': 1222, 'MATRIX_WIDTH': 2178}

# Compile kernel code
mod = compiler.SourceModule(kernel_code)
given_hist = mod.get_function("func1")
back_proj = mod.get_function("backproj")

# OpenCV function to load videos
cap = cv2.VideoCapture('soccer.MP4')
# OpenCV cap.read() function return the frame of the video,
# If read successfully, return True to ret
# Variable "frame" is the whole frame of the video
ret, frame = cap.read()

# hardcoded initial location of the tracking windows
# 1 denotes the first window; 2 denotes the second window
# r1: row index of the top-left corner of the first tracking window
# c1: column index of the top-left corner of the first tracking window
# h1: height of the first tracking window
# w1: width of the first tracking window
r1, h1, c1, w1 = 400, 150, 950, 80
r2, h2, c2, w2 = 750, 200, 1150, 100

# Create two tracking windows
track_window1 = (c1, r1, w1, h1)
track_window2 = (c2, r2, w2, h2)

# Extract tracking window from the whole frame
roi1 = frame[r1:r1 + h1, c1:c1 + w1]
roi2 = frame[r2:r2 + h2, c2:c2 + w2]

# Transform image from RGB to HSV color using OpenCV function
# HSV stands for hue, saturation, and value, and is also often called HSB (B for brightness)
hsv_roi1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2HSV)
hsv_roi2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2HSV)

# hsv_roi1 and hsv_roi2 are three dimentional matrixs
# We only need 2-D matrix
ChanelH1 = np.zeros((h1, w1)).astype(np.uint8)
ChanelH2 = np.zeros((h2, w2)).astype(np.uint8)

# Extract the H (hue) dimention from hsv_roi1
for i in range(0, h1):
    for j in range(0, w1):
        ChanelH1[i, j] = hsv_roi1[i, j, 0].astype(np.uint8)

# Extract the H (hue) dimention from hsv_roi2
for k in range(0, h2):
    for m in range(0, w2):
        ChanelH2[k, m] = hsv_roi2[k, m, 0].astype(np.uint8)

###########################################################
# Calculate histogram of the tracking window using cpu
###########################################################
#hist_time_cpu1=[]
#for i in range(1):
    #start = time.time()
    #roi_hist1_cpu = hist(ChanelH1)
    #end = time.time()
    #hist_time_cpu1.append(end-start)
#print"Window 1 histogram cpu time is ", np.average(hist_time_cpu1)

#hist_time_cpu2=[]
#for i in range(1):
    #start = time.time()
    #roi_hist2_cpu = hist(ChanelH2)
    #end = time.time()
    #hist_time_cpu2.append(end-start)
#print"Window 2 histogram cpu time is ", np.average(hist_time_cpu2)

#################################################################
# Initialize GPU memory for histogram calculation
##################################################################
img_gpu1 = gpuarray.to_gpu(ChanelH1)
bin_gpu1 = gpuarray.zeros((180, 1), np.uint32)
img_gpu2 = gpuarray.to_gpu(ChanelH2)
bin_gpu2 = gpuarray.zeros((180, 1), np.uint32)

# Call GPU function to do histogram
# First window
hist_time_gpu1=[]
hist_time_gpu2=[]
for i in range(1):
    #start1 = time.time()
    given_hist(img_gpu1, bin_gpu1, block=(1, 1, 1), grid=(h1 * w1 / 96, 1, 1))
    #end1 = time.time()
    #hist_time_gpu1.append(end1-start1)
roi_hist1 = bin_gpu1.get().astype(np.uint32)

# Second window
for i in range(1):
    #start2 = time.time()
    given_hist(img_gpu2, bin_gpu2, block=(1, 1, 1), grid=(h1 * w1 / 96, 1, 1))
    #end2 = time.time()
    #hist_time_gpu2.append(end2-start2)
roi_hist2 = bin_gpu2.get().astype(np.uint32)
#print "Window 1 GPU time is ", np.average(hist_time_gpu1)
#print "Histogram speedup T(CPU)/T(GPU): ", np.average(hist_time_cpu1)/(np.average(hist_time_gpu1))


#print "Window 2 GPU time is ", np.average(hist_time_gpu2)
#print "Histogram speedup T(CPU)/T(GPU): ", np.average(hist_time_cpu2)/(np.average(hist_time_gpu2))

#print "GPU histogram equals to CPU result? ", np.allclose(roi_hist1_cpu, roi_hist1)

##################################################
# Do normalization before doing back projection
###################################################
roi_hist1 = roi_hist1.astype(np.float32)
roi_hist2 = roi_hist2.astype(np.float32)
# Use OpenCV function to do normalization
cv2.normalize(roi_hist1, roi_hist1, 0, 255, cv2.NORM_MINMAX)
cv2.normalize(roi_hist2, roi_hist2, 0, 255, cv2.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

#######################################################################
# Initialize GPU memory for calculating back projection
########################################################################
bin_gpu1 = gpuarray.to_gpu(roi_hist1.astype(np.uint32))
backproj_gpu1 = gpuarray.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint32)
backproj_cpu1 = np.zeros_like(frame[:, :, 0], dtype=np.uint32)

bin_gpu2 = gpuarray.to_gpu(roi_hist2.astype(np.uint32))
backproj_gpu2 = gpuarray.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint32)
backproj_cpu2 = np.zeros_like(frame[:, :, 0], dtype=np.uint32)

time_cpu1 = []
time_gpu1 = []
time_cpu2 = []
time_gpu2 = []

Time_cpu1 = []
Time_gpu1 = []
Time_cpu2 = []
Time_gpu2 = []

frame_index = 1
while (1):
    print "This is the {} frame".format(frame_index)
    frame_index += 1
    ret, frame = cap.read()
    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv_gpu = gpuarray.to_gpu(hsv[:, :, 0].astype(np.uint32))

        for i in range(0,1):
            #start = time.time()
            back_proj(hsv_gpu, bin_gpu1, backproj_gpu1, block=(32, 32, 1), grid=(100, 220))
            #end = time.time()
            #time_gpu1.append(end - start)
        #Time_gpu1.append(np.average(time_gpu1))
        #print "Window 1 Back projection GPU time", np.average(time_gpu1)
        backproj_cpu1 = backproj_gpu1.get()

        for i in range(0,1):
            #start = time.time()
            back_proj(hsv_gpu, bin_gpu2, backproj_gpu2, block=(32, 32, 1), grid=(100, 220))
            #end = time.time()
            #time_gpu2.append(end-start)
        #Time_gpu2.append(np.average(time_gpu2))
        #print "Window 2 Back projection GPU time", np.average(time_gpu2)
        backproj_cpu2 = backproj_gpu2.get()

        for i in xrange(0,1):  # repeat 10 times to get average runtime
            #start = time.time()
            dst1 = cv2.calcBackProject([hsv], [0], roi_hist1, [0, 180], 1)
            #end = time.time()
            #time_cpu1.append(end - start)
        #Time_cpu1.append(np.average(time_cpu1))
        #print "Window 1 Back projection CPU time", np.average(time_cpu1)

        for i in range(0,1):
            #start = time.time()
            dst2 = cv2.calcBackProject([hsv], [0], roi_hist2, [0, 180], 1)
            #end = time.time()
            #time_cpu2.append(end-start)
        #Time_cpu2.append(np.average(time_cpu2))
        #print "Window 2 Back projection CPU time", np.average(time_cpu2)
        
        # Validate, it takes a lot of time to run on CPU, just comment it
        #print "Window 1 Back projection GPU equals to CPU?", back_proj_validate(dst1,backproj_cpu1)
        #print "Window 1 speedup ", np.average(time_cpu1)/np.average(time_gpu1)
        #print "Window 2 Back projection GPU equals to CPU?", back_proj_validate(dst2,backproj_cpu2)
        #print "Window 2 speedup ", np.average(time_cpu2)/np.average(time_gpu2)
        
        # apply meanshift to get the new location
        ret, track_window1 = cv2.meanShift(np.float32(backproj_cpu1), track_window1, term_crit)
        ret, track_window2 = cv2.meanShift(dst2, track_window2, term_crit)

        # Draw it on image
        x1, y1, w1, h1 = track_window1
        x2, y2, w2, h2 = track_window2

        img1 = cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), 255, 2)
        img2 = cv2.rectangle(frame, (x2, y2), (x2 + w2, y2 + h2), (10, 255, 255), 2)
        imS = cv2.resize(frame, (1089, 611))  # Resize image
        cv2.imshow('Object Tracking', imS)

        k = cv2.waitKey(5) & 0xff
        if k == 27:
            break
        else:
            cv2.imwrite(chr(k) + ".jpg", img1)
    else:
        break
cv2.destroyAllWindows()
cap.release()

#plt.ioff()
#plt.gcf()
#plt.plot(Time_cpu1, label="cpu")
#plt.plot(Time_gpu1, label="gpu")
#plt.legend(loc='upper right')
#plt.title('Tracking window 1')
#plt.xlabel('frame index')
#plt.ylabel('time/second')
#plt.gca().set_xlim(0, frame_index)
#plt.savefig('win1_runtime.png')

#plt.plot(Time_cpu2, label="cpu")
#plt.plot(Time_gpu2, label="gpu")
#plt.legend(loc='upper right')
#plt.title('Tracking window 2')
#plt.xlabel('frame index')
#plt.ylabel('time/second')
#plt.gca().set_xlim(0, frame_index)
#plt.savefig('win2_runtime.png')
#print "The average back projection speedup for window 1 is", np.average(Time_cpu1)/np.average(Time_gpu1)
#print "The average back projection speedup for window 2 is", np.average(Time_cpu2)/np.average(Time_gpu2)
#print "Program runs over"