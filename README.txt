#################################################################################
#This is for the final project of Hybrid Computing course, Columbia University
#Project name: Object tracking based on video analysis (OTVA)
#Author: Guowei Xu (gx2127), Jianqiao Zhao (jz2778)
#Professor: Zoran Kostic
##################################################################################

Our GPU implementation achieved a 50-100 times speedup compared to traditional CPU-based tracking methods.  

Dependency required to run the code:
OpenCV 2.4.13
CUDA Toolkit 8.0
PyCUDA

We set up all dependency on local machine and ran it.

Input file: "soccer.mp4"
Output file: "win1_runtime.png", "win2_runtime.png" and a Jupyter notebook file "test.ipynb" which contains 
all the print out

OTVA_PYCUDA.py incorporates both CPU and GPU version of the histogram calculation and back projection calculation.
It does everything including the basic object tracking task as well as doing validation, printing runtime and speedup.

To run the code, simply run "python OTVA_PYCUDA.py" or you can open the "test.ipynb" Jupyter notebook file
and run the command "%run OTVA_PYCUDA.py", it will give you all the details.

The video file must be put in the same directory as the source code. We don't recommend you change another video
since we hardcoded the initial tracking window index, which is very likely to be different in different videos.
