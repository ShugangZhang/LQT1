#
# General Code Structure (GCS) for one-dimensional plot
# 
# A. Code sructure is re-organised using standard C++ to fit into my project.
# B. The friendly style makes the project easier understood.
# C. The version is more extendable whatever to further 1D,2D or another single cell model. 
# 
# Under Intellectual Property Protection.
# 
# 
# Author      : Shugang Zhang <zhangshugang@hotmail.com>
# Last update : 06-10-2018
# 

#coding=utf-8    
from matplotlib import pyplot as plt    
import numpy as np
import math as math

 
# ------------put your filename here for visulization------------
filename = "../Outputs/Transmural1D_s2@274.50.dat"


	
# -------------read in data------------
# time = np.loadtxt(filename)[:,0].T   # the first colum should be time in my code structure
time = np.arange(0, 999, 1)   # if above time colum is not available, uncomment this line to auto-generate x-axis
data = np.loadtxt(filename)[:,0:].T # the rest columns are membrane potential. T is transverse matrix for better visulization
single = data[15,:] # this is for  singlecell observation, first dimension is the cell index after above transverse


# -------------time information------------
step_num = len(time[:])
print("Total time = %.3f ms" %(time[len(time)-1]))


# -------------space information------------
cell_num = len(data[:,0])
space = np.arange(0, cell_num, 1)  # Space range with space step
print("Cell num =",space.shape[0])
print("Cell num =",time.shape[0])
print("Cell num =",data.shape[0])
print("Cell num =",data.shape[1])

# -------------1D visualization------------
plt.figure(0)
im = plt.contourf(time,space,data,500,cmap=plt.cm.jet)
plt.clim(-81,40)
# plt.contourf(time,space,data,cmap=plt.cm.jet) 
# plt.colorbar(im) # plot the color bar
plt.axis('off')
plt.tight_layout()

# -------------Single cell AP observation------------
plt.figure(1) 
for i in range(10,19):
	plt.plot(time,data[i,:])
plt.show()