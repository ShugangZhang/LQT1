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
    
   
# ------------put your filename here for visulization------------
filename1 = "../Outputs/VentriSingleCellResults_LQT1_ISO_MCELL.dat"
# filename2 = "VentriSingleCellResults.dat"

data1 = np.loadtxt(filename1)
# data2 = np.loadtxt(filename2)
# data1 = np.loadtxt("EPI.dat")
# data2 = np.loadtxt("MCELL.dat")
# data3 = np.loadtxt("ENDO.dat")


plt.plot(data1[:,1],label = 'Volt')  # 3=>CaL


plt.show()