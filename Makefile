#
# General Code Structure (GCS) for Makefile
# 
# A. Code sructure is re-organised using standard C++ to fit into my project.
# B. The friendly style makes the project easier understood.
# C. The version is more extendable whatever to further 1D,2D or another single cell model. 
# 
# Under Intellectual Property Protection.
# 
# 
# Author      : Shugang Zhang <zsg@ouc.edu.cn>
# Last update : 27-01-2024
# 

all: SingleCell OenD OneD_VW TwoD_real 3D

common  =   SingleCell/Cell.cc   SingleCell/TPORd.cc

CC  	=	g++

CFLAGS	=	-w -O3 #-g:warning output to screen   -w:warning ignored

CFLAGS2	=	-fopenmp

CFLAGS3 =   -arch sm_61 -Xptxas -dlcm=cg


SingleCell: $(common) SingleCell.cc
	$(CC) $(CFLAGS) -o model_single_cell $(common) SingleCell.cc

OneD: $(common) OneD.cc
	$(CC) $(CFLAGS) $(CFLAGS2) -o model_oned $(common) OneD.cc

OneD_VW: $(common) OneD_VW.cc
	$(CC) $(CFLAGS) $(CFLAGS2) -o model_oned_vw $(common) OneD_VW.cc

TwoD_real: new_11.cu
	nvcc $(CFLAGS) $(CFLAGS3) -o model_twod_real new_11.cu

3D: GPU3D.cu
	nvcc $(CFLAGS) $(CFLAGS3) -o model_3D GPU3D.cu

clean:
	rm model_*
