#coding=utf-8    
from matplotlib import pyplot as plt  
from matplotlib.pyplot import MultipleLocator  
import numpy as np
    
 


def useLargeSize(axis,marker_lines = None, fontsize = 'xx-large',fontproperties=None):
    '''
      将X,Y坐标轴的标签、刻度以及legend都使用大字体,
      所有线条采用粗线
    '''
    axis.spines['top'].set_color('none')
    axis.spines['right'].set_color('none')
    axis.spines['bottom'].set_position(('data',0))
    axis.spines['left'].set_position(('data',0))
    axis.spines['left'].set_linewidth(1.5)
    # axis.xaxis.set_label_coords(0.5,-0.05)
    # axis.yaxis.set_label_coords(0.4,0.6)
    axis.xaxis.get_label().set_size(18)
    axis.yaxis.get_label().set_size(18)

    #分别设置x轴和y轴上刻度值的字体大小
    for label in axis.xaxis.get_ticklabels():
        label.set_fontsize(18)
    for label in axis.yaxis.get_ticklabels():
        label.set_fontsize(18) 
    # 设置线的粗细
    LW = 2.
    for line in axis.get_lines():
        line.set_lw( LW )
    leg = axis.get_legend()
    if(leg):
        ltext  = leg.get_texts()  # all the text.Text instance in the legend
        if(fontproperties):
            plt.setp(ltext, fontproperties=fontproperties)        
        plt.setp(ltext, fontsize='x-large')
        llines = leg.get_lines()  # all the lines.Line2D instance in the legend
        plt.setp(llines,linewidth= LW )
        if(marker_lines and len(marker_lines)>=len(llines)):
            for i in range(0,len(llines)):
                plt.setp(llines[i], 
                    marker = marker_lines[i].get_marker(), 
                    markeredgecolor= marker_lines[i].get_markeredgecolor(),\
                    markerfacecolor= marker_lines[i].get_markerfacecolor(),\
                    markeredgewidth= marker_lines[i].get_markeredgewidth(),
                    markersize= marker_lines[i].get_markersize() )




# filename = "Outputs/VentOneDResults_BASELINE.dat"
filename = "../Outputs/VentOneDResults_CON_ISO.dat"
filename2 = "../Outputs/VentOneDResults_CON_ISO.dat"
BCL = 1000  # this is to determine the time shift
data = np.loadtxt(filename).T[1:,:]
data2 = np.loadtxt(filename2).T[1:,:]
time = np.loadtxt(filename).T[0,:]  # 读不同的文件，加载进来的time是不一样的，这里要注意 
print(data.shape) # 2001,100


step_num = len(data[0,:]) # time
cell_num = len(data[:,0])


space = np.arange(0, cell_num, 1)       # Space range with space step
print(space.shape)



plt.figure(10,figsize=(10,3))
ax1 = plt.gca()
# plt.xlim(0,600) # Glory
plt.ylim(-0.05,0.3)
plt.ylabel('Φ (mV)')
ax1.yaxis.set_major_locator(MultipleLocator(0.1))
# dx = 0.15e-3 #m
# x0 = 20e-3 #m
# alpha = 11e-6 #m

dx = 0.15 #mm
x0 = 20+15 #mm
alpha = 11e-3 #mm

ECGx = []
ECGy = []
ECGy2 = []
for t in range(0,step_num,1):
	fai = 0
	fai2 = 0
	for i in range(0,cell_num,1):
		if(i == 0):
			fai += -(data[i+1,t] - data[i,t])/((i*dx-x0)*(i*dx-x0))
			fai2 += -(data2[i+1,t] - data2[i,t])/((i*dx-x0)*(i*dx-x0))
		elif(i == cell_num - 1):
			fai += -(data[i,t] - data[i-1,t])/((i*dx-x0)*(i*dx-x0))
			fai2 += -(data2[i,t] - data2[i-1,t])/((i*dx-x0)*(i*dx-x0))
		else:
			fai += -(data[i+1,t] - data[i-1,t])/(2*(i*dx-x0)*(i*dx-x0))
			fai2 += -(data2[i+1,t] - data2[i-1,t])/(2*(i*dx-x0)*(i*dx-x0))
	# ECGx.append(time[t]-BCL*10+100)   # -------------------- time changed ------------------------
	ECGx.append(time[t]) 
	ECGy.append(fai)
	ECGy2.append(fai2)

ax1.plot(ECGx[:],ECGy[:],'-',color='black',label='Wild-type')
ax1.plot(ECGx[:],ECGy2[:],'--',color='red',label='Y171X')
# ax1.spines['top'].set_visible(False)
# ax1.spines['right'].set_visible(False)
# ax1.spines['bottom'].set_visible(False)
# ax1.set_xticks([])

# set range
# ax1.set_yticks(range(-0.2,0.31,0.1))
# change label fontsize


useLargeSize(plt.gca())

labels = ax1.get_xticklabels()+ax1.get_yticklabels()
[label.set_fontsize(16) for label in labels]
plt.legend(fontsize = 16, loc = (0.08,0.7))

# plt.xlim(15000,18000)  # for enhanced ISO
# plt.axis('off')
plt.tight_layout()
plt.savefig('ECG_1.6_1.png',dpi=300)
plt.show()
