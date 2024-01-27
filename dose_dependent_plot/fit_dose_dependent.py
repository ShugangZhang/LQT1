# coding=utf-8
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
# Last update : 15-11-2018
# 



import sys
defaultencoding = 'utf-8'
if sys.getdefaultencoding() != defaultencoding:
    reload(sys)
    sys.setdefaultencoding(defaultencoding)




import numpy as np
import scipy as sp
import scipy.optimize as opt
import matplotlib.pyplot as plt

def useLargeSize(axis,marker_lines = None, fontsize = 18,fontproperties=None):
    '''
      Use larger font for labels, legend of X Y axis
      Bold lines for all
    '''
    axis.spines['top'].set_color('none')
    axis.spines['right'].set_color('none')
    axis.xaxis.get_label().set_size(fontsize)
    axis.yaxis.get_label().set_size(fontsize)
    # Set font size separately for x and y axis
    for label in axis.xaxis.get_ticklabels():
        label.set_fontsize(18)
    for label in axis.yaxis.get_ticklabels():
        label.set_fontsize(18)     
    # Set line width
    LW = 2.3
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









# Pandit model original

		# // L-type Ca2+ current
		# ICaL = gCaL*d*((0.9+Cainact/10.0)*f11+(0.1-Cainact/10.0)*f12)*(Volt-65.0);
		# d_ = 1.0/(1+exp((Volt+15.3)/-5.0));
		# f11_ = 1.0/(1+exp((Volt+26.7)/5.4));
		# f12_ = f11_;
		# Cainact_ = 1.0/(1.0+CaSS/0.01);
		# taud = 0.00305*exp(-0.0045*pow(Volt+7.0,2))+0.00105*exp(-0.002*pow(Volt-18.0,2))+0.00025;
		# tauf11 = 0.105*exp(-pow((Volt+45.0)/12.0,2))
		# 		+0.04/(1.0+exp((-Volt+25.0)/25.0))
		# 		+0.015/(1.0+exp((Volt+75.0)/25.0))
		# 		+0.0017;
		# tauf12 = 0.041*exp(-pow((Volt+47.0)/12.0,2))
		# 		+0.08/(1.0+exp((Volt+55.0)/-5.0))
		# 		+0.015/(1.0+exp((Volt+75.0)/25.0))
		# 		+0.0017;


# dose-dependent



# control activation gate steady state curve

# # dose dependent
#   0.00, 1.00
#  25.00, 0.54
#  50.00, 0.30
#  75.00, 0.27
# 100.00, 0.24
# 200.00, 0.19
# 300.00, 0.14

# dose dependent
#   0.00, 1.00
#  25.00, 0.77
#  50.00, 0.94
#  75.00, 1.02
# 100.00, 1.36
# 200.00, 1.88
# 300.00, 2.03





# Sup
volt = np.array([0.00001, 25.00, 50.00, 75.00, 100.00, 200.00, 300.00])
d_inf = np.array([1.00, 0.54, 0.30, 0.27, 0.24, 0.19, 0.14])
# Rep
shimm = np.array([0.00001, 0.23, 0.64, 0.75, 1.12, 1.69, 1.89])
# Total
total = np.array([1.00, 0.77, 0.94, 1.02, 1.36, 1.88, 2.03])





# so2 activation gate steady state curve
# so2_m_inf = np.array([0.04, 0.16, 0.76, 0.97, 0.97, 0.99])
# temp = np.array([0.05, 0.17, 0.80, 1.02, 1.02, 1.04])
# so2_m_inf_err = np.abs(so2_m_inf - temp)




# Way 1: standard error bar 
plt.figure(1)
# useLargeSize(plt.gca())
plt.xlabel("Dose Transfected (fmol)")
plt.ylabel("Normalized KCNQ1")
# -- Sup --
plt.plot(volt,d_inf,'o',color='black',mec='black',ms = 12,label = "WT-Sup")
# -- Rep --
plt.plot(volt,shimm,'o',color='red',mec='red',ms = 12,label = "ShIMM-Rep")
useLargeSize(plt.gca())
# -- SQT6 --
plt.plot(volt,total,'o',color='green',mec='green',ms = 12,label = "Total-SupRep")
useLargeSize(plt.gca())

# Way 2：Standard for logx  Dont delete！！
# plt.figure(2)
# useLargeSize(plt.gca())
# plt.xlabel("Concentration of SO$_{2}$ derivatives (μmol)")
# plt.ylabel("Increase rate of I$\mathrm{_{CaL}}$(%)")
# plt.semilogx(SO2,incre_rate,'o',ms=5,lw = '2.3',color='black')
# plt.show()





# --------------------------分割线，以下为拟合部分---------------------------

def Boltzmannequation(x, a, b):
    return 1.0/(1.0+np.exp(1.0*(x + a)/b));

def exp_func(x, a, b):
    return a*pow(x,b)
    
def exp_func2(x, c):
    return pow(c,x)

def ln_func(x, a, b, c):
    # return a*np.log(x+b) + c 
    # return a * (1 - np.exp(-2*b*x)) / (1 + np.exp(-2*b*x))
    x = x*b
    return a*(1 - np.exp(-2*x)) / (1 + np.exp(-2*x))

# 这个小细节：x可以是单个数，也可以是数组的形式，如果x是数组就返回所有x对应的hillequation计算结果
# 不要先入为主的认为只能计算单个数
def err(beta, x, y):
    return np.sum(((y[1:] - exp_func(x[1:],*beta))) ** 2)

def err2(beta, x, y):
    return np.sum((([1,0.515] - exp_func2(x[0:2],*beta))) ** 2)

def err3(beta, x, y):
    return np.sum(((y - ln_func(x,*beta))) ** 2)

param_init = np.array([0.5, -0.5]) # test?
param_best = opt.minimize(err,param_init,args=(volt,d_inf)).x

param2_init = np.array([0.9]) # test?
param2_best = opt.minimize(err2,param2_init,args=(volt,d_inf)).x

param3_init = np.array([2,0.01,1]) # test?
param3_best = opt.minimize(err3,param3_init,args=(volt,shimm)).x



xsim = []
ysim = []
print("param_best = ", param_best) 
print("param2_best = ", param2_best)
print("param3_best = ", param3_best)

for x in range(0,300):
    # xsim.append(x)
    # ysim.append(exp_func(x,*param_best))#plot simulated curve
    if x >= 25:
        xsim.append(x)
        ysim.append(exp_func(x,*param_best))#plot simulated curve
    if x < 25:
        xsim.append(x)
        ysim.append(exp_func2(x,*param2_best))

x2sim = []
y2sim = []
for x in range(0,300): 
    x2sim.append(x)       
    y2sim.append(ln_func(x,*param3_best))

# print(x2sim)
# print(y2sim)

# plt.ylim(-0.1,1.1)
# y_ticks = np.arange(0, 1.1, 0.5)
# plt.yticks(y_ticks)
plt.yticks(np.arange(0,2.1,0.5))

# plt.plot(xsim,y2sim,color='black',linewidth=2.3,label='',zorder=0)
plt.plot(xsim,ysim,color='black',linewidth=2.3,label='',zorder=0)
plt.plot(x2sim,y2sim,color='red',linewidth=2.3,label='')
plt.plot(x2sim,np.array(ysim)+np.array(y2sim),color='green',linewidth=2.3,label='')
plt.legend(fontsize = 15, loc = 'best')
plt.tight_layout()
plt.savefig('activation.png',dpi=500)
plt.show()

# fitting result:
# param_best =  [ 3.12921394 -0.56034481]
# param2_best =  [0.97380564]
# param3_best =  [2.01756673 0.00590645 1.        ]