#!/usr/bin/python3
import numpy as np# matrices
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import sys


x= np.loadtxt("test_pix.txt",dtype=np.float32)
t= np.arange(1,x.shape[0]+1,dtype=np.int32)


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))

f.subplots_adjust(wspace=0.15)
ax1.set_xlabel("frame")
ax1.set_ylabel("x")
h = ax1.axis([0,len(t),0,255])
l, = ax1.plot([],[])



###Dispersion
BUF_SIZE = 11
buf = np.empty([BUF_SIZE],dtype=np.float32)
TICKS = x.shape[0]+1
Dx = np.arange(1,TICKS,dtype=np.int32)

for i in range(len(t)):
        buf[:]=0
        for j in range(-5,6):
            if(i+j>=0 and i+j<len(t)):
                buf[5+j] = x[i+j]
            elif(i+j<0):
                buf[5+j]=x[0]
            else:
                buf[5+j]=x[len(t)-1]
        Dx[i]=np.std(buf)**2


h = ax2.axis([0,len(t),Dx.min(),Dx.max()])
l2, = ax2.plot([],[],color="red",linewidth=1)
ax2.set_xlabel("ticks")
ax2.set_ylabel("Dx")
def animate(i):
    l.set_data(t[:i], x[:i])
    l2.set_data(t[:i],Dx[:i])
for i in range(len(t)):
    animate(i)
    plt.savefig('/tmp/cache_plot_%03d.png'%(i+1))
