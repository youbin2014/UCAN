import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker



runtime=np.load('./runtime.npy')
runtime[0]=runtime[1]
x=[]
for i in range(51):
    i = i + 10
    x.append(i / 10)

fig, ax=plt.subplots()
plt.rcParams["font.family"] = "Times New Roman"
plt.xlabel('Input Dimension',fontsize=15)
plt.ylabel('Certification Time (s)',fontsize=15)
ax.plot(x,runtime,linewidth=2,c='coral')
ax.set_xticks([1,2,3,4,5,6])
ax.set_xticklabels([r'$10^1$',r'$10^2$',r'$10^3$',r'$10^4$',r'$10^5$',r'$10^6$'],fontsize=15)
plt.grid(True)
plt.tight_layout()
plt.show()
