# https://arxiv.org/pdf/1610.05359.pdf Tamayo 2016 on using machine learning to predict orbital stability
import numpy as np
import rebound
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import math
from mpl_toolkits.mplot3d import Axes3D

# calculate a_crit using equation for Holman line
def acrit(ebin,mu_bin):
    ac = 1.60+5.10*ebin+(-2.22)*ebin**2+4.12*mu_bin+(-4.27)*ebin*mu_bin+(-5.09)*mu_bin**2 + 4.61*(ebin**2)*(mu_bin**2)
    return ac

# re-parameterization of test set
reparam_mu_10 = np.vstack([np.array(map(float, line.split())) for line in open('test_mu_10.txt')])
reparam_mu_10 = pd.DataFrame(reparam_mu_10,columns=['ebin','a','out'])
reparam_mu_10['binary out'] = np.floor(reparam_mu_10['out'])
mu = 0.1
reparam_mu_10['param a'] = reparam_mu_10['a']/acrit(reparam_mu_10['ebin'],mu) - 1

# get ground truth of edge cases only because we're mainly interested in running the model on islands
data1 = reparam_mu_10[(reparam_mu_10['param a'] <= 0.2) & (reparam_mu_10['param a'] >= -0.2)]
data1_stable = data1[data1['binary out'] == 1.0]

f, ax1 = plt.subplots(1, 1)
ax1.set_facecolor('grey')
ebins = np.linspace(0,0.99,1000)

# fill in parameter space for which we don't need a DNN to know stability
ax1.fill_between(ebins, acrit(ebins,mu)*1.333, acrit(ebins,mu)*0.667, where= acrit(ebins,mu)*0.667 <= acrit(ebins,mu)*1.333, facecolor='white', interpolate=True) # white fill for instability, save the time-intensive scatter plotting for DNN
ax1.fill_between(ebins, acrit(ebins,mu)*1.333, acrit(ebins,mu)*1.19, where= acrit(ebins,mu)*1.19 <= acrit(ebins,mu)*1.333, facecolor='black', interpolate=True) # ditto for stable fill

# plot ground truth as provided by REBOUND; note that I'm cheating for time by plotting only stable points (black) because the background fill is already white
ax1.scatter(data1_stable['ebin'],data1_stable['a'],c='k',s=5)
ax1.plot(ebins,acrit(ebins,mu),c='cyan',linestyle='dashed') # analytic Holman-Wiegert line
ax1.plot(ebins,acrit(ebins,mu)*(1.333),c='cyan') # upper envelope
ax1.plot(ebins,acrit(ebins,mu)*(0.667),c='cyan') # lower envelope
#ax1.text(0.425, 2.0, "$HW99 criterion +/-33{\%}$",fontsize=16,color='cyan')
ax1.set_xlabel('$e_{bin}$',fontsize=22)
ax1.set_ylabel('$a_{p}/a_{bin}$',fontsize=22)
ax1.set_xlim([0,0.99])
ax1.set_ylim([min((reparam_mu_10['param a']+1)*acrit(reparam_mu_10['ebin'],0.1)),max((reparam_mu_10['param a']+1)*acrit(reparam_mu_10['ebin'],0.1))])
#plt.savefig('figure3_unparam.pdf')
plt.show()


