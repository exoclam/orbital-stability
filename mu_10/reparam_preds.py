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

# read in predictions and x_vals associated with those predictions
columns = ['mubin','param a','ebin','zeta']
features_10 = np.vstack([np.array(map(float, line.split())) for line in open('slice_x_val_6layer_24neuron.txt')])
df_features_10 = pd.DataFrame(features_10,columns=columns)
predictions_10 = np.vstack([np.array(map(float, line.split())) for line in open('slice_pred_6layer_24neuron.txt')])
df_predictions_10 = pd.DataFrame(predictions_10,columns=['mlp out','mlp binary out','flags'])
master_10 = pd.concat([df_features_10,df_predictions_10], axis=1)

# get only edge cases for plotting
data2 = master_10[(master_10['param a'] <= 0.2) & (master_10['param a'] >= -0.2)]
data2_stable = data2[data2['mlp binary out'] == 1.0]
data2_unstable = data2[data2['mlp binary out'] == 0.]

f, ax2 = plt.subplots(1, 1)
ax2.set_facecolor('white')
ebins = np.linspace(0,1,100)
# pretty hacky, please forgive
def point_four():
    return 0.4
def point_two():
    return 0.2

ax2.fill_between(ebins, point_four(), point_two(), where=point_four()>=point_two(), interpolate=True, color='black')
ax2.scatter(data2_stable['ebin'],data2_stable['param a'],c='k',s=5,label='_nolegend_')
ax2.text(0.05, -0.25, "$DNN   for   \mu = 0.1$",fontsize=18)

false_positives_x = master_10[(master_10['mlp binary out'] == 1) & (reparam_mu_10['binary out'] == 0)]['ebin']
false_negatives_x = master_10[(master_10['mlp binary out'] == 0) & (reparam_mu_10['binary out'] == 1)]['ebin']
false_positives_y = master_10[(master_10['mlp binary out'] == 1) & (reparam_mu_10['binary out'] == 0)]['param a']
false_negatives_y = master_10[(master_10['mlp binary out'] == 0) & (reparam_mu_10['binary out'] == 1)]['param a']

ax2.scatter(false_positives_x,false_positives_y,c='r',s=5,alpha=0.2,label='$FPs$')
ax2.scatter(false_negatives_x,false_negatives_y,c='darkorchid',s=5,alpha=0.2,label='$FNs$')
ax2.set_xlabel('$e_{bin}$',fontsize=22)
ax2.set_ylabel('$(a_{p}/a_{bin})/a_{HW99} - 1$',fontsize=22)
ax2.set_xlim([0,0.99])
ax2.set_ylim([min(master_10['param a']),max(master_10['param a'])])
leg = ax2.legend(frameon=True,loc=(0.75,0.8),markerscale=5,fontsize=14)

#plt.savefig('figure3_param.pdf')
plt.show()


