import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.models import load_model
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import time
import pydot
import graphviz
from keras.utils import plot_model
import matplotlib as mpl
mpl.use("pgf")
pgf_with_rc_fonts = {
    "font.family": "serif",
    "font.serif": [],                   # use latex default serif font
    "font.sans-serif": ["DejaVu Sans"], # use a specific sans-serif font
}
mpl.rcParams.update(pgf_with_rc_fonts)
import matplotlib.pyplot as plt

mA = 1
G = 39.4769264214
abin = 1

def acrit(ebin,mu_bin):
    ac = 1.60+5.10*ebin+(-2.22)*ebin**2+4.12*mu_bin+(-4.27)*ebin*mu_bin+(-5.09)*mu_bin**2 + 4.61*(ebin**2)*(mu_bin**2)
    return ac

def massB(mu,mA):
    return mu*mA/(1-mu)

def period_ratio(mA,mB,ab,abin):
    return np.sqrt((4 * (ab/abin)**3 * np.pi**2)/(G*(mA+mB)))

validation_set = np.vstack([np.array(map(float, line.split())) for line in open('kepler16_apoverabin_ebin.dat')])
validation_set = pd.DataFrame(validation_set,columns=['ebin','mubin','a','out'])
validation_set['binary out'] = np.floor(validation_set['out'])
validation_set['param a'] = validation_set['a']/acrit(validation_set['ebin'],validation_set['mubin']) - 1
period_ratios = period_ratio(mA,massB(validation_set['mubin'],mA),validation_set['a'],abin)
validation_set['period ratio'] = np.asarray(0.5*(period_ratios-np.floor(period_ratios)))

model = load_model('model_100.h5')

x_validation = np.asarray(validation_set[['ebin','mubin','param a','period ratio']])
y_validation = np.asarray(validation_set['binary out'])

start = time.clock()
predictions = []
flags = []
for row in x_validation:
    if row[2] >= 0.2: # selected because of work in Shrink Wrap notebook
        predictions.append(1)
        flags.append(1)
    elif row[2] <= -0.2:
        predictions.append(0)
        flags.append(0)
    else:
        pred = model.predict(np.atleast_2d(row))[0][0]
        predictions.append(pred)
        if np.round(pred) == 0:
            flags.append(2)
        elif np.round(pred) == 1:
            flags.append(3)

end = time.clock()
elapsed = end-start
print ("elapsed: ", elapsed)

predictions = np.asarray(zip(predictions,np.round(predictions),flags))
#np.savetxt(r'/Users/coolworlds/Desktop/Orbital Stability/x_val_100epoch_10mu.txt', x_validation, fmt='%f')
#np.savetxt(r'/Users/coolworlds/Desktop/Orbital Stability/pred_100epoch_10mu.txt', predictions, fmt='%f')
#np.savetxt(r'/Users/coolworlds/Desktop/Orbital Stability/y_val_100epoch_10mu.txt', y_validation, fmt='%f')

n_classes = 2
fpr = dict()
tpr = dict()
roc_auc = dict()
lw = 2
precision_mlp, recall_mlp, thresholds_mlp = precision_recall_curve(y_validation,predictions[:,0])
# MLP precision-recall curve                                                                                                                   
plt.plot(recall_mlp,precision_mlp,lw=lw,color='k',label='MLP (area = %0.3f)' % auc(recall_mlp,precision_mlp))

# Holman line precision-recall curve                                                                                                           
holman_predictions = np.where((validation_set['param a'] < 0),0,1)
precision_holman, recall_holman, thresholds_holman = precision_recall_curve(y_validation,holman_predictions)
plt.plot(recall_holman,precision_holman,lw=lw,color='k',ls='dashed',label='Holman-Wiegert (1999) (area = %0.3f)' % auc(recall_holman,precision_holman))
#plot_model(model, to_file='model_100.pdf')                                                                                                    

plt.legend(loc="lower left",fontsize=16)
plt.savefig('precision-recall-on-k16b.pdf')

quit()
