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

import matplotlib as mpl
import matplotlib.pyplot as plt
 
#mpl.use("pgf")
#pgf_with_rc_fonts = {
#    "font.family": "serif",
#    "font.serif": [],                   # use latex default serif font
#    "font.sans-serif": ["DejaVu Sans"], # use a specific sans-serif font
#}
#mpl.rcParams.update(pgf_with_rc_fonts)

mA = 1
G = 39.4769264214
abin = 1

def acrit(ebin,mu_bin):
    ac = 1.60+5.10*ebin+(-2.22)*ebin**2+4.12*mu_bin+(-4.27)*ebin*mu_bin+(-5.09)*mu_bin**2 + 4.61*(ebin**2)*(mu_bin**2)
    return ac

def massB(mu,mA):
    return mu*mA/(1-mu)

def period_ratio(ab,abin):
    return np.sqrt((ab/abin)**3)

validation_set = np.vstack([np.array(map(float, line.split())) for line in open('more_samples.txt')]) 
validation_set = pd.DataFrame(validation_set,columns=['ebin','mubin','ap/abin','out'])
validation_set['binary out'] = np.floor(validation_set['out']) # if less than 1, set to 0
validation_set['param a'] = validation_set['ap/abin']/acrit(validation_set['ebin'],validation_set['mubin']) - 1
validation_set['zeta'] = period_ratio(validation_set['ap/abin'],abin)
validation_set['epsilon'] = 0.5*(validation_set['zeta']-np.floor(validation_set['zeta']))

model = load_model('6layer_48neuron.h5')

x_validation = np.asarray(validation_set[['ebin','mubin','param a','epsilon']])
y_validation = np.asarray(validation_set['binary out'])

start = time.clock()
predictions = []
flags = []
for row in x_validation:
    # only run model on edge cases
    if row[2] >= 0.2:
        predictions.append(1)
        flags.append(1)  # stable by prior
    elif row[2] <= -0.2:
        predictions.append(0)
        flags.append(0)  # unstable by prior
    else:
        pred = model.predict(np.atleast_2d(row))[0][0]
        predictions.append(pred)
        if np.round(pred) == 0:
            flags.append(2)  # unstable by model
        elif np.round(pred) == 1:
            flags.append(3)  # stable by model

end = time.clock()
elapsed = end-start
print ("elapsed: ", elapsed)

predictions = np.asarray(zip(predictions,np.round(predictions),flags))
np.savetxt(r'x_val_6_24.txt', x_validation, fmt='%f')
np.savetxt(r'pred_6_24.txt', predictions, fmt='%f')
np.savetxt(r'y_val_6_24.txt', y_validation, fmt='%f')

