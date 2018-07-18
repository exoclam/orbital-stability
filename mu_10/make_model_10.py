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

batch_size = 128
epochs = 100

# calculate a_crit using equation for Holman line                                                                                   # source: Holman & Wiegert, 1998, https://arxiv.org/pdf/astro-ph/9809315.pdf                                                         
def acrit(ebin,mu_bin):
    ac = 1.60+5.10*ebin+(-2.22)*ebin**2+4.12*mu_bin+(-4.27)*ebin*mu_bin+(-5.09)*mu_bin**2 + 4.61*(ebin**2)*(mu_bin**2)
    return ac

def massB(mu,mA):
    return mu*mA/(1-mu)

#def period_ratio(mA,mB,ab,abin):
#    return np.sqrt((4 * (ab/abin)**3 * np.pi**2)/(G*(mA+mB)))

def period_ratio(ab,abin):
    return np.sqrt((ab/abin)**3)

mA = 1
G = 39.4769264214
abin = 1
mu = 0.1

columns = ['ebin', 'ap', 'out']
big_job = np.vstack([np.array(map(float, line.split())) for line in open('train_mu_10.txt')])
big_batch = pd.DataFrame(big_job,columns=columns)
big_batch['mubin'] = mu
big_batch['(ap/abin)/ahw99 - 1'] = big_batch['ap']/acrit(big_batch['ebin'],big_batch['mubin']) - 1.  
big_batch['zeta'] = period_ratio(big_batch['ap'],abin)
big_batch['epsilon'] = np.asarray(0.5*(big_batch['zeta'] - np.floor(big_batch['zeta'])))
big_batch['binary out'] = np.floor(big_batch['out'])

X = np.asarray(big_batch[['mubin','(ap/abin)/ahw99 - 1','ebin','epsilon']])
y = np.asarray(big_batch[['binary out']]) # choose not the direct output but the binary version of it                                
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

start = time.clock()

model = Sequential()
model.add(Dense(24, activation='relu', input_shape=(4,)))
model.add(Dropout(0.2))
model.add(Dense(24, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(24, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(24, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(24, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(24, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid')) # sigmoid/logistic function simpler than softmax                                           

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0],score[0].shape)
print('Test accuracy:', score[1], score[1].shape)

end = time.clock()
print(end-start)

# Saving the model                                                                                                                   
model.save('dropout_6layer_24neuron.h5')
