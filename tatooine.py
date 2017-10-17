import argparse
import numpy as np 
import pandas as pd 
import keras
from keras.models import load_model
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
#import warnings

parser = argparse.ArgumentParser(description='Predicts stability of circumbinary planet')
parser.add_argument('-a','--ap/abin', type=float, help='Ratio of planet semi-major axis to binary semi-major axis', required=True)
parser.add_argument('-e','--ebin', type=float, help='Eccentricity of the binary', required=True)	
parser.add_argument('-m','--mubin', type=float, help='Mass ratio of the binary', required=True) 

args = vars(parser.parse_args())
a_ratio = args['ap/abin']
e_bin = args['ebin']
mu_bin = args['mubin']

mA = 1
G = 39.4769264214
abin = 1

def acrit(ebin,mubin):
    ac = 1.60+5.10*ebin+(-2.22)*ebin**2+4.12*mubin+(-4.27)*ebin*mubin+(-5.09)*mubin**2 + 4.61*(ebin**2)*(mubin**2)
    return ac

def massB(mu,mA):
    return mu*mA/(1-mu)

def period_ratio(ab,abin):
    return np.sqrt((ab/abin)**3)

# load model
model = load_model('6layer_48neuron.h5')

# transform input to features for DNN
param_a = a_ratio/acrit(e_bin,mu_bin) - 1
zeta = period_ratio(a_ratio,abin)
epsilon = 0.5*(zeta-np.floor(zeta))
features = np.array([[mu_bin,param_a,e_bin,epsilon]])
features_df = pd.DataFrame(features,columns=['mubin','param a','ebin','epsilon'])

# load input to DNN
if features_df['param a'].iloc[0] >= 0.2:
  	pred = 1
elif features_df['param a'].iloc[0] <= -0.2:
   	pred = 0
else:
	pred = model.predict(features)

if round(pred) == 0:
	print "Unstable"
else:
	print "Stable"