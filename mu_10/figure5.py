# code for assessing performance of our models closer and closer to critical threshold

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

#from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
##rc('font',**{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)

# define HW99 function
def acrit(ebin,mubin):
    ac = 1.60+5.10*ebin+(-2.22)*ebin**2+4.12*mubin+(-4.27)*ebin*mubin+(-5.09)*mubin**2 + 4.61*(ebin**2)*(mubin**2)
    return ac

# define function to parameterize a_p
def param_a(ap,ebin,mubin):
    return (ap/acrit(ebin,mubin)) - 1

# define accuracy as total trues over all possibilities
def calculate_accuracy(tp,tn,fp,fn):
    try:
        return float(tp+tn)/(tp+tn+fp+fn)
    except:
        pass

# define precision as true positives divided by total of predicted positives
def calculate_precision(tp,fp):
    try:
        return float(tp)/(tp+fp)
    except:
        pass

# define recall as true positives divided by total of actual positives 
def calculate_recall(tp,fn):
    try:
        return float(tp)/(tp+fn)
    except:
        pass

# validation set
reparam_mu_10 = np.vstack([np.array(map(float, line.split())) for line in open('test_mu_10.txt')])
reparam_mu_10 = pd.DataFrame(reparam_mu_10,columns=['ebin','a','out'])
reparam_mu_10['binary out'] = np.floor(reparam_mu_10['out'])
reparam_mu_10['param a'] = param_a(reparam_mu_10['a'],reparam_mu_10['ebin'],0.1)

# MLP predictions
predictions = np.vstack([np.array(map(float, line.split())) for line in open('slice_pred_6layer_24neuron.txt')])
predictions = pd.DataFrame(predictions,columns=['raw pred mlp','pred mlp','flag'])
# HW99 predictions
predictions['pred hw99'] = np.where((reparam_mu_10['param a'] < 0),0,1)

# put them all together
master = pd.concat([reparam_mu_10,predictions], axis=1)
print (master.head())

accuracies_mlp = []
accuracies_hw99 = []
precisions_mlp = []
precisions_hw99 = []
recalls_mlp = []
recalls_hw99 = []
steps = 100 # or however finely you want to make the plots
max_envelope = np.max(master['param a'])
envelope_sizes = np.linspace(max_envelope,max_envelope/steps,steps)
for envelope in envelope_sizes:
	# keep only those points in test set that fall within this envelope
	survivors = master.loc[np.abs(master['param a']) <= envelope]
 
	# count TPs, TNs, FPs, and FNs for MLP and HW99 models
	fp_mlp = len(survivors[(survivors['pred mlp'] == 1) & (survivors['binary out'] == 0)]['ebin'])
	fn_mlp = len(survivors[(survivors['pred mlp'] == 0) & (survivors['binary out'] == 1)]['ebin'])
	tp_mlp = len(survivors[(survivors['pred mlp'] == 1) & (survivors['binary out'] == 1)]['ebin'])
	tn_mlp = len(survivors[(survivors['pred mlp'] == 0) & (survivors['binary out'] == 0)]['ebin'])

	fp_hw99 = len(survivors[(survivors['pred hw99'] == 1) & (survivors['binary out'] == 0)]['ebin'])
	fn_hw99 = len(survivors[(survivors['pred hw99'] == 0) & (survivors['binary out'] == 1)]['ebin'])
	tp_hw99 = len(survivors[(survivors['pred hw99'] == 1) & (survivors['binary out'] == 1)]['ebin'])
	tn_hw99 = len(survivors[(survivors['pred hw99'] == 0) & (survivors['binary out'] == 0)]['ebin'])

	# calculate accuracy
	accuracy_mlp = calculate_accuracy(tp_mlp,tn_mlp,fp_mlp,fn_mlp)
	accuracies_mlp.append(accuracy_mlp)

	accuracy_hw99 = calculate_accuracy(tp_hw99,tn_hw99,fp_hw99,fn_hw99)
	accuracies_hw99.append(accuracy_hw99)

        # calculate precision
        precision_mlp = calculate_precision(tp_mlp,fp_mlp)
        precisions_mlp.append(precision_mlp)
        
        precision_hw99 = calculate_precision(tp_hw99,fp_hw99)
        precisions_hw99.append(precision_hw99)

        # calculate recall
        recall_mlp = calculate_recall(tp_mlp,fn_mlp)
        recalls_mlp.append(recall_mlp)

        recall_hw99 = calculate_recall(tp_hw99,fn_hw99)
        recalls_hw99.append(recall_hw99)

min_acc_mlp = min(accuracies_mlp)
min_acc_hw99 = min(accuracies_hw99)
min_prec_mlp = min(precisions_mlp)
min_prec_hw99 = min(precisions_hw99)
min_rec_mlp = min(recalls_mlp)
min_rec_hw99 = min(recalls_hw99)

fig = plt.figure()
ax = fig.add_subplot(111)

f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True,figsize=(15,5))

#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

ax1.plot(envelope_sizes,accuracies_mlp,'--',c='k',label='DNN (min acc = %.3f)' % min_acc_mlp)
ax1.plot(envelope_sizes,accuracies_hw99,c='k',label='HW99 (min acc = %.3f)' % min_acc_hw99)
#ax1.plot([0,0.35],[0.9,0.9],'--',c='g',alpha=0.4,linewidth=0.4)
ax1.set_ylabel(r'accuracy',fontsize=15)
#ax1.set_xlabel(r'\|[(a_p/a_{bin})/a_{HW99}]-1\|',fontsize=15)
ax1.set_xlabel(r'envelope size',fontsize=15)
ax1.legend(loc='lower right',fontsize=16)
#ax1.text(0.15, 0.905, "{90$\%$ accuracy",fontsize=10,color='g',style='italic')

ax2.plot(envelope_sizes,precisions_mlp,'--',c='k',label='DNN (min prec = %.3f)' % min_prec_mlp)
ax2.plot(envelope_sizes,precisions_hw99,c='k',label='HW99 (min prec = %.3f)' % min_prec_hw99)
ax2.set_ylabel(r'precision',fontsize=15)
ax2.set_xlabel(r'envelope size',fontsize=15)
ax2.legend(loc='lower right',fontsize=16)

ax3.plot(envelope_sizes,recalls_mlp,'--',c='k',label='DNN (min rec = %.3f)' % min_rec_mlp)
ax3.plot(envelope_sizes,recalls_hw99,c='k',label='HW99 (min rec = %.3f)' % min_rec_hw99)
ax3.set_ylabel(r'recall',fontsize=15)
ax3.set_xlabel(r'envelope size',fontsize=15)
ax3.legend(loc='lower right',fontsize=16)

#plt.savefig('figure5.pdf')
plt.show()
