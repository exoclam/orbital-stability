# A Machine Learns to Predict the Stability of Tatooines

This is a tutorial for using a multi-layer perceptron (MLP) to predict the orbital stability of circumbinary planets. The MLP was trained on one million simulations run on [REBOUND](http://rebound.readthedocs.io/en/latest/index.html), a numerical integrator by Hanno Rein et al. Our paper is forthcoming, but for now feel free to play around with the codes and generate class probability predictions for the orbital stability of the circumbinary planetary system of your choice. 

Here, we've decided to use simulated posteriors from [Kepler-16b](https://arxiv.org/pdf/1109.3432.pdf). To reproduce our results, simply run the precision_recall_on_k16.py script, which will apply the previously trained model (model_100.h5) on the posteriors. The script will output precision and recall values that you can then plot to visualize the MLP's performance. For fully detailed information, please see our paper, "A Machine Learns to Predict the Stability of Tatooines", coming soon.

In the near future, we'll also provide the scripts for training your own MLP.