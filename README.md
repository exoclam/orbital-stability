# A Machine Learns to Predict the Stability of Circumbinary Planets

This is a tutorial for using a deep neural network (DNN) to predict the orbital stability of circumbinary planets. The DNN was trained on one million simulations run on [REBOUND](http://rebound.readthedocs.io/en/latest/index.html), a numerical integrator by Hanno Rein et al. You can use our code to generate stability predictions for the circumbinary planetary system of your choice. As a proof of concept, this model is simplified - one can imagine introducing additional parameters, such as orbit inclination. 

To run predictions on the stability of a circumbinary system given binary eccentricity, binary mass ratio, and binary and planet semi-major axes, simply run 'python tatooine.py -a ____ -e ____ -m ____', where the quantity following the -a flag is the ratio of the planet's semi-major axis to the binary semi-major axis; the quantity following -e is the binary eccentricity; and the quantity following -m is the mass ratio. For fully detailed information, please see our paper, "A Machine Learns to Predict the Stability of Circumbinary Planets", out now on MNRAS and [arXiv](https://arxiv.org/abs/1801.03955).

In the near future, we'll also provide the scripts and a tutorial for training your own DNN.

# Tutorial (working, slowly)

## Generating training data
With observational data from only a handful of circumbinary planets, we elected instead to generate the training data using the numerical integrator REBOUND. We begin by dropping the mu dimension, holding it constant at 0.1 while still varying semi-major axis, eccentricity, and initial phase. We build the training set with the following script in /mu_10/.  

```
python staircase_10.py
```

In this script, we first set the hyperparameters of our simulation: here we use 10 phases per orientation and 1000 draws from [] space.

We then build and train a neural network on the data with the following script. 

```
python model_10.py
```
 
