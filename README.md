# A Machine Learns to Predict the Stability of Circumbinary Planets

This is a tutorial for using a deep neural network (DNN) to predict the orbital stability of circumbinary planets. The DNN was trained on one million simulations run on [REBOUND](http://rebound.readthedocs.io/en/latest/index.html), a numerical integrator by Hanno Rein et al. You can use our code to generate stability predictions for the circumbinary planetary system of your choice. As a proof of concept, this model is simplified - one can imagine introducing additional parameters, such as orbit inclination. 

To run predictions on the stability of a circumbinary system given binary eccentricity, binary mass ratio (µ), and binary and planet semi-major axes, simply run 'python tatooine.py -a ____ -e ____ -m ____', where the quantity following the -a flag is the ratio of the planet's semi-major axis to the binary semi-major axis; the quantity following -e is the binary eccentricity; and the quantity following -m is the mass ratio. For fully detailed information, please see our paper, "A Machine Learns to Predict the Stability of Circumbinary Planets", out now on MNRAS and [arXiv](https://arxiv.org/abs/1801.03955).

In the near future, we'll also provide the scripts and a tutorial for training your own DNN.

# Tutorial (working, slowly)

Please note some embarrassing or nonsensical comments may have survived in the provided code. Please also note this code was written in Python 2 and before I knew of the existence of PEP-8.

## Generating training data
With observational data from only a handful of circumbinary planets, we elected to generate the training data using the numerical integrator REBOUND. We begin by dropping the µ dimension, holding it constant at 0.1 while still varying semi-major axis, eccentricity, and initial phase. We build the training set with the following script in the /mu_10/ directory.  

```
python staircase_10.py
```

In this script, we first set the hyperparameters of our simulation: here we use 10 phases per draw from [e, a] space and 1000 draws, where e is drawn (naively) uniformly from [0, 0.99] and a is drawn uniformly from a +/- 33% envelope surrounding the output of the function a<sub>crit</sub>(µ,e), described in Holman & Weigert (1999) as a sum of terms of products of µ and e up to the second power. Of course, for now µ is 0.1. For each of these 100000 initial seedings, we use REBOUND's IAS15 integrator to simulate 100000 binary periods. At each timestep during the simulation, REBOUND provides x, y, z, v<sub>x</sub>, v<sub>y</sub>, and v<sub>z</sub>. 

A seeding is labeled unstable if at any timestep v surpasses the escape velocity, or if the sampled initial semi-major axis is less than apoapsis, indicating the possibility of a crossed orbit. The latter case is checked before the costly simulation. If even one of the ten simulations per draw turn out unstable, we call the whole seeding unstable. 

I had output the labeled training data of ten thousand points (1 for stable, 0 for unstable) to .out files in batch jobs run on an HPC, but for this size of data you can certainly output to .txt or .csv files. The rest of this tutorial will assume the latter case and the code has been updated to reflect this.

You can replicate this exercise for different slices in mass ratio space to see how the islands of instability change.


## Building and training µ-constant DNN
We then build and train a neural network on the data with the following script. 

```
python make_model_10.py
```

In this script, we read in the seeding information (e and a expressed as a ratio of planet semi-major axis and binary semi-major axis, which, in another assumption within our simplified model, is 1.0) and fraction of samples per seeding found to be stable. We take the floor of this value to be our stability label. So far, all we've done was build towards a fancier model of what Holman and Wiegert did almost twenty years ago. 

Now we introduce another feature: how far away a planet is from a mean motion resonance. To do this we normalize the semi-major axis ratio against the semi-major axis predicted by the Holman-Wiegert formula given (µ=0.1, e), convert this re-parameterized semi-major axis ratio to a period ratio (ζ) using the Newtonian version of Kepler's third law, then use it to inform what we call the "resonance proxy", ε.

There is more than one way to do this, but here we simply take ε = 0.5 * (ζ - ⌊ζ⌋), where ⌊ζ⌋ is the floor of ζ, representing how far a planet is from the next largest integer. To see how far away a planet is from the next closest integer period ratio, one could also take ε = 0.5 - |ζ - ⌊ζ⌋ - 0.5|. 

We use sklearn to split the training data 0.75/0.25 between training/validation. We use the Keras library to build our deep neural network. Okay, so this is probably why you're here.

Keras helps you abstract away the code and think of DNNs as building blocks of customizable layers. We ended up choosing 6 layers each with 24 neurons and 20% dropout. The intermediary hidden layers use the rectified linear unit (ReLU) activation function, while the final output layer uses the sigmoid activation function. Our optimizer was rmsprop and our loss function was binary crossentropy. Our batch size was 120 and we trained the DNN for 100 epochs (one epoch is a back-and-forth pass through the network). 

We experimented with lots of different setups, and since we settled for the first configuration that outperformed Holman-Wiegert for precision, recall, and accuracy, you could probably beat us! Definitely check out the keras documentation for more ways you can design your neural network. There are certainly tons of hyperparameters to play with.


## Predicting with our trained DNN
Now we can make predictions and plot our results. But in order to do so we first need some test data. So we change the output file name to something like test_mu_10.txt and re-run staircase_10.py. You can change the number of draws for a bigger or smaller test set.

For our paper we used LaTeX fonts, but we're not here to write a paper, so most of the formatting has been removed or commented out. The left panel of Figure 3 in our paper simply shows the unparameterized simulated data from REBOUND.

```
python unparam_simulated.py
```

Now let's actually make our predictions using our trained DNN and plot the results in reparameterized space. 

```
python use_model_10.py
```

All we do here is take test_mu_10.txt as input and, to save resources, run it through some checks rather than run it all through the trained DNN. Only if the sample is seeded within +/- 20% of the Holman-Wiegert critical threshold do we feed it into the model. Note the flags differentiating model- and heuristic-predicted outputs; we eventually elected not to use them but these can be useful when plotting results.  

```
python reparam_preds.py
```

Here we plot the right panel of Figure 4, visualizing the islands of instability, false positives, and false negatives. We can get a better read on our model's performaance by comparing its accuracy, precision, and recall with that of the Holman-Wiegert model for an increasingly narrower band around the critical threshold. Far from this boundary, both models work reasonably well, but as we get towards the resonances and islands, the DNN begins to do much better. We create the plots in Figure 5 with the following code.

```
python figure5.py
```



