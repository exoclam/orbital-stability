# A Machine Learns to Predict the Stability of Circumbinary Planets

This is a tutorial for using a deep neural network (DNN) to predict the orbital stability of circumbinary planets. The DNN was trained on one million simulations run on [REBOUND](http://rebound.readthedocs.io/en/latest/index.html), a numerical integrator by Hanno Rein et al. You can use our code to generate stability predictions for the circumbinary planetary system of your choice. As a proof of concept, this model is simplified - one can imagine introducing additional parameters, such as orbit inclination. 

To run predictions on the stability of a circumbinary system given binary eccentricity, binary mass ratio (µ), and binary and planet semi-major axes, simply run 'python tatooine.py -a ____ -e ____ -m ____', where the quantity following the -a flag is the ratio of the planet's semi-major axis to the binary semi-major axis; the quantity following -e is the binary eccentricity; and the quantity following -m is the mass ratio. For fully detailed information, please see our paper, "A Machine Learns to Predict the Stability of Circumbinary Planets", out now on MNRAS and [arXiv](https://arxiv.org/abs/1801.03955).

In the near future, we'll also provide the scripts and a tutorial for training your own DNN.

# Tutorial (working, slowly)

Please note some embarrassing or nonsensical comments may have survived in the provided code. Please also note this code was written before I knew of the existence of PEP-8.

## Generating training data
With observational data from only a handful of circumbinary planets, we elected to generate the training data using the numerical integrator REBOUND. We begin by dropping the µ dimension, holding it constant at 0.1 while still varying semi-major axis, eccentricity, and initial phase. We build the training set with the following script in the /mu_10/ directory.  

```
python staircase_10.py
```

In this script, we first set the hyperparameters of our simulation: here we use 10 phases per draw from [e, a] space and 1000 draws, where e is drawn (naively) uniformly from [0, 0.99] and a is drawn uniformly from a +/- 33% envelope surrounding the output of the function a<sub>crit</sub>(µ,e), described in Holman & Weigert (1999) as a sum of terms of products of µ and e up to the second power. Of course, for now µ is 0.1. For each of these 10000 initial seedings, we use REBOUND's IAS15 integrator to simulate 10000 binary periods. At each timestep during the simulation, REBOUND provides x, y, z, v<sub>x</sub>, v<sub>y</sub>, and v<sub>z</sub>. 

A seeding is labeled unstable if at any timestep v surpasses the escape velocity, or if the sampled initial semi-major axis is less than apoapsis, indicating the possibility of a crossed orbit. The latter case is checked before the costly simulation. If even one of the ten simulations per draw turn out unstable, we call the whole seeding unstable. I output the labeled training data of ten thousand points (1 for stable, 0 for unstable) to .out files in batch jobs run on Columbia's Habanero HPC, but if you are not as blessed as I was, for this size of data you can output to .txt or .csv files. This may be an issue for the full battery of µ later though. 

You can replicate this exercise for different slices in mass ratio space to see how the islands of instability change, and also to get more pretty plots. More on plots later.


## Building and training µ-constant DNN
We then build and train a neural network on the data with the following script. 

```
python make_model_10.py
```

In this script, we read in the seeding information (e and a expressed as a ratio of planet semi-major axis and binary semi-major axis, which, in another assumption within our simplified model, is 1.0) and fraction of samples per seeding found to be stable. We take the floor of this value to be our stability label. So far, all we've done was build towards a fancier model of what Holman and Wiegert did almost twenty years ago. 

Now we introduce another feature: how far away a planet is from a mean motion resonance. To do this we normalize the semi-major axis ratio against the semi-major axis predicted by the Holman-Wiegert formula given (µ=0.1, e), convert this re-parameterized semi-major axis ratio to a period ratio (ζ) using the Newtonian version of Kepler's third law, then use it to inform what we call the "resonance proxy". 

There is more than one way to do this, but here we simply take 0.5 * (ζ - ⌊ζ⌋), where ⌊ζ⌋ is the floor of ζ, representing how far a planet is from the next largest integer. To see how far away a planet is from the next closest intege ζ, one could also take 0.5 - |ζ - ⌊ζ⌋ - 0.5|. We call the "resonance proxy" ε. 

