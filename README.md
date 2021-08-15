# [OILMM](http://github.com/wesselb/oilmm)

[![CI](https://github.com/wesselb/oilmm/workflows/CI/badge.svg?branch=master)](https://github.com/wesselb/oilmm/actions?query=workflow%3ACI)
[![Coverage Status](https://coveralls.io/repos/github/wesselb/oilmm/badge.svg?branch=master)](https://coveralls.io/github/wesselb/oilmm?branch=master)
[![Latest Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://wesselb.github.io/oilmm)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Implementation of the Orthogonal Instantaneous Linear Mixing Model

Contents:
* [Requirements and Installation](#reproduce-experiments)
* [TLDR](#tldr)
* [Basic Usage](#basic-usage)
  * [Examples of Latent Process Models](#example)
    * [Smooth Processes](#smooth-processes)
    * [Smooth Processes with a Rational Quadratic Kernel](#smooth-processes-with-a-rational-quadratic-kernel) 
    * [Weakly Periodic Processes](#weakly-periodic-processes)
    * [Bayesian Linear Regression](#bayesian-linear-regression)
* [Advanced Usage](#basic-usage)
  * [Use the OILMM Within Your Model](#use-the-oilmm-within-your-model)
  * [Kronecker-Structured Mixing Matrix](#kronecker-structured-mixing-matrix)
* [Reproduce Experiments From the Paper](#reproduce-experiments-from-the-paper)

## Requirements and Installation

See [the instructions here](https://gist.github.com/wesselb/4b44bf87f3789425f96e26c4308d0adc).
Then simply

```bash
pip install oilmm
```

## TLDR

```python
import numpy as np
from stheno import EQ, GP

# Use TensorFlow as the backend for the OILMM.
import tensorflow as tf
from oilmm.tensorflow import OILMM


def build_latent_processes(ps):
    # Return models for latent processes, which are noise-contaminated GPs.
    return [
        (
            p.variance.positive(1) * GP(EQ().stretch(p.length_scale.positive(1))),
            p.noise.positive(1e-2),
        )
        for p, _ in zip(ps, range(3))
    ]

# Construct model.  
prior = OILMM(tf.float32, build_latent_processes, num_outputs=6)

# Create some sample data.
x = np.linspace(0, 10, 100)
y = prior.sample(x)  # Sample from the prior.

# Fit the model to the data.
prior.fit(x, y, trace=True, jit=True)
prior.vs.print()  # Print all learned parameters.

# Make predictions.
posterior = prior.condition(x, y)  # Construct posterior model.
mean, var = posterior.predict(x)  # Predict with the posterior model.
lower = mean - 1.96 * np.sqrt(var)
upper = mean + 1.96 * np.sqrt(var)
```

```
Minimisation of "negative_log_marginal_likelihood":
    Iteration 1/1000:
        Time elapsed: 0.9 s
        Time left:  855.4 s
        Objective value: -0.1574
    Iteration 105/1000:
        Time elapsed: 1.0 s
        Time left:  15.5 s
        Objective value: -0.5402
    Done!
Termination message:
    CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH
latent_processes.processes[0].variance: 1.829
latent_processes.processes[0].length_scale: 1.078
latent_processes.processes[0].noise: 9.979e-03
latent_processes.processes[1].variance: 1.276
latent_processes.processes[1].length_scale: 0.9262
latent_processes.processes[1].noise: 0.03924
latent_processes.processes[2].variance: 1.497
latent_processes.processes[2].length_scale: 1.092
latent_processes.processes[2].noise: 0.04833
mixing_matrix.u:
    (6x3 array of data type float32)
    [[ 0.543 -0.237 -0.111]
     [ 0.578 -0.185 -0.357]
     [-0.204 -0.094 -0.567]
     [-0.554 -0.413 -0.081]
     [-0.12   0.571 -0.66 ]
     [-0.089 -0.636 -0.31 ]]
noise:      0.02245
```

## Basic Usage

### Examples of Latent Process Models

#### Smooth Processes
```python
from stheno import GP, EQ

def build_latent_processes(ps):
    return [
        (
            p.variance.positive(1) * GP(EQ().stretch(p.length_scale.positive(1))),
            p.noise.positive(1e-2),
        )
        for p, _ in zip(ps, range(3))
    ]
```

#### Smooth Processes With A Rational Quadratic Kernel
```python
from stheno import GP, RQ

def build_latent_processes(ps):
    return [
        (
            p.variance.positive(1)
            * GP(RQ(p.alpha.positive(1e-2)).stretch(p.length_scale.positive(1))),
            p.noise.positive(1e-2),
        )
        for p, _ in zip(ps, range(3))
    ]
```

#### Weakly Periodic Processes
```python
from stheno import GP, EQ

def build_latent_processes(ps):
    return [
        (
            p.variance.positive(1)
            * GP(
                # Periodic component:
                EQ()
                .stretch(p.periodic.length_scale.positive(0.7))
                .periodic(p.periodic.period.positive(24))
                # Make the periodic component slowly change over time:
                * EQ().stretch(p.periodic.decay.positive(72))
            ),
            p.noise.positive(1e-2),
        )
        for p, _ in zip(ps, range(3))
    ]
```

#### Bayesian Linear Regression
```python
from stheno import GP, Linear

num_features = 10


def build_latent_processes(ps):
    return [
        (
            GP(Linear().stretch(p.length_scales.positive(1, shape=(num_features,)))),
            p.noise.positive(1e-2),
        )
        for p, _ in zip(ps, range(3))
    ]
```

## Advanced Usage


### Use the OILMM Within Your Model

### Kronecker-Structured Mixing Matrix
```python
from matrix import Kronecker

p_left, m_left = 10, 3  # Shape of left factor in Kronecker product
p_right, m_right = 5, 2  # Shape of right factor in Kronecker product


def build_mixing_matrix(ps, p, m):
    return Kronecker(
        ps.left.orthogonal(shape=(p_left, m_left)),
        ps.right.orthogonal(shape=(p_right, m_right)),
    )


prior = OILMM(
    dtype,
    latent_processes=build_latent_processes,
    mixing_matrix=build_mixing_matrix,
    num_outputs=p_left * p_right
)
```

## Reproduce Experiments From the Paper

*TODO:* Install requirements.

Scripts to rerun individual experiments from the paper can be found in the
`experiments` folder.
A shell script is provided to rerun all experiments from the paper at once:

```bash
sh run_experiments.sh
```

The results can then be found in the generated `_experiments` folder.

