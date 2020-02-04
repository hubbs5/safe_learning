#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import gpflow
from scipy import linalg
import matplotlib.pyplot as plt
from matplotlib import colors
print(tf.__version__)

import safe_learning
import plotting
np.random.seed(0)

try:
    session.close()
except NameError:
    pass

session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())

# x_min, x_max, discretization\
state_limits = np.array([[-1., 1.]])
action_limits = np.array([[-.5, .5]])
num_states = 1000
num_actions = 101

safety_disc = safe_learning.GridWorld(state_limits, num_states)

# Discretization for optimizing the policy (discrete action space)
# This is not necessary if one uses gradients to optimize the policy
action_disc = safe_learning.GridWorld(action_limits, num_actions)

# Discretization constant
tau = np.max(safety_disc.unit_maxes)

# Initial policy: All zeros
policy_disc = safe_learning.GridWorld(state_limits, 51)
policy = safe_learning.Triangulation(policy_disc, np.zeros(len(policy_disc)), name='policy')

print('Grid size: {0}'.format(len(safety_disc)))

# GP Dynamics model

kernel = (gpflow.kernels.Matern32(2, lengthscales=1, active_dims=[0, 1]) *
          gpflow.kernels.Linear(2, variance=[0.2, 1], ARD=True))

noise_var = 0.01 ** 2

# Mean dynamics
mean_function = safe_learning.LinearSystem(([1, 0.1]), name='prior_dynamics')

mean_lipschitz = 0.8
gp_lipschitz = 0.5 # beta * np.sqrt(kernel.Mat32.variance) / kernel.Mat32.lengthscale * np.max(np.abs(state_limits))
lipschitz_dynamics = mean_lipschitz + gp_lipschitz

a = 1.2
b = 1.
q = 1.
r = 1.

true_dynamics = safe_learning.LinearSystem((a, b), name='true_dynamics')

# Define a GP model over the dynamics
gp = gpflow.models.gpr.GPR(np.empty((0, 2), dtype=safe_learning.config.np_dtype),
                    np.empty((0, 1), dtype=safe_learning.config.np_dtype),
                    kernel,
                    mean_function=mean_function)
gp.likelihood.variance = noise_var

dynamics = safe_learning.GaussianProcess(gp, name='gp_dynamics')

k_opt, s_opt = safe_learning.utilities.dlqr(a, b, q, r)

# C:\Users\U755275\AppData\Local\Continuum\anaconda3\lib\site-packages;C:\Users\U755275\AppData\Local\Continuum\anaconda3\DLLs;C:\Users\U755275\AppData\Local\Continuum\anaconda3\lib;C:\Users\U755275\AppData\Local\Continuum\anaconda3;C:\Users\U755275\AppData\Roaming\Python\Python35\site-packages;C:\Users\U755275\AppData\Local\Continuum\anaconda3\lib\site-packages;C:\Users\U755275\AppData\Local\Continuum\anaconda3\lib\site-packages\Sphinx-1.4.1-py3.5.egg;C:\Users\U755275\AppData\Local\Continuum\anaconda3\lib\site-packages\textract-1.5.0-py3.5.egg;C:\Users\U755275\AppData\Local\Continuum\anaconda3\lib\site-packages\speechrecognition-3.5.0-py3.5.egg;C:\Users\U755275\AppData\Local\Continuum\anaconda3\lib\site-packages\win32;C:\Users\U755275\AppData\Local\Continuum\anaconda3\lib\site-packages\win32\lib;C:\Users\U755275\AppData\Local\Continuum\anaconda3\lib\site-packages\Pythonwin;C:\Users\U755275\AppData\Local\Continuum\anaconda3\lib\site-packages\IPython\extensions;C:\Users\U755275\.ipython;C:\Users\U755275\Documents\GitHub\AlphaDow;C:\Users\U755275\Documents\GitHub\safe_learning
# C:\Users\U755275\AppData\Local\Continuum\anaconda3\lib\site-packages;C:\Users\U755275\AppData\Local\Continuum\anaconda3\DLLs;C:\Users\U755275\AppData\Local\Continuum\anaconda3\lib;C:\Users\U755275\AppData\Local\Continuum\anaconda3;C:\Users\U755275\AppData\Roaming\Python\Python35\site-packages;C:\Users\U755275\AppData\Local\Continuum\anaconda3\lib\site-packages;C:\Users\U755275\AppData\Local\Continuum\anaconda3\lib\site-packages\Sphinx-1.4.1-py3.5.egg;C:\Users\U755275\AppData\Local\Continuum\anaconda3\lib\site-packages\textract-1.5.0-py3.5.egg;C:\Users\U755275\AppData\Local\Continuum\anaconda3\lib\site-packages\speechrecognition-3.5.0-py3.5.egg;C:\Users\U755275\AppData\Local\Continuum\anaconda3\lib\site-packages\win32;C:\Users\U755275\AppData\Local\Continuum\anaconda3\lib\site-packages\win32\lib;C:\Users\U755275\AppData\Local\Continuum\anaconda3\lib\site-packages\Pythonwin;C:\Users\U755275\AppData\Local\Continuum\anaconda3\lib\site-packages\IPython\extensions;C:\Users\U755275\.ipython;C:\Users\U755275\Documents\GitHub\AlphaDow;C:\Users\U755275\Documents\GitHub\safe_learning
