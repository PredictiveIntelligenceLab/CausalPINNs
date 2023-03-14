import numpy as onp
import jax.numpy as np
from jax import random, grad, vmap, jit, jacfwd, jacrev
from jax.experimental import optimizers
from jax.experimental.ode import odeint
from jax.experimental.jet import jet
from jax.nn import relu
from jax.config import config
from jax import lax
from jax.flatten_util import ravel_pytree
import itertools
from functools import partial
from torch.utils import data
from tqdm import trange

import scipy.io
from scipy.interpolate import griddata
import matplotlib.pyplot as plt


# Define MLP
def MLP(layers, L=1.0, M=1, activation=relu):
  # Define input encoding function
    def input_encoding(t, x):
        w = 2.0 * np.pi / L
        k = np.arange(1, M + 1)
        out = np.hstack([t, 1, 
                         np.cos(k * w * x), np.sin(k * w * x)])
        return out
   
    def init(rng_key):
      def init_layer(key, d_in, d_out):
          k1, k2 = random.split(key)
          glorot_stddev = 1.0 / np.sqrt((d_in + d_out) / 2.)
          W = glorot_stddev * random.normal(k1, (d_in, d_out))
          b = np.zeros(d_out)
          return W, b
      key, *keys = random.split(rng_key, len(layers))
      params = list(map(init_layer, keys, layers[:-1], layers[1:]))
      return params
    def apply(params, inputs):
        t = inputs[0]
        x = inputs[1]
        H = input_encoding(t, x)
        for W, b in params[:-1]:
            outputs = np.dot(H, W) + b
            H = activation(outputs)
        W, b = params[-1]
        outputs = np.dot(H, W) + b
        return outputs
    return init, apply


# Define modified MLP
def modified_MLP(layers, L=1.0, M=1, activation=relu):
  def xavier_init(key, d_in, d_out):
      glorot_stddev = 1. / np.sqrt((d_in + d_out) / 2.)
      W = glorot_stddev * random.normal(key, (d_in, d_out))
      b = np.zeros(d_out)
      return W, b

  # Define input encoding function
  def input_encoding(t, x):
      w = 2 * np.pi / L
      k = np.arange(1, M + 1)
      out = np.hstack([t, 1, 
                         np.cos(k * w * x), np.sin(k * w * x)])
      return out


  def init(rng_key):
      U1, b1 =  xavier_init(random.PRNGKey(12345), layers[0], layers[1])
      U2, b2 =  xavier_init(random.PRNGKey(54321), layers[0], layers[1])
      def init_layer(key, d_in, d_out):
          k1, k2 = random.split(key)
          W, b = xavier_init(k1, d_in, d_out)
          return W, b
      key, *keys = random.split(rng_key, len(layers))
      params = list(map(init_layer, keys, layers[:-1], layers[1:]))
      return (params, U1, b1, U2, b2) 

  def apply(params, inputs):
      params, U1, b1, U2, b2 = params
        
      t = inputs[0]
      x = inputs[1]
      inputs = input_encoding(t, x)  
      U = activation(np.dot(inputs, U1) + b1)
      V = activation(np.dot(inputs, U2) + b2)
      for W, b in params[:-1]:
          outputs = activation(np.dot(inputs, W) + b)
          inputs = np.multiply(outputs, U) + np.multiply(1 - outputs, V) 
      W, b = params[-1]
      outputs = np.dot(inputs, W) + b
      return outputs
  return init, apply     


class DataGenerator(data.Dataset):
    def __init__(self, t0, t1, n_t=10, n_x=64, rng_key=random.PRNGKey(1234)):
        'Initialization'
        self.t0 = t0
        self.t1 = t1
        self.n_t = n_t
        self.n_x = n_x
        self.key = rng_key

    def __getitem__(self, index):
        'Generate one batch of data'
        self.key, subkey = random.split(self.key)
        batch = self.__data_generation(subkey)
        return batch

    @partial(jit, static_argnums=(0,))
    def __data_generation(self, key):
        'Generates data containing batch_size samples'
        subkeys = random.split(key, 2)
        t_r = random.uniform(subkeys[0], shape=(self.n_t,), minval=self.t0, maxval=self.t1).sort()
        x_r = random.uniform(subkeys[1], shape=(self.n_x,), minval=-1.0, maxval=1.0)
        batch = (t_r, x_r)
        return batch
    
  

# Define the model
class PINN:
    def __init__(self, key, arch, layers, M_x, state0, t0, t1, n_t, n_x, tol=1.0): 
        
        # grid
        eps = 0.01 * t1
        self.t_r = np.linspace(t0, t1 + eps, n_t)
        self.x_r = np.linspace(-1.0, 1.0, n_x)

        # IC
        t_ic = np.zeros((x_star.shape[0], 1))
        x_ic = x_star.reshape(-1, 1)
        self.X_ic = np.hstack([t_ic, x_ic])
        self.Y_ic = state0
    
        # Weight matrix and causal parameter
        self.M = np.triu(np.ones((n_t, n_t)), k=1).T 
        self.tol = tol
              
        if arch == 'MLP':
            d0 = 2 * M_x + 2
            layers = [d0] + layers
            self.init, self.apply = MLP(layers, L=2.0, M=M_x, activation=np.tanh)
            params = self.init(rng_key = key)
        
        if arch == 'modified_MLP':
            d0 = 2 * M_x + 2
            layers = [d0] + layers
            self.init, self.apply = modified_MLP(layers, L=2.0, M=M_x, activation=np.tanh)
            params = self.init(rng_key = key)

            
        # Use optimizers to set optimizer initialization and update functions
        lr = optimizers.exponential_decay(1e-3, decay_steps=5000, decay_rate=0.9)
        self.opt_init,  self.opt_update, self.get_params = optimizers.adam(lr)
        self.opt_state = self.opt_init(params) 
        _, self.unravel = ravel_pytree(params)
        
        # Evaluate functions over a grid
        self.u_pred_fn = vmap(vmap(self.neural_net, (None, 0, None)), (None, None, 0))  # consistent with the dataset
        self.r_pred_fn = vmap(vmap(self.residual_net, (None, None, 0)), (None, 0, None))

        # Logger
        self.loss_log = []
        self.loss_ics_log = []
        self.loss_res_log = []
        
        self.itercount = itertools.count()
    
    
    def neural_net(self, params, t, x):
        z = np.stack([t, x])
        outputs = self.apply(params, z)
        return outputs[0]

    def residual_net(self, params, t, x): 
        u = self.neural_net(params, t, x)
        u_t = grad(self.neural_net, argnums=1)(params, t, x)
        u_fn = lambda x: self.neural_net(params, t, x) # For using Taylor-mode AD
        _, (u_x, u_xx, u_xxx, u_xxxx) = jet(u_fn, (x, ), [[1.0, 0.0, 0.0, 0.0]]) #  Taylor-mode AD
        return u_t + 5 * u * u_x + 0.5 * u_xx + 0.005 * u_xxxx
    
    # Compute the temporal weights
    @partial(jit, static_argnums=(0,))
    def residuals_and_weights(self, params, batch, tol):
        t_r, x_r = batch
        L_0 = 1e3 * self.loss_ics(params)
        r_pred = self.r_pred_fn(params, t_r, x_r)
        L_t = np.mean(r_pred**2, axis=1)
        W = lax.stop_gradient(np.exp(- tol * (self.M @ L_t + L_0) ))
        return L_0, L_t, W

    # Initial condition loss
    @partial(jit, static_argnums=(0,))
    def loss_ics(self, params):
        # Compute forward pass
        u_pred = vmap(self.neural_net, (None, 0, 0))(params, self.X_ic[:,0], self.X_ic[:,1])
        # Compute loss
        loss_ics = np.mean((self.Y_ic.flatten() - u_pred.flatten())**2)
        return loss_ics

    # Residual loss
    @partial(jit, static_argnums=(0,))
    def loss_res(self, params, batch):
        t_r, x_r = batch
        # Compute forward pass        
        r_pred = self.r_pred_fn(params, t_r, x_r)
        # Compute loss
        loss_r = np.mean(r_pred**2)
        return loss_r  

    # Total loss
    @partial(jit, static_argnums=(0,))
    def loss(self, params, batch):
        L_0, L_t, W = self.residuals_and_weights(params, batch, self.tol)
        # Compute loss
        loss = np.mean(W * L_t + L_0)
        return loss

    # Define a compiled update step
    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state, batch):
        params = self.get_params(opt_state)
        g = grad(self.loss)(params, batch)
        return self.opt_update(i, g, opt_state)

    # Optimize parameters in a loop
    def train(self, dataset, nIter = 10000):
        res_data = iter(dataset)
        pbar = trange(nIter)
        # Main training loop
        for it in pbar:
            # Get batch
            batch= next(res_data)
            self.current_count = next(self.itercount)
            self.opt_state = self.step(self.current_count, self.opt_state, batch)
            
            if it % 1000 == 0:
                params = self.get_params(self.opt_state)

                loss_value = self.loss(params, batch)
                loss_ics_value = self.loss_ics(params)
                loss_res_value = self.loss_res(params, batch)
                _, _, W_value = self.residuals_and_weights(params, batch, self.tol)

                self.loss_log.append(loss_value)
                self.loss_ics_log.append(loss_ics_value)
                self.loss_res_log.append(loss_res_value)

                pbar.set_postfix({'Loss': loss_value, 
                                  'loss_ics' : loss_ics_value, 
                                  'loss_res':  loss_res_value,
                                  'W_min'  : W_value.min()})
                
                if W_value.min() > 0.99:
                    break
           

# Load data
data = scipy.io.loadmat('ks_simple.mat')
# Test data
usol = data['usol']


# Hpyer-parameters
key = random.PRNGKey(1234)
M_t = 2
M_x = 5
t0 = 0.0
t1 = 0.1
n_t = 32
n_x = 64
tol_list = [1e-2, 1e-1, 1e0, 1e1, 1e2]
layers = [256, 256, 256, 1] # using Fourier embedding so it is not 1

# Initial state
state0 = usol[:, 0:1]
dt = 1 / 250
idx = int(t1 / dt)
t_star = data['t'][0][:idx]
x_star = data['x'][0]

# Create data set
dataset = DataGenerator(t0, t1, n_t, n_x)

arch = 'modified_MLP'
print('arch:', arch)

N = 10
u_pred_list = []
params_list = []
losses_list = []


# Time marching
for k in range(N):
    # Initialize model
    print('Final Time: {}'.format((k + 1) * t1))
    model = PINN(key, arch, layers, M_x, state0, t0, t1, n_t, n_x)

    # Train
    for tol in tol_list:    
        model.tol = tol
        print("tol:", model.tol)
        # Train
        model.train(dataset, nIter=200000)
        
    # Store
    params = model.get_params(model.opt_state) 
    u_pred = model.u_pred_fn(params, t_star, x_star)
    u_pred_list.append(u_pred)
    flat_params, _  = ravel_pytree(params)
    params_list.append(flat_params)
    losses_list.append([model.loss_log, model.loss_ics_log, model.loss_res_log])
    

    np.save('u_pred_list.npy', u_pred_list)
    np.save('params_list.npy', params_list)
    np.save('losses_list.npy', losses_list)
    
    # error 
    u_preds = np.hstack(u_pred_list)
    error = np.linalg.norm(u_preds - usol[:, :(k+1) * idx]) / np.linalg.norm(usol[:, :(k+1) * idx]) 
    print('Relative l2 error: {:.3e}'.format(error))
    
    params = model.get_params(model.opt_state)
    u0_pred = vmap(model.neural_net, (None, None, 0))(params, t1, x_star)
    state0 = u0_pred
