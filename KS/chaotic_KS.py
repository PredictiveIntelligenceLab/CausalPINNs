import numpy as onp
import jax.numpy as np
from jax import random, grad, vmap, jit, jacfwd, jacrev
from jax.example_libraries import optimizers
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


# Define the neural net
def modified_MLP(layers, L=1.0, M_t=1, M_x=1, activation=relu):
  def xavier_init(key, d_in, d_out):
      glorot_stddev = 1. / np.sqrt((d_in + d_out) / 2.)
      W = glorot_stddev * random.normal(key, (d_in, d_out))
      b = np.zeros(d_out)
      return W, b

  # Define input encoding function
  def input_encoding(t, x):
      w = 2 * np.pi / L
      k_t = np.power(10, np.arange(-M_t//2, M_t//2))
      k_x = np.arange(1, M_x + 1)
        
      out = np.hstack([k_t * t ,
                       1, np.cos(k_x * w * x), np.sin(k_x * w * x)])
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
        self.t1 = (1 + 0.01) * t1
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
        x_r = random.uniform(subkeys[1], shape=(self.n_x,), minval=0.0, maxval=2.0*np.pi)
        batch = (t_r, x_r)
        return batch


# Define the model
class PINN:
    def __init__(self, key, u_exact, arch, layers, M_t, M_x, state0, t0, t1, n_t, n_x, tol): 
        
        self.u_exact = u_exact
        
        self.M_t = M_t
        self.M_x = M_x

        # grid
        self.n_t = n_t
        self.n_x = n_x

        self.t0 = t0
        self.t1 = t1
        eps = 0.01 * self.t1
        self.t_r   = np.linspace(self.t0, self.t1 + eps, n_t)
        self.x_r = np.linspace(0, 2.0 * np.pi, n_x)

        # IC
        t_ic = np.zeros((x_star.shape[0], 1))
        x_ic = x_star.reshape(-1, 1)
        self.X_ic = np.hstack([t_ic, x_ic])
        self.Y_ic = state0
    
        # Weight matrix
        self.M = np.triu(np.ones((n_t, n_t)), k=1).T 
        self.tol = tol


        d0 = 2 * M_x + M_t + 1
        layers = [d0] + layers
        self.init, self.apply = modified_MLP(layers, L=2.0*np.pi, M_t=self.M_t, M_x=self.M_x, activation=np.tanh)
        params = self.init(rng_key = key)
         
        # Use optimizers to set optimizer initialization and update functions
        self.opt_init,         self.opt_update,         self.get_params = optimizers.adam(optimizers.exponential_decay(1e-3, 
                                                                      decay_steps=5000, 
                                                                      decay_rate=0.9))
        self.opt_state = self.opt_init(params) 
        _, self.unravel = ravel_pytree(params)
        
        
        self.u_pred_fn = vmap(vmap(self.neural_net, (None, 0, None)), (None, None, 0))  # consistent with the dataset
        self.r_pred_fn = vmap(vmap(self.residual_net, (None, None, 0)), (None, 0, None))

        # Logger
        self.itercount = itertools.count()

        self.l2_error_log = []
        self.loss_log = []
        self.loss_ics_log = []
        self.loss_res_log = []
    
    def neural_net(self, params, t, x):
        z = np.stack([t, x])
        outputs = self.apply(params, z)
        return outputs[0]

    def residual_net(self, params, t, x): 
        u = self.neural_net(params, t, x)
        u_t = grad(self.neural_net, argnums=1)(params, t, x)

        u_fn = lambda x: self.neural_net(params, t, x)
        _, (u_x, u_xx, u_xxx, u_xxxx) = jet(u_fn, (x, ), [[1.0, 0.0, 0.0, 0.0]])

        return u_t + 100.0 / 16.0 * u * u_x + 100.0 / 16.0**2 * u_xx + 100.0 / 16.0**4 * u_xxxx
    

    @partial(jit, static_argnums=(0,))
    def residuals_and_weights(self, params, batch, tol):
        t_r, x_r = batch
        L_0 = 1e4 * self.loss_ics(params)
        r_pred = self.r_pred_fn(params, t_r, x_r)
        L_t = np.mean(r_pred**2, axis=1)
        W = lax.stop_gradient(np.exp(- tol * (self.M @ L_t + L_0) ))
        return L_0, L_t, W

    @partial(jit, static_argnums=(0,))
    def loss_ics(self, params):
        # Compute forward pass
        u_pred = vmap(self.neural_net, (None, 0, 0))(params, self.X_ic[:,0], self.X_ic[:,1])
        # Compute loss
        loss_ics = np.mean((self.Y_ic.flatten() - u_pred.flatten())**2)
        return loss_ics


    @partial(jit, static_argnums=(0,))
    def loss_res(self, params, batch):
        t_r, x_r = batch
        # Compute forward pass        
        r_pred = self.r_pred_fn(params, t_r, x_r)
        # Compute loss
        loss_r = np.mean(r_pred**2)
        return loss_r  

    @partial(jit, static_argnums=(0,))
    def loss(self, params, batch):
        L_0, L_t, W = self.residuals_and_weights(params, batch, self.tol)
        # Compute loss
        loss = np.mean(W * L_t + L_0)
        return loss

    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params):
        u_pred = self.u_pred_fn(params, t_star[:num_step], x_star)
        l2_error = np.linalg.norm(u_pred - self.u_exact) / np.linalg.norm(self.u_exact) 
        return l2_error

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
            batch= next(res_data)
            self.current_count = next(self.itercount)
            self.opt_state = self.step(self.current_count, self.opt_state, batch)
            
            if it % 1000 == 0:
                params = self.get_params(self.opt_state)
                
                
                l2_error_value = self.compute_l2_error(params)
                loss_value = self.loss(params, batch)

                loss_ics_value = self.loss_ics(params)
                loss_res_value = self.loss_res(params, batch)
                
                _, _, W_value = self.residuals_and_weights(params, batch, self.tol)

                self.l2_error_log.append(l2_error_value)
                self.loss_log.append(loss_value)
                self.loss_ics_log.append(loss_ics_value)
                self.loss_res_log.append(loss_res_value)

                pbar.set_postfix({'l2 error': l2_error_value,
                                  'Loss': loss_value, 
                                  'loss_ics' : loss_ics_value, 
                                  'loss_res':  loss_res_value,
                                  'W_min'  : W_value.min()})
                
                if W_value.min() > 0.99:
                    break
           
    # Evaluates predictions at test points  
    @partial(jit, static_argnums=(0,))
    def predict_u(self, params, X_star):
        u_pred = vmap(self.u_net, (None, 0, 0))(params, X_star[:,0], X_star[:,1])
        return u_pred


data = scipy.io.loadmat('../ks_chaotic.mat')
# Test data
usol = data['usol']

t_star = data['t'][0]
x_star = data['x'][0]
TT, XX = np.meshgrid(t_star, x_star)
X_star =  np.hstack((TT.flatten()[:, None], XX.flatten()[:, None]))



# Hpyer-parameters
key = random.PRNGKey(1234)
M_t = 6
M_x = 5
layers = [128, 128, 128, 128, 128, 128, 128, 128, 1]
num_step = 25
t0 = 0.0
t1 = t_star[num_step]
n_t = 32
n_x = 256

tol = 1.0
tol_list = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]
time_step = 0

state0 = usol[:, time_step:time_step+1]
t_star = data['t'][0][:num_step]
x_star = data['x'][0]

# Create data set
dataset = DataGenerator(t0, t1, n_t, n_x)


# arch = 'MLP'
arch = 'modified_MLP'
print('Arch:', arch)
print('Alg: temporal reweighting, Random collocation points')


N = 250 // num_step

u_pred_list = []
params_list = []
losses_list = []

for k in range(N):
    # Initialize model
    u_exact = usol[:, time_step + k * num_step:time_step + (k+1) * num_step] # (512, num_step)
    print('Final Time: {}'.format(k + 1))
    model = PINN(key, u_exact, arch, layers, M_t, M_x, state0, t0, t1, n_t, n_x, tol)

    # Train
    for tol in tol_list:    
        model.tol = tol
        print('tol: ', tol)
        # Train
        model.train(dataset, nIter=200000)
        
    # Store
    params = model.get_params(model.opt_state)
    u_pred = model.u_pred_fn(params, t_star, x_star)
    u_pred_list.append(u_pred)
    flat_params, _  = ravel_pytree(params)
    params_list.append(flat_params)
    losses_list.append([model.loss_log, model.loss_ics_log, model.loss_res_log])
    

    np.save(arch + '_u_pred_list.npy', u_pred_list)
    np.save(arch + '_params_list.npy', params_list)
    np.save(arch + '_losses_list.npy', losses_list)

    u_preds = np.hstack(u_pred_list)
    error = np.linalg.norm(u_preds - usol[:, time_step:time_step + (k+1) * num_step]) / np.linalg.norm(usol[:, time_step:time_step + (k+1) * num_step]) 
    print('Relative l2 error: {:.3e}'.format(error))
    
    params = model.get_params(model.opt_state)
    u0_pred = vmap(model.neural_net, (None, None, 0))(params, t1, x_star)
    state0 = u0_pred

