###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Author: Yulia Rubanova
###########################

# Create a synthetic dataset
from __future__ import absolute_import, division
from __future__ import print_function
import os
from re import S
import matplotlib
if os.path.exists("/Users/yulia"):
	matplotlib.use('TkAgg')
else:
	matplotlib.use('Agg')

import numpy as np
import numpy.random as npr
from scipy.special import expit as sigmoid
import pickle
import matplotlib.pyplot as plt
import matplotlib.image
import torch
import lib.utils as utils
import scipy.signal
import scipy.integrate

# ======================================================================================

def get_next_val(init, t, tmin, tmax, final = None):
	if final is None:
		return init
	val = init + (final - init) / (tmax - tmin) * t
	return val

def generate_periodic(time_steps, starting_point, data='sine'):
	times = time_steps.clone().numpy()
	if data == 'sine':
		return np.sin(1.5*times[:,None]) + starting_point
	elif data == 'square':
		return np.sign(np.sin(1.5 * times[:,None])) + starting_point
	elif data == 'sine2':
		s = 0
		for i in range(2,6,2):
			s += np.sin((2*i+1)*time_steps[:,None])
		return np.array(s) + starting_point
	elif data == 'population':
		return np.array(np.exp(0.2 * time_steps[:,None])) + starting_point
	elif data == 'pp':
		starting_point = np.random.uniform(1, 5, (2,))
		a,b,e,gamma = 1.*np.random.uniform(0.9,1.1) ,2.*np.random.uniform(0.9,1.1) ,3.*np.random.uniform(0.9,1.1) ,4.*np.random.uniform(0.9,1.1)
		ode = lambda y, t: np.array([a*y[0]-b*y[0]*y[1], e*y[0]*y[1]-gamma*y[1]])
		return scipy.integrate.odeint(ode, starting_point, time_steps) + starting_point
	elif data == 'sir':
		y = [np.random.uniform(0.7, 0.8), np.random.uniform(0.05, 0.15), 0] # S, I, R
		y[2] = 1 - y[0] - y[1]
		b, gamma = 0.4*np.random.uniform(0.9,1.1), 0.2*np.random.uniform(0.9,1.1)
		ode = lambda y, t: np.array([-b*y[0]*y[1], b*y[0]*y[1]-gamma*y[1], gamma*y[1]])
		return scipy.integrate.odeint(ode, y, time_steps)
	elif data == 'high':
		starting_point = np.random.uniform(1, 2 , (2,))
		x1 = np.exp(-0.1*times[:, None]) * (3*np.sin(times[:,None]) + np.cos(times[:,None]))
		x2 = np.exp(-0.3*times[:, None]) * (2*np.sin(1.2*times[:,None]) - 5 *  np.cos(1.2*times[:,None]))
		s =  np.concatenate([x1, x2], axis=1) + starting_point
		return s

def assign_value_or_sample(value, sampling_interval = [0.,1.]):
	if value is None:
		int_length = sampling_interval[1] - sampling_interval[0]
		return np.random.random() * int_length + sampling_interval[0]
	else:
		return value

class TimeSeries:
	def __init__(self, device = torch.device("cpu")):
		self.device = device
		self.z0 = None

	def init_visualization(self):
		self.fig = plt.figure(figsize=(10, 4), facecolor='white')
		self.ax = self.fig.add_subplot(111, frameon=False)
		plt.show(block=False)

	def visualize(self, truth):
		self.ax.plot(truth[:,0], truth[:,1])

	def add_noise(self, traj_list, time_steps, noise_weight):
		n_samples = traj_list.size(0)

		# Add noise to all the points except the first point
		n_tp = len(time_steps) - 1
		noise = np.random.sample((n_samples, n_tp))
		noise = torch.Tensor(noise).to(self.device)

		traj_list_w_noise = traj_list.clone()
		# Dimension [:,:,0] is a time dimension -- do not add noise to that
		traj_list_w_noise[:,1:,0] += noise_weight * noise
		return traj_list_w_noise


class Periodic(TimeSeries):
	def __init__(self, data='sine', device = torch.device("cpu"), 
		z0 = 0.):
		"""
		If some of the parameters (init_freq, init_amplitude, final_amplitude, final_freq) is not provided, it is randomly sampled.
		For now, all the time series share the time points and the starting point.
		"""
		super(Periodic, self).__init__(device)
		
		self.data = data
		self.z0 = z0

	def sample_traj(self, time_steps, n_samples = 1, noise_weight = 1.,
		cut_out_section = None):
		"""
		Sample periodic functions. 
		"""
		traj_list = []
		for i in range(n_samples):
			# init_freq = assign_value_or_sample(self.init_freq, [0.4,0.8])
			# if self.final_freq is None:
			# 	final_freq = init_freq
			# else:
			# 	final_freq = assign_value_or_sample(self.final_freq, [0.4,0.8])
			# init_amplitude = assign_value_or_sample(self.init_amplitude, [0.,1.])
			# final_amplitude = assign_value_or_sample(self.final_amplitude, [0.,1.])

			noisy_z0 = self.z0 + np.random.normal(loc=0., scale=0.1)

			# traj = generate_periodic(time_steps, init_freq = init_freq, 
			# 	init_amplitude = init_amplitude, starting_point = noisy_z0, 
			# 	final_amplitude = final_amplitude, final_freq = final_freq)

			# # Cut the time dimension
			# traj = np.expand_dims(traj[:,1:], 0)
			traj = generate_periodic(time_steps=time_steps, data=self.data, starting_point=noisy_z0)
			traj_list.append(traj)

		# shape: [n_samples, n_timesteps, 2]
		# traj_list[:,:,0] -- time stamps
		# traj_list[:,:,1] -- values at the time stamps
		traj_list = np.array(traj_list)
		traj_list = torch.Tensor().new_tensor(traj_list, device = self.device)
		traj_list = traj_list.squeeze(1)

		traj_list = self.add_noise(traj_list, time_steps, noise_weight)
		return traj_list


if __name__ == '__main__':
	generate = Periodic(data='high')
	time_step1 = np.linspace(0, 20, 100)
	time_steps = torch.from_numpy(np.linspace(0, 20, 100))
	traj_list = generate.sample_traj(time_steps, n_samples=10)
	traj_list = traj_list.cpu().numpy()
	plt.figure(figsize=(8, 4))
	plt.xkcd()
	plt.subplot(121)
	plt.plot(time_step1, traj_list[0,:,0], color = 'gray', label='x1')
	plt.plot(time_step1, traj_list[0,:,1], color = 'black', label='x2')
	plt.legend()
	plt.subplot(122)
	plt.plot(traj_list[0,:,0], traj_list[0,:,1], color='black')
	plt.xlabel('x1')
	plt.ylabel('x2')
	plt.tight_layout()
	plt.savefig('results/high.png')
	