###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Author: Yulia Rubanova
###########################

import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import os
from scipy.stats import kde

import numpy as np
import subprocess
import torch
import lib.utils as utils
import matplotlib.gridspec as gridspec
from lib.utils import get_device

from lib.encoder_decoder import *
from lib.rnn_baselines import *
from lib.ode_rnn import *
import torch.nn.functional as functional
from torch.distributions.normal import Normal
from lib.latent_ode import LatentODE

from lib.likelihood_eval import masked_gaussian_log_density
try:
	import umap
except:
	print("Couldn't import umap")


from lib.utils import compute_loss_all_batches


SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18
LARGE_SIZE = 22

def init_fonts(main_font_size = LARGE_SIZE):
	plt.rc('font', size=main_font_size)          # controls default text sizes
	plt.rc('axes', titlesize=main_font_size)     # fontsize of the axes title
	plt.rc('axes', labelsize=main_font_size - 2)    # fontsize of the x and y labels
	plt.rc('xtick', labelsize=main_font_size - 2)    # fontsize of the tick labels
	plt.rc('ytick', labelsize=main_font_size - 2)    # fontsize of the tick labels
	plt.rc('legend', fontsize=main_font_size - 2)    # legend fontsize
	plt.rc('figure', titlesize=main_font_size)  # fontsize of the figure title


def plot_trajectories(ax, traj, time_steps, min_y = None, max_y = None, title = "", 
		add_to_plot = False, label = None, add_legend = False, dim_to_show = 0,
		linestyle = '-', marker = 'o', mask = None, color = None, linewidth = 1):
	# expected shape of traj: [n_traj, n_timesteps, n_dims]
	# The function will produce one line per trajectory (n_traj lines in total)
	if not add_to_plot:
		ax.cla()
	ax.set_title(title)
	ax.set_xlabel('Time')
	ax.set_ylabel('x')
	
	if min_y is not None:
		ax.set_ylim(bottom = min_y)

	if max_y is not None:
		ax.set_ylim(top = max_y)

	for i in range(traj.size()[0]):
		d = traj[i].cpu().numpy()[:, dim_to_show]
		ts = time_steps.cpu().numpy()
		if mask is not None:
			m = mask[i].cpu().numpy()[:, dim_to_show]
			d = d[m == 1]
			ts = ts[m == 1]
		ax.plot(ts, d, linestyle = linestyle, label = label, marker=marker, color = color, linewidth = linewidth)

	if add_legend:
		ax.legend()


def plot_std(ax, traj, traj_std, time_steps, min_y = None, max_y = None, title = "", 
	add_to_plot = False, label = None, alpha=0.2, color = None):

	# take only the first (and only?) dimension
	mean_minus_std = (traj - traj_std).cpu().numpy()[:, :, 0]
	mean_plus_std = (traj + traj_std).cpu().numpy()[:, :, 0]

	for i in range(traj.size()[0]):
		ax.fill_between(time_steps.cpu().numpy(), mean_minus_std[i], mean_plus_std[i], 
			alpha=alpha, color = color)


def get_meshgrid(npts, int_y1, int_y2):
	min_y1, max_y1 = int_y1
	min_y2, max_y2 = int_y2
	
	y1_grid = np.linspace(min_y1, max_y1, npts)
	y2_grid = np.linspace(min_y2, max_y2, npts)

	xx, yy = np.meshgrid(y1_grid, y2_grid)

	flat_inputs = np.concatenate((np.expand_dims(xx.flatten(),1), np.expand_dims(yy.flatten(),1)), 1)
	flat_inputs = torch.from_numpy(flat_inputs).float()

	return xx, yy, flat_inputs


def add_white(cmap):
	cmaplist = [cmap(i) for i in range(cmap.N)]
	# force the first color entry to be grey
	cmaplist[0] = (1.,1.,1.,1.0)
	# create the new map
	cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
	return cmap


class Visualizations():
	def __init__(self, dim, device):
		self.init_visualization(dim)
		init_fonts(SMALL_SIZE)
		self.device = device

	def init_visualization(self, dim):
		self.fig = plt.figure(figsize=(4*dim, 7), facecolor='white')
		
		self.ax_traj = []
		for i in range(1,dim*2+1):
			self.ax_traj.append(self.fig.add_subplot(2, dim, i, frameon=False))

		self.plot_limits = {}
		plt.show(block=False)

	def set_plot_lims(self, ax, name):
		if name not in self.plot_limits:
			self.plot_limits[name] = (ax.get_xlim(), ax.get_ylim())
			return

		xlim, ylim = self.plot_limits[name]
		ax.set_xlim(xlim)
		ax.set_ylim(ylim)

	def draw_all_plots(self, data_dict, model,
		plot_name = "", save = False, experimentID = 0.):

		data =  data_dict["data_to_predict"]
		time_steps = data_dict["tp_to_predict"]
		mask = data_dict["mask_predicted_data"]
		
		observed_data =  data_dict["observed_data"]
		observed_time_steps = data_dict["observed_tp"]
		observed_mask = data_dict["observed_mask"]

		device = get_device(time_steps)

		for process in ['interpolate','extrapolate']:
			if process == 'interpolate':
				time_steps_to_predict = utils.linspace_vector(observed_time_steps[0], observed_time_steps[-1], 100).to(device)
				id_ = 0
				reconstructions, info = model.get_reconstruction(time_steps_to_predict, 
				observed_data, observed_time_steps, mask = observed_mask, n_traj_samples = 3, run_backwards = True)
				n_traj_to_show = 3
				# plot at most 10 trajectories
				data_for_plotting = observed_data[:n_traj_to_show]
				mask_for_plotting = observed_mask[:n_traj_to_show]
			else:
				time_steps_to_predict = utils.linspace_vector(observed_time_steps[-1], observed_time_steps[-1]+20, 100).to(device)
				id_ = data.shape[-1]
				reconstructions, info = model.get_reconstruction(time_steps_to_predict, 
				observed_data, observed_time_steps, mask = observed_mask, n_traj_samples = 3, run_backwards = False)
				n_traj_to_show = 3
				# plot at most 10 trajectories
				data_for_plotting = data[:n_traj_to_show]
				mask_for_plotting = mask[:n_traj_to_show]#torch.ones_like(data[:n_traj_to_show])

			reconstructions_for_plotting = reconstructions.mean(dim=0)[:n_traj_to_show]
			reconstr_std = reconstructions.std(dim=0)[:n_traj_to_show]

			dim_to_show = data.shape[-1]

			############################################
			# Plot reconstructions, true postrior and approximate posterior
			cmap = plt.cm.get_cmap('Set1')
			for dim_id in range(dim_to_show):
				max_y = max(
				data_for_plotting[:,:,dim_id].cpu().numpy().max(),
				reconstructions[:,:,dim_id].cpu().numpy().max())
				min_y = min(
					data_for_plotting[:,:,dim_id].cpu().numpy().min(),
					reconstructions[:,:,dim_id].cpu().numpy().min())
				# Plot observations
				plot_trajectories(self.ax_traj[dim_id+id_], 
					data_for_plotting[dim_id].unsqueeze(0), observed_time_steps, 
					mask = mask_for_plotting[dim_id].unsqueeze(0),
					min_y = min_y, max_y = max_y, #title="True trajectories", 
					marker = 'o', linestyle='', dim_to_show = dim_id,
					color = 'gray')
				# Plot reconstructions
				plot_trajectories(self.ax_traj[dim_id+id_],
					reconstructions_for_plotting[dim_id].unsqueeze(0), time_steps_to_predict, 
					min_y = min_y, max_y = max_y, title=f"{process} for dimension {dim_id}", dim_to_show = dim_id,
					add_to_plot = True, marker = '', color = 'orange', linewidth = 3)
				# Plot variance estimated over multiple samples from approx posterior
				plot_std(self.ax_traj[dim_id+id_], 
					reconstructions_for_plotting[dim_id].unsqueeze(0), reconstr_std[dim_id].unsqueeze(0), 
					time_steps_to_predict, alpha=0.5, color = 'orange')


		################################################

		self.fig.tight_layout()
		plt.draw()

		if save:
			dirname = "plots/" + str(experimentID) + "/"
			os.makedirs(dirname, exist_ok=True)
			self.fig.savefig(dirname + plot_name)





