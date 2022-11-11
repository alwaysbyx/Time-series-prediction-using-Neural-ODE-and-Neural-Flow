###########################
# Latent ODEs and Latent Flows
# Author: Yuexin
###########################

import os
import sys
import matplotlib
import matplotlib.pyplot
import matplotlib.pyplot as plt
# matplotlib.use('MacOSX')
import time
import argparse
import numpy as np
from random import SystemRandom
import wandb
from sklearn import model_selection

import torch
import torch.nn as nn
from torch.nn.functional import relu
import torch.optim as optim

import lib.utils as utils
from lib.plotting import *

from lib.rnn_baselines import *
from lib.ode_rnn import *
from lib.create_latent_ode_model2 import create_LatentODE_model
from lib.parse_datasets import parse_datasets
from lib.ode_func import ODEFunc, ODEFunc_w_Poisson
from lib.diffeq_solver import DiffeqSolver

from lib.utils import compute_loss_all_batches

# Generative model for noisy data based on ODE
parser = argparse.ArgumentParser('Latent ODE')
parser.add_argument('-n',  type=int, default=1000, help="Size of the dataset")
parser.add_argument('--niters', type=int, default=300)
parser.add_argument('--lr',  type=float, default=1e-2, help="Starting learning rate.")
parser.add_argument('-b', '--batch-size', type=int, default=50)
parser.add_argument('--viz', action='store_true', help="Show plots while training")

parser.add_argument('--wandb', type=int, default=0, choices=[0,1])
parser.add_argument('--save', type=str, default='experiments/', help="Path for save checkpoints")
parser.add_argument('--load', type=str, default=None, help="ID of the experiment to load for evaluation. If None, run a new experiment.")
parser.add_argument('-r', '--random-seed', type=int, default=1991, help="Random_seed")

parser.add_argument('--dataset', type=str, default='periodic')
parser.add_argument('--data', type=str, default='sine')
parser.add_argument('-s', '--sample-tp', type=float, default=None, help="Number of time points to sub-sample."
	"If > 1, subsample exact number of points. If the number is in [0,1], take a percentage of available points per time series. If None, do not subsample")

parser.add_argument('-c', '--cut-tp', type=int, default=1, help="Cut out the section of the timeline of the specified length (in number of points)."
	"Used for periodic function demo.")

parser.add_argument('--quantization', type=float, default=0.1, help="Quantization on the physionet dataset."
	"Value 1 means quantization by 1 hour, value 0.1 means quantization by 0.1 hour = 6 min")

parser.add_argument('--latent-ode', action='store_true', help="Run Latent ODE seq2seq model")
parser.add_argument('--z0-encoder', type=str, default='ode', help="Type of encoder for Latent ODE model: odernn or rnn", choices=['ode','rnn','flow'])

# Flow model args
parser.add_argument('--hidden-layers', type=int, default=2, help='Number of hidden layers')
parser.add_argument('--hidden-dim', type=int, default=64, help='Size of hidden layer')
parser.add_argument('--flow-model', type=str, default='coupling', help='Model name', choices=['coupling', 'resnet', 'gru'])
parser.add_argument('--flow-layers', type=int, default=4, help='Number of hidden layers')
parser.add_argument('--time-net', type=str, default='TimeFourier', help='Name of time net', choices=['TimeFourier', 'TimeFourierBounded', 'TimeLinear', 'TimeTanh'])
parser.add_argument('--time-hidden-dim', type=int, default=8, help='Number of time features (only for Fourier)')


parser.add_argument('--classic-rnn', action='store_true', help="Run RNN baseline: classic RNN that sees true points at every point. Used for interpolation only.")
parser.add_argument('--rnn-cell', default="gru", help="RNN Cell type. Available: gru (default), expdecay")
parser.add_argument('--input-decay', action='store_true', help="For RNN: use the input that is the weighted average of impirical mean and previous value (like in GRU-D)")

parser.add_argument('--ode-rnn', action='store_true', help="Run ODE-RNN baseline: RNN-style that sees true points at every point. Used for interpolation only.")

parser.add_argument('--rnn-vae', action='store_true', help="Run RNN baseline: seq2seq model with sampling of the h0 and ELBO loss.")

parser.add_argument('-l', '--latents', type=int, default=6, help="Size of the latent state")
parser.add_argument('--rec-dims', type=int, default=20, help="Dimensionality of the recognition model (ODE or RNN).")

parser.add_argument('--rec-layers', type=int, default=2, help="Number of layers in ODE func in recognition ODE")
parser.add_argument('--gen-layers', type=int, default=2, help="Number of layers in ODE func in generative ODE")

parser.add_argument('-u', '--units', type=int, default=100, help="Number of units per layer in ODE func")
parser.add_argument('-g', '--gru-units', type=int, default=100, help="Number of units per layer in each of GRU update networks")

parser.add_argument('--poisson', action='store_true', help="Model poisson-process likelihood for the density of events in addition to reconstruction.")
parser.add_argument('--classif', action='store_true', help="Include binary classification loss -- used for Physionet dataset for hospiral mortality")

parser.add_argument('--linear-classif', action='store_true', help="If using a classifier, use a linear classifier instead of 1-layer NN")
parser.add_argument('--extrap', action='store_true', help="Set extrapolation mode. If this flag is not set, run interpolation mode.")

parser.add_argument('-t', '--timepoints', type=int, default=100, help="Total number of time-points")
parser.add_argument('--max-t',  type=float, default=20., help="We subsample points in the interval [0, args.max_tp]")
parser.add_argument('--noise-weight', type=float, default=0.01, help="Noise amplitude for generated traejctories")


args = parser.parse_args()
args.extrap = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
	torch.manual_seed(args.random_seed)
	np.random.seed(args.random_seed)
    
	experimentID = args.load
	if experimentID is None:
		# Make a new experiment ID
		experimentID = args.data
		if args.extrap:
			experimentID += '_extrap'
	data_obj = parse_datasets(args, device)
	input_dim = data_obj["input_dim"]

	classif_per_tp = False
	n_labels = 1
	obsrv_std = 0.01
	obsrv_std = torch.Tensor([obsrv_std]).to(device)
	z0_prior = Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.]).to(device))
	odemodel = create_LatentODE_model(args, input_dim, z0_prior, obsrv_std, device, 
			classif_per_tp = classif_per_tp,
			n_labels = n_labels)
	ckpt_path1 = os.path.join(args.save, "experiment_" + args.data + '_ode_coupling_extrap' + '.ckpt')
	ckpt_path2 = os.path.join(args.save, "experiment_" + args.data + '_flow_resnet_extrap' + '.ckpt')
	ckpt_path3 = os.path.join(args.save, "experiment_" + args.data + '_flow_coupling_extrap' + '.ckpt')
	utils.get_ckpt_model(ckpt_path1, odemodel, device)
	args.z0_encoder = 'flow'
	args.flow_model = 'resnet'
	flowmodel = create_LatentODE_model(args, input_dim, z0_prior, obsrv_std, device, 
			classif_per_tp = classif_per_tp,
			n_labels = n_labels)
	utils.get_ckpt_model(ckpt_path2, flowmodel, device)
	args.z0_encoder = 'flow'
	args.flow_model = 'coupling'
	flowmodel_c = create_LatentODE_model(args, input_dim, z0_prior, obsrv_std, device, 
			classif_per_tp = classif_per_tp,
			n_labels = n_labels)
	utils.get_ckpt_model(ckpt_path3, flowmodel_c, device)

	with torch.no_grad():
		data_dict = utils.get_next_batch(data_obj["test_dataloader"])
		# test_res_ode = compute_loss_all_batches(odemodel, 
		# 			data_obj["test_dataloader"], args,
		# 			n_batches = data_obj["n_test_batches"],
		# 			experimentID = experimentID,
		# 			device = device,
		# 			n_traj_samples = 3, kl_coef = 0.)
		# test_res_flow = compute_loss_all_batches(flowmodel, 
		# 			data_obj["test_dataloader"], args,
		# 			n_batches = data_obj["n_test_batches"],
		# 			experimentID = experimentID,
		# 			device = device,
		# 			n_traj_samples = 3, kl_coef = 0.)
		# test_res_flow_c = compute_loss_all_batches(flowmodel_c, 
		# 			data_obj["test_dataloader"], args,
		# 			n_batches = data_obj["n_test_batches"],
		# 			experimentID = experimentID,
		# 			device = device,
		# 			n_traj_samples = 3, kl_coef = 0.)
		# print(f"ode model Test Error: {test_res_ode['mse']}; Flow model test error: {test_res_flow['mse']}, Flow model coupling test error: {test_res_flow_c['mse']}")

		data =  data_dict["data_to_predict"]
		time_steps = data_dict["tp_to_predict"]
		mask = data_dict["mask_predicted_data"]
		
		observed_data =  data_dict["observed_data"]
		observed_time_steps = data_dict["observed_tp"]
		observed_mask = data_dict["observed_mask"]
		
		print('observed time', observed_time_steps[:5], 'tp_to_predict', time_steps[:5])

		time_steps_to_predict = utils.linspace_vector(observed_time_steps[0], observed_time_steps[-1], 100).to(device)
		reconstructions_ode, info = odemodel.get_reconstruction(time_steps_to_predict,
        observed_data, observed_time_steps, mask = observed_mask, n_traj_samples = 3, run_backwards = True)
		reconstructions_ode_plot = reconstructions_ode.mean(dim=0)[:3]
		reconstructions_flow, info = flowmodel.get_reconstruction(time_steps_to_predict,
        observed_data, observed_time_steps, mask = observed_mask, n_traj_samples = 3, run_backwards = True)
		reconstructions_flow_plot = reconstructions_flow.mean(dim=0)[:3]
		reconstructions_flow_c, info = flowmodel_c.get_reconstruction(time_steps_to_predict,
        observed_data, observed_time_steps, mask = observed_mask, n_traj_samples = 3, run_backwards = True)
		reconstructions_flow_c_plot = reconstructions_flow_c.mean(dim=0)[:3]
		
		print('reconstruction.shape', reconstructions_flow_c.shape)
		
		fig = plt.figure()
		plt.plot(time_steps.detach().numpy().reshape(-1), data.detach().numpy()[0,:], 'o', color='orange', alpha=0.5, label='Extrapolate')
		plt.plot(observed_time_steps.detach().numpy(), observed_data.detach().numpy()[0,:], 'o', color='grey', alpha=0.5, label='Train')
		plt.plot(time_steps_to_predict.cpu().numpy(), reconstructions_ode_plot.cpu().numpy()[1,:], color='blue', alpha=0.9, label='NeuralODE')
		plt.plot(time_steps_to_predict.cpu().numpy(), reconstructions_flow_plot.cpu().numpy()[1,:], color='green', alpha=0.9, label='Flow-Resnet')
		plt.plot(time_steps_to_predict.cpu().numpy(), reconstructions_flow_c_plot.cpu().numpy()[1,:], color='pink', alpha=0.9, label='Flow-Coupling')
		plt.grid(axis='x', color='0.95')
		plt.ylim(-4,4)
		plt.legend()
		plt.savefig(f'results/comparison/{args.data}',dpi=300)