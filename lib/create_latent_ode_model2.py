###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Author: Yulia Rubanova
###########################

import os
import numpy as np

import torch
import torch.nn as nn
from torch.nn.functional import relu

import lib.utils as utils
from lib.latent_ode import LatentODE
from lib.encoder_decoder import *
from lib.diffeq_solver import DiffeqSolver

from torch.distributions.normal import Normal
from lib.ode_func import ODEFunc, ODEFunc_w_Poisson
from lib.flow import ResNetFlow, CouplingFlow

#####################################################################################################

def create_LatentODE_model(args, input_dim, z0_prior, obsrv_std, device, 
	classif_per_tp = False, n_labels = 1):

	dim = args.latents
    
	ode_func_net = utils.create_net(dim, args.latents, 
        n_layers = args.gen_layers, n_units = args.units, nonlinear = nn.Tanh)

	gen_ode_func = ODEFunc(
        input_dim = input_dim, 
        latent_dim = args.latents, 
        ode_func_net = ode_func_net,
        device = device).to(device)

	z0_diffeq_solver = None
	n_rec_dims = args.rec_dims
	enc_input_dim = int(input_dim) * 2 # we concatenate the mask
	gen_data_dim = input_dim

	z0_dim = args.latents

	if args.z0_encoder == "ode":
		ode_func_net = utils.create_net(n_rec_dims, n_rec_dims, 
			n_layers = args.rec_layers, n_units = args.units, nonlinear = nn.Tanh)

		rec_ode_func = ODEFunc(
			input_dim = enc_input_dim, 
			latent_dim = n_rec_dims,
			ode_func_net = ode_func_net,
			device = device).to(device)

		z0_diffeq_solver = DiffeqSolver(enc_input_dim, rec_ode_func, "euler", args.latents, 
			odeint_rtol = 1e-3, odeint_atol = 1e-4, device = device)
		
		encoder_z0 = Encoder_z0_ODE_RNN(n_rec_dims, enc_input_dim, z0_diffeq_solver, 
			z0_dim = z0_dim, n_gru_units = args.gru_units, device = device).to(device)
        
		diffeq_solver = DiffeqSolver(gen_data_dim, gen_ode_func, 'dopri5', args.latents, 
		odeint_rtol = 1e-3, odeint_atol = 1e-4, device = device)

	elif args.z0_encoder == "rnn":
		encoder_z0 = Encoder_z0_RNN(z0_dim, enc_input_dim,
			lstm_output_size = n_rec_dims, device = device).to(device)
		diffeq_solver = DiffeqSolver(gen_data_dim, gen_ode_func, 'dopri5', args.latents, 
		odeint_rtol = 1e-3, odeint_atol = 1e-4, device = device)
	
	elif args.z0_encoder == 'flow':
		if args.flow_model == 'coupling':
			flow = CouplingFlow
		elif args.flow_model == 'resnet':
			flow = ResNetFlow
		hidden_dims = [args.hidden_dim] * args.hidden_layers
		z0_diffeq_solver = DiffeqSolver(enc_input_dim, flow(n_rec_dims, args.flow_layers, hidden_dims, args.time_net, args.time_hidden_dim), "flow", args.latents, 
			odeint_rtol = 1e-3, odeint_atol = 1e-4, device = device)
		encoder_z0 = Encoder_z0_ODE_RNN(n_rec_dims, enc_input_dim, z0_diffeq_solver, 
			z0_dim = z0_dim, n_gru_units = args.gru_units, device = device, method='flow').to(device)
		diffeq_solver = DiffeqSolver(gen_data_dim, flow(args.latents, args.flow_layers, hidden_dims, args.time_net, args.time_hidden_dim), "flow", args.latents, 
		odeint_rtol = 1e-3, odeint_atol = 1e-4, device = device)


	decoder = Decoder(args.latents, gen_data_dim).to(device)

	model = LatentODE(
		input_dim = gen_data_dim, 
		latent_dim = args.latents, 
		encoder_z0 = encoder_z0, 
		decoder = decoder, 
		diffeq_solver = diffeq_solver, 
		z0_prior = z0_prior, 
		device = device,
		obsrv_std = obsrv_std,
		use_poisson_proc = args.poisson, 
		use_binary_classif = args.classif,
		linear_classifier = args.linear_classif,
		classif_per_tp = classif_per_tp,
		n_labels = n_labels,
		train_classif_w_reconstr = (args.dataset == "physionet")
		).to(device)

	return model
