# Neural-ODE and Neural-Flows for time series interpolation and extrapolation

Several points:
- neural-flows achieve better results in interpolation tasks, and perform much worse in extrapolation tasks. It would be a good choice to use neural-flows for missing data imputation as it is faster and has good interpolation ability. However, although the idea is novel, direcly modeling the solution of ODEs doesn't work better or even similarly as neural-ODEs do.
- On the other hand, experiments showed that using neural-ODE and neural-Flows to predict more complex data has long way to go. For instance, it is even better to use that average data in the previous period as estimated data for periodical data compared to neural-ODE and neural-Flows. 

## Prerequisites

Install `torchdiffeq` from https://github.com/rtqichen/torchdiffeq.


## Running script
* Time series interpolation and extrapolation
```
python3 run_models.py --datas sine  --latent-ode --z0-encoder ode --noise-weight 0.01 # using Neural-ODEs
python3 run_models.py --datas sine2  --latent-ode --z0-encoder flow --flow-model coupling --noise-weight 0.01  
python3 run_models.py --datas square  --latent-ode --z0-encoder flow --flow-model resnet --noise-weight 0.01 
python3 run_models.py --datas high  --latent-ode --noise-weight 0.01 
python3 run_models.py --datas sir  --latent-ode --noise-weight 0.01  
python3 run_models.py --datas pp  --latent-ode --noise-weight 0.01  
python3 run_models.py --datas polulation  --latent-ode --noise-weight 0.01  
```

* Time series interpolation and extrapolation
```
python3 run_models.py --data sine  --latent-ode --noise-weight 0.01 
```


## Dynamic results
- We show how Neural-ODE (left) and neural-flow (right) performs in both interpolation and extrapolation tasks.
- Neural-flow quickly converges for the training interval, but perform much worse in extrapolation tasks.
<p align="center">
<img align="middle" src="./results/sine_ode_coupling_extrap.gif" width="200" /><img align="middle" src="./results/sine_flow_coupling_extrap.gif" width="200" />
</p>

- See below Predator-pray model and addtion of sines.  Neural-flows perform better than interpolation tasks. Neural-ODEs perform better than extrapolation tasks.
<p align="center">
<img align="middle" src="./results/pp_ode_coupling_extrap.gif" width="300" /><img align="middle" src="./results/pp_flow_coupling_extrap.gif" width="300" />
</p>

<p align="center">
<img align="middle" src="./results/sine2_ode_coupling_extrap.gif" width="200" /><img align="middle" src="./results/sine2_flow_coupling_extrap.gif" width="200" />
</p>

- Data with second order dynamics 
<p align="center">
<img align="middle" src="./results/high_ode_coupling_extrap.gif" width="300" /><img align="middle" src="./results/high_flow_coupling_extrap.gif" width="300" />
</p>

- Population model
<p align="center">
<img align="middle" src="./results/population_ode_coupling_extrap.gif" width="200" /><img align="middle" src="./results/population_flow_coupling_extrap.gif" width="200" />
</p>

- SIR model
<p align="center">
<img align="middle" src="./results/sir_ode_coupling_extrap.gif" width="400" /><img align="middle" src="./results/sir_flow_coupling_extrap.gif" width="400" />
</p>

- Square
<p align="center">
<img align="middle" src="./results/square_ode_coupling_extrap.gif" width="200" /><img align="middle" src="./results/square_flow_coupling_extrap.gif" width="200" />
</p>

