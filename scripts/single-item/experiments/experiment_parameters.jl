#cd("./DRL_MMULS")
#println(pwd())
using Pkg; Pkg.activate("."); Pkg.instantiate()
using DRL_MMULS
using Distributions, Flux, BSON, CSV, DataFrames, ProgressMeter, Random
using CUDA
using UnicodePlots, ParameterSchedulers
import ParameterSchedulers.Scheduler

#include("forecast_generation.jl") 
Random.seed!(0)
#device selection, change to `cpu` if you do not have an Nvidia compatible GPU. This will significantly slow down the method.
device = gpu
test_periods = 104
#default parameters of train and test environments.
μ = 10.
holding = 1.
leadtime_d = 2
shortage_db = 25.
shortages_dls = 75.
setup_d = 320.
CV_d = 0.2
horizon_d = 32
μ_distribution =  Uniform(0,20)
policy = sSPolicy()
#test values
leadtimes = [8, 4, 1, 0]
shortages_b = [10, 25, 50]
shortages_ls = [50, 75, 100]
setups = [0., 80., 1280.]
CVs = [0.1, 0.3]
lostsales = [false, true]
horizons = [16, 8, 4]
#Iterators
n_instances = sum(length.([leadtimes, shortages_b, setups, CVs, horizons]))
i_leadtimes = fill(leadtime_d, n_instances)
i_shortages_b = fill(shortage_db, n_instances)
i_shortages_ls = fill(shortages_dls, n_instances)
i_setups = fill(setup_d, n_instances)
i_CVs = fill(CV_d, n_instances)
i_horizons = fill(horizon_d, n_instances)
let idx = 0
    for lt in leadtimes
        idx += 1
        i_leadtimes[idx] = lt
    end
    for (bd, bls) in zip(shortages_b, shortages_ls)
        idx += 1
        i_shortages_b[idx] = bd
        i_shortages_ls[idx] = bls
    end
    for K in setups
        idx += 1
        i_setups[idx] = K
    end
    for cv in CVs
        idx += 1
        i_CVs[idx] = cv
    end
    for horizon in horizons
        idx += 1
        i_horizons[idx] = horizon
    end
end
i_leadtimes = repeat(i_leadtimes, 2)
i_shortages = [i_shortages_b; i_shortages_ls]
i_setups = repeat(i_setups, 2)
i_CVs = repeat(i_CVs, 2)
i_lostsales = repeat(lostsales, inner = n_instances)
i_horizons = repeat(i_horizons, 2)
n_instances *= 2

#load forecasts
forecast_df = CSV.read("data/single-item/forecasts.csv", DataFrame)
forecasts = Vector{Float64}[]
for forecast_string in forecast_df.forecast
    f = eval(Meta.parse(forecast_string))
    push!(forecasts, f)
end

#ADI parameters
first_var = 0.01
var_discount = 0.9
#(see exp_multiplicative_mmfe)

#hyperparameters
steps_per_episode = 52
batch_size = 256
n_actors = 30
stop_iterations = 10000
n_epochs = 5

#Learning rate schedules
warmup_iterations = 2000
actor_updates = stop_iterations*n_actors*steps_per_episode÷batch_size
critic_updates = actor_updates*n_epochs
warmup = Int(round(warmup_iterations/stop_iterations*actor_updates))
actor_warmup = Exp(λ = 1f-6, γ = 100^(1f0/warmup))
actor_sinexp = Exp(λ = 1f-4, γ = (1f0/10)^(1f0/(actor_updates-warmup)))
actor_schedule = Sequence(actor_warmup => warmup, actor_sinexp => actor_updates - warmup)
critic_schedule = Sequence(1f-4 => critic_updates)
