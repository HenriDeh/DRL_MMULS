using DRL_MMULS
using InventoryModels
using Distributions, Flux, BSON, CSV, DataFrames, ProgressMeter, Random
using CUDA
using UnicodePlots, ParameterSchedulers
import ParameterSchedulers.Scheduler

#include("forecast_generation.jl") 

Random.seed!(0)
#device selection, change to `cpu` if you do not have an Nvidia compatible GPU.
device = gpu
test_periods = 400
#default parameters of train and test environments.
μ = 10.
holding = 1.
leadtime_d = 2
shortage_db = 25.
shortages_dls = 75.
setup_d = 320.
CV_d = 0.2
forecast_horizon = 32
#test values
leadtimes = [8, 4, 1, 0]
shortages_b = [10, 25, 50]
shortages_ls = [50, 75, 100]
setups = [80., 1280.]
CVs = [0.1, 0.3]
lostsales = [false, true]
#Iterators
n = sum(length.([leadtimes, shortages_b, setups, CVs]))
i_leadtimes = fill(leadtime_d, n)
i_shortages_b = fill(shortage_db, n)
i_shortages_ls = fill(shortages_dls, n)
i_setups = fill(setup_d, n)
i_CVs = fill(CV_d, n)
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
end
i_leadtimes = repeat(i_leadtimes, 2)
i_shortages = [i_shortages_b; i_shortages_ls]
i_setups = repeat(i_setups, 2)
i_CVs = repeat(i_CVs, 2)
i_lostsales = repeat(lostsales, inner = n)

#train parameters
μ_distributions =  [Uniform(0,20)]

#parameters of agents
policies = [sSPolicy()]

#load forecasts
forecast_df = CSV.read("data/single-item/forecasts.csv", DataFrame)
forecasts = Vector{Float64}[]
for forecast_string in forecast_df.forecast
    f = eval(Meta.parse(forecast_string))
    push!(forecasts, f)
end


#presolve testbed
function solve()
    CSV.write("data/single-item/scarf_testbed.csv", DataFrame(leadtime = Int[], shortage = Float64[], setup = Int[], lostsales = Bool[], CV = Float64[], forecast_id = Int[], opt_cost = Float64[], opt_MC_std = Float64[], solve_time_s = Float64[]))
    p = Progress(length(forecasts)*22)
    Threads.@threads for (leadtime, shortage, setup, CV, lostsale) in collect(zip(i_leadtimes, i_shortages, i_setups, i_CVs, i_lostsales))
        for (f_ID, forecast) in collect(enumerate(forecasts))
            scarf_df = DataFrame(leadtime = Int[], shortage = Float64[], setup = Int[], lostsales = Bool[], CV = Float64[], forecast_id = Int[], avg_cost = Float64[], MC_std = Float64[], solve_time_s = Float64[])
            env = sl_sip(holding, shortage, setup, CV, 0, forecast, leadtime*μ, leadtime, lostsales = lostsale, horizon = forecast_horizon, periods = test_periods)
            instance = Scarf.Instance(env, 0.99)
            instance.backlog = true
            time = @elapsed Scarf.DP_sS(instance, 0.1)
            (cost, std) = test_ss_policy(env, instance.s, instance.S)
            push!(scarf_df, [leadtime, shortage, setup, lostsale, CV, f_ID, cost, std, time])
            CSV.write("data/single-item/scarf_testbed.csv", scarf_df, append = true)
            ProgressMeter.next!(p)
        end
    end
end
#solve()#uncomment this to resolve the testbed with DP method


#hyperparameters
steps_per_episode = 52
batch_size = 252
n_actors = 30
stop_iterations = 15000
warmup_iterations = 1000
n_epochs = 5

#Learning rate schedules
actor_updates = stop_iterations*n_actors*steps_per_episode÷batch_size
critic_updates = actor_updates*n_epochs
warmup = Int(round(warmup_iterations/stop_iterations*actor_updates))
actor_warmup = Triangle(λ0 = 1f-6, λ1 = 1f-5, period = 2*warmup)
actor_sinexp = SinExp(λ0 = 1f-5, λ1 = 1f-4, period = (actor_updates-2*warmup)÷5, γ = 1f0/10^(1f0/(0.5f0actor_updates)))
actor_schedule = Sequence(actor_warmup => warmup, actor_sinexp => actor_updates - 2*warmup, Shifted(actor_warmup, warmup) => warmup)
critic_schedule = SinExp(λ0 = 1f-4, λ1 = 1f-3, period = (actor_updates-2*warmup)÷5*n_epochs, γ = 1f0/10^(1f0/0.5f0critic_updates))

#train N agents per setup
function ppo_testbed()
    N = 10 # agents trained per environment
    CSV.write("data/single-item/ppo_testbed.csv", DataFrame(leadtime = Int[], shortage = Float64[], setup = Int[], lostsales = Bool[], CV = Float64[], policy = String[], mu_dist_bounds = String[], avg_cost = Float64[], MC_std = [], train_time_s = Float64[], forecast_id = Int[], agent_id=Int[]))
    agent_id = 0
    for μ_distribution in μ_distributions,
        policy in policies
        for (leadtime, shortage, setup, CV, lostsale) in zip(i_leadtimes, i_shortages, i_setups, i_CVs, i_lostsales) 
            println("Solving LT= $leadtime, b=$shortage, K=$setup, CV = $CV, lostsales = $lostsale, policy = $policy, μ_distribution = $μ_distribution")
            for n in 1:N
                agent_id += 1
                Random.seed!(agent_id) #to reprocude correctly if restarted
                println("agent $agent_id / $(N*length(μ_distributions)*length(policies)*22)")
                #=env = sl_sip(holding, shortage, setup, CV, 0.,μ_distribution, Uniform(-5*(leadtime+1),10*(leadtime + 1)), leadtime, lostsales = lostsale, horizon = forecast_horizon, periods = steps_per_episode, policy = policy)
                agent_d = PPOPolicy(env, actor_optimiser = Scheduler(actor_schedule, ADAM()),  critic_optimiser = Scheduler(critic_schedule, ADAM()), n_hidden = 128,
                            γ = 0.99f0,λ = 0.90f0, clip_range = 0.2f0, entropy_weight = 1f-2, n_actors = n_actors, n_epochs = n_epochs, batch_size = batch_size,
                            target_function = TD1_target, device = gpu)
                time = @elapsed run(agent_d, env, stop_iterations = stop_iterations)=#
                time = 0.
                #test on each forecast
                BSON.@load "data/single-item/agents/ppo_agent_$agent_id.bson" agent 
                agent_d = device(agent)
                ppo_df = DataFrame(leadtime = Int[], shortage = Float64[], setup = Int[], lostsales = Bool[], CV = Float64[], policy = String[], mu_dist_bounds = String[], avg_cost = Float64[], MC_std = [], train_time_s = Float64[], forecast_id = Int[], agent_id=Int[])
                Random.seed!(agent_id*1000) #In case we must reevaluate agents but not retrain
                @showprogress "Benchmarking..." for (f_ID, forecast) in collect(enumerate(forecasts))
                    test_env = sl_sip(holding, shortage, setup, CV, 0, forecast, leadtime*μ, leadtime, lostsales = lostsale, horizon = forecast_horizon, policy = policy, periods = test_periods)
                    cost, std = test_agent(agent_d, test_env, 1000)
                    push!(ppo_df, [leadtime, shortage, setup, lostsale, CV, String(Symbol(policy))[1:end-2],"($(μ_distribution.a), $(μ_distribution.b))", -cost, std, time, f_ID, agent_id])
                end
                agent = cpu(agent_d)
                CSV.write("data/single-item/ppo_testbed.csv", ppo_df, append = true)
                #BSON.@save "data/single-item/agents/ppo_agent_$agent_id.bson" agent
            end  
        end
    end
end
ppo_testbed()