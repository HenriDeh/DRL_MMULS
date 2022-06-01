using DRL_MMULS
using InventoryModels
using Distributions, Flux, BSON, CSV, DataFrames, ProgressMeter, Random
using CUDA
using UnicodePlots, ParameterSchedulers
import ParameterSchedulers.Scheduler

#opt = 6791
holding = 1
shortage = 75
setup = 320
CV = 0.2
μ_distribution = Uniform(0,20)
leadtime = 8
lostsale = true
policy = sSPolicy()
μ = 10
forecast_horizon = 32

forecast_df = CSV.read("data/single-item/forecasts.csv", DataFrame)
forecasts = Vector{Float64}[]
for forecast_string in forecast_df.forecast
    f = eval(Meta.parse(forecast_string))
    push!(forecasts, f)
end

steps_per_episode = 104
batch_size = 252
n_actors = 10
stop_iterations = 15000
warmup_iterations = 1000
n_epochs = 5

actor_updates = stop_iterations*n_actors*steps_per_episode÷batch_size
critic_updates = actor_updates*n_epochs
warmup = Int(round(warmup_iterations/stop_iterations*actor_updates))

actor_warmup = Triangle(λ0 = 1f-6, λ1 = 1f-5, period = 2*warmup)
actor_sinexp = SinExp(λ0 = 1f-5, λ1 = 1f-4, period = (actor_updates-2*warmup)÷5, γ = 1f0/10^(1f0/(0.5f0actor_updates)))
actor_schedule = Sequence(actor_warmup => warmup, actor_sinexp => actor_updates - 2*warmup, Shifted(actor_warmup, warmup) => warmup)
critic_schedule = SinExp(λ0 = 1f-5, λ1 = 1f-4, period = (actor_updates-2*warmup)÷5*n_epochs, γ = 1f0/10^(1f0/0.5f0critic_updates))

fc = forecasts[6] #35716

env = sl_sip(holding, shortage, setup, CV, 0.,μ_distribution, Uniform(-5*(leadtime+1),10*(leadtime + 1)), leadtime, lostsales = lostsale, horizon = forecast_horizon, periods = steps_per_episode, policy = policy)
agent_d = PPOPolicy(env, actor_optimiser = Scheduler(actor_schedule, ADAM()), critic_optimiser = Scheduler(critic_schedule, ADAM()), n_hidden = 128,
            γ = 0.99f0,λ = 0.90f0, clip_range = 0.2f0, entropy_weight = 1f-2, n_actors = n_actors, n_epochs = n_epochs, batch_size = batch_size,
            target_function = TD1_target, device = gpu)

tester = TestEnvironment(sl_sip(holding, shortage, setup, CV, 0, fc, leadtime*μ, leadtime, lostsales = lostsale, horizon = forecast_horizon, periods = 400), 100, 100)
run(agent_d, env, stop_iterations = stop_iterations, hook = Hook(tester))
p = lineplot(first.(tester.log)[131:end], ylim = (-6000, -3500));




BSON.@load "data/single-item/agents/ppo_agent_114.bson" agent 
test_env = sl_sip(holding, shortage, setup, CV, 0, fc, leadtime*μ, leadtime, lostsales = lostsale, horizon = forecast_horizon, periods = 400)
ins = Scarf.Instance(test_env, 0.99)
ins.backlog = true
Scarf.DP_sS(ins,0.1)
scarf = test_ss_policy(test_env, ins.s, ins.S)

ag = test_agent(agent, test_env)
gap = -ag[1]/scarf[1] - 1
