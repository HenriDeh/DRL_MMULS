using DRL_MMULS
using InventoryModels
using Distributions, Flux, BSON, CSV, DataFrames, ProgressMeter
using Random
include("../../src/testbed/sspolicy_test.jl")
Random.seed!(2021)

#device selection, change to `cpu` if you do not have an Nvidia compatible GPU.
device = gpu
#default parameters of train and test environments.
holding = 1.
leadtime_d = 2
shortage_db = 25.
shortages_dls = 75.
setup_d = 320.
CV_d = 0.5
#test values
leadtimes = [0, 1, 4,8]
shortages_b = [10, 25, 50]
shortages_ls = [50, 75, 100]
setups = [80., 1280.]
CVs = [0.3, 0.7]
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

forecast_horizon = 32

#train parameters
μ_distributions =  [Uniform(1,19)]#, Uniform(5,15), Uniform(1,10), Uniform(20, 30)]

#parameters of agents
policies = [sSPolicy(), QPolicy(), RQPolicy()]

#load forecasts, take only 52 periods from the data
forecast_df = CSV.read("data/single-item/forecasts.csv", DataFrame)
forecasts = Vector{Float64}[]
for forecast_string in forecast_df.forecast
    f = eval(Meta.parse(forecast_string))
    push!(forecasts, f)
end

#presolve testbed
function solve()
    CSV.write("data/single-item/scarf_testbed.csv", DataFrame(leadtime = Int[], shortage = Float64[], setup = Int[], lostsales = Bool[], CV = Float64[], forecast_id = Int[], opt_cost = Float64[], opt_MC_std = Float64[], solve_time_s = Float64[]))
    for (leadtime, shortage, setup, CV, lostsale) in zip(i_leadtimes, i_shortages, i_setups, i_CVs, i_lostsales) 
        scarf_df = DataFrame(leadtime = Int[], shortage = Float64[], setup = Int[], lostsales = Bool[], CV = Float64[], forecast_id = Int[], avg_cost = Float64[], MC_std = Float64[], solve_time_s = Float64[])
        println("Solving LT= $leadtime, b=$shortage, K=$setup, CV = $CV, lostsales = $lostsale")
        prog = Progress(length(forecasts))
        for (f_ID, forecast) in collect(enumerate(forecasts))
            env = sl_sip(holding, shortage, setup, CV, 0, forecast, 0.0, leadtime, lostsales = lostsale, horizon = 32)
            instance = Scarf.Instance(env, 0.99)
            instance.backlog = true
            time = @elapsed Scarf.DP_sS(instance, 0.1)
            (cost, std) = test_ss_policy(env, instance.s, instance.S)
            push!(scarf_df, [leadtime, shortage, setup, lostsale, CV, f_ID, cost, std, time])
            next!(prog)
        end
        CSV.write("data/single-item/scarf_testbed.csv", scarf_df, append = true)
    end
end
solve()

#train N agents per setup
function ppo_testbed()
    N = 10 # agents trained per environment
    stop_iterations = 10000 
    CSV.write("data/single-item/ppo_testbed.csv", DataFrame(leadtime = Int[], shortage = Float64[], setup = Int[], lostsales = Bool[], CV = Float64[], policy = String[], mu_dist_bounds = String[], avg_cost = Float64[], MC_std = [], train_time_s = Float64[], forecast_id = Int[], agent_id=Int[]))
    agent_id = 0
    for μ_distribution in μ_distributions,
        policy in policies,
        lostsale in lostsales
        for (leadtime, shortage, setup, CV) in zip(i_leadtimes, i_shortages, i_setups, i_CVs) 
            println("Solving LT= $leadtime, b=$shortage, K=$setup, CV = $CV, lostsales = $lostsale, policy = $policy, μ_distribution = $μ_distribution")
            for n in 1:N
                ppo_df = DataFrame(leadtime = Int[], shortage = Float64[], setup = Int[], lostsales = Bool[], CV = Float64[], policy = String[], mu_dist_bounds = String[], avg_cost = Float64[], MC_std = Float64[], train_time_s = Float64[], forecast_id = Int[], agent_id=Int[])
                agent_id += 1
                println("$n/$N")
                env = sl_sip(holding, shortage, 0., CV, 0, μ_distribution, Uniform(-10,20), leadtime, lostsales = lostsale, horizon = 32, policy = policy, periods = 52)
                
                agent_d = PPOPolicy(env, actor_optimiser = ADAM(3f-4), critic_optimiser = ADAM(3f-4), n_hidden = 128,
                γ = 0.99f0,λ = 0.95f0, clip_range = 0.2f0, entropy_weight = 1f-2, n_actors = 25, n_epochs = 10, batch_size = 128,
                target_function = TDλ_target, device = device)
                
                tester = TestEnvironment(sl_sip(holding, shortage, setup, CV, 0, fill(10.0,52), 0.0, leadtime, lostsales = lostsale, horizon = 32, policy = policy), 100, 100)
                time = @elapsed run(agent_d, env, stop_iterations = stop_iterations, hook = Hook(tester, Kscheduler(0,setup, 1000:(stop_iterations-1000))))
                #test on each forecast
                for (f_ID, forecast) in collect(enumerate(forecasts))
                    test_env = sl_sip(holding, shortage, setup, CV, 0, forecast, 0.0, leadtime, lostsales = lostsale, horizon = 32, policy = policy)
                    cost, std = test_agent(agent_d, test_env, 1000)
                    push!(ppo_df, [leadtime, shortage, setup, lostsale, CV, String(Symbol(policy))[1:end-2],"($(μ_distribution.a), $(μ_distribution.b))", -cost, std, time, f_ID, agent_id])
                end
                agent = cpu(agent_d)
                CSV.write("data/single-item/ppo_testbed.csv", ppo_df, append = true)
                BSON.@save "data/single-item/agents/ppo_agent_$agent_id.bson" agent tester
            end
        end   
    end
end
ppo_testbed()

#=
#data inspection
=#
using CSV, DataFrames, Statistics
df_ppo = CSV.read("data/single-item/ppo_testbed.csv", DataFrame)
df_opt = CSV.read("data/single-item/scarf_testbed.csv", DataFrame)
df = innerjoin(df_ppo, df_opt, on = [:leadtime, :shortage, :setup, :lostsales, :CV, :forecast_id])
df.gap = df.avg_cost ./ df.opt_cost .-1
gdf  = groupby(df, [:leadtime, :shortage, :setup, :lostsales, :CV, :policy, :agent_id])
df2 = combine(gdf, :gap => mean)
gdf = groupby(df2, [:leadtime, :shortage, :setup, :lostsales, :CV, :policy])

gdf_opt = groupby(df_opt, [:leadtime, :shortage, :setup, :lostsales, :CV])
df_opt = combine(gdf_opt, :avg_cost => mean)