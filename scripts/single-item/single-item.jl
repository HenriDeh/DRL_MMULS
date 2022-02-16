using DRL_MMULS
using InventoryModels
using Distributions, Flux, BSON, CSV, DataFrames, ProgressMeter
using CUDA

#device selection, change to `cpu` if you do not have an Nvidia compatible GPU.
device = gpu
#default parameters of train and test environments.
μ = 10.
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
μ_distributions =  [Uniform(0.2μ,1.8μ)]#, Uniform(5,15), Uniform(1,10), Uniform(20, 30)]

#parameters of agents
policies = [sSPolicy(), RQPolicy(), QPolicy()]

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
    p = Progress(500*22)
    Threads.@threads for (leadtime, shortage, setup, CV, lostsale) in collect(zip(i_leadtimes, i_shortages, i_setups, i_CVs, i_lostsales))
        scarf_df = DataFrame(leadtime = Int[], shortage = Float64[], setup = Int[], lostsales = Bool[], CV = Float64[], forecast_id = Int[], avg_cost = Float64[], MC_std = Float64[], solve_time_s = Float64[])
        for (f_ID, forecast) in collect(enumerate(forecasts))
            env = sl_sip(holding, shortage, setup, CV, 0, forecast, leadtime*μ, leadtime, lostsales = lostsale, horizon = 32, periods = 20)
            instance = Scarf.Instance(env, 0.99)
            instance.backlog = true
            time = @elapsed Scarf.DP_sS(instance, 0.1)
            (cost, std) = test_ss_policy(env, instance.s, instance.S)
            push!(scarf_df, [leadtime, shortage, setup, lostsale, CV, f_ID, cost, std, time])
            ProgressMeter.next!(p)
        end
        CSV.write("data/single-item/scarf_testbed.csv", scarf_df, append = true)
    end
end
#solve()#uncomment this to resolve the testbed with DP method

#train N agents per setup
function ppo_testbed()
    N = 10 # agents trained per environment
    stop_iterations = 10000 
    CSV.write("data/single-item/ppo_testbed.csv", DataFrame(leadtime = Int[], shortage = Float64[], setup = Int[], lostsales = Bool[], CV = Float64[], policy = String[], mu_dist_bounds = String[], avg_cost = Float64[], MC_std = [], train_time_s = Float64[], forecast_id = Int[], agent_id=Int[]))
    agent_id = 0
    for μ_distribution in μ_distributions,
        policy in policies
        for (leadtime, shortage, setup, CV, lostsale) in zip(i_leadtimes, i_shortages, i_setups, i_CVs, i_lostsales) 
            println("Solving LT= $leadtime, b=$shortage, K=$setup, CV = $CV, lostsales = $lostsale, policy = $policy, μ_distribution = $μ_distribution")
            for n in 1:N
                agent_id += 1
                
                env = sl_sip(holding, shortage, setup, CV, 0.,μ_distribution, Uniform(-5*(leadtime+1),10*(leadtime + 1)), leadtime, lostsales = lostsale, horizon = 32, periods = 52, policy = policy)
                agent_d = PPOPolicy(env, actor_optimiser = ADAM(3f-5), critic_optimiser = ADAM(3f-4), n_hidden = 128,
                            γ = 0.99f0,λ = 0.90f0, clip_range = 0.2f0, entropy_weight = 1f-2, n_actors = 40, n_epochs = 5, batch_size = 256,
                            target_function = TD1_target, device = gpu)
                
                tester = TestEnvironment(sl_sip(holding, shortage, setup, CV, 0, fill(10.0,52), 0.0, leadtime, lostsales = lostsale, horizon = 32, policy = policy), 100, 100)
                time = @elapsed run(agent_d, env, stop_iterations = stop_iterations, hook = Hook(tester))
                #test on each forecast
                ppo_df = DataFrame(leadtime = Int[], shortage = Float64[], setup = Int[], lostsales = Bool[], CV = Float64[], policy = String[], mu_dist_bounds = String[], avg_cost = Float64[], MC_std = [], train_time_s = Float64[], forecast_id = Int[], agent_id=Int[])
                @showprogress "Benchmarking..." for (f_ID, forecast) in collect(enumerate(forecasts))
                    test_env = sl_sip(holding, shortage, setup, CV, 0, forecast, leadtime*μ, leadtime, lostsales = lostsale, horizon = 32, policy = policy, periods = 20)
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