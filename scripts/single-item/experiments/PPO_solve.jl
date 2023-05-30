using Distributed, InteractiveUtils
@everywhere include("experiment_parameters.jl")

versioninfo()
println("\nCUDA\n----")
CUDA.versioninfo()

CSV.write("data/single-item/ppo_testbed.csv", DataFrame(leadtime = Int[], shortage = Float64[], setup = Int[], lostsales = Bool[], CV = Float64[], horizon = Int[], ADI = Bool[], policy = String[], mu_dist_bounds = String[], avg_cost = Float64[], MC_std = [], train_time_s = Float64[], forecast_id = Int[], agent_id=Int[]))

#train N agents per setup
function ppo_testbed()
    N = 10 # agents trained per environment
    for a in 1:N
        @sync @distributed for (id, (leadtime, shortage, setup, CV, lostsale, forecast_horizon)) in collect(enumerate(zip(i_leadtimes, i_shortages, i_setups, i_CVs, i_lostsales, i_horizons)))
            println("Solving LT= $leadtime, b=$shortage, K=$setup, CV = $CV, lostsales = $lostsale, policy = $policy, μ_distribution = $μ_distribution, horizon = $forecast_horizon")            
            agent_id = (a-1)*n_instances + id
            Random.seed!(agent_id) #to reprocude correctly if restarted
            devi = myid()%length(CUDA.devices())
            CUDA.device!(devi) #one GPU per worker if nprocs = ndevices
            println("agent $agent_id / $(N*n_instances) using device $devi")
            env = sl_sip(holding, shortage, setup, 0.,μ_distribution, Uniform(-5*(leadtime+1)*(1-lostsale),10*(leadtime + 1)), leadtime, lostsales = lostsale, horizon = forecast_horizon, periods = steps_per_episode, policy = policy, d_type = CVNormal{CV})
            agent_d = PPOPolicy(env, actor_optimiser = Scheduler(actor_schedule, ADAM()),  critic_optimiser = Scheduler(critic_schedule, ADAM()), n_hidden = 128,
                        γ = 0.99f0,λ = 0.90f0, clip_range = 0.2f0, entropy_weight = 1f-3, n_actors = n_actors, n_epochs = n_epochs, batch_size = batch_size,
                        target_function = TD1_target, device = gpu)
            tester = TestEnvironment(sl_sip(holding, shortage, setup, 0, forecasts[2], leadtime*μ, leadtime, lostsales = lostsale, horizon = forecast_horizon, periods = test_periods,d_type=CVNormal{CV}), 100, 100)
            tester2 = TestEnvironment(sl_sip(holding, shortage, setup, 0, forecasts[6], leadtime*μ, leadtime, lostsales = lostsale, horizon = forecast_horizon, periods = test_periods,d_type=CVNormal{CV}), 100, 100)
            tester3 = TestEnvironment(sl_sip(holding, shortage, setup, 0, forecasts[1], leadtime*μ, leadtime, lostsales = lostsale, horizon = forecast_horizon, periods = test_periods,d_type=CVNormal{CV}), 100, 100)
            es = Sequence(Exp(1f-3, 10f0^(1f0/warmup_iterations)) => warmup_iterations, Exp(1f-2, (1f0/100)^(1f0/(stop_iterations-warmup_iterations))) => (stop_iterations - warmup_iterations))
            time = @elapsed run(agent_d, env, stop_iterations = stop_iterations, hook = Hook(EntropyWeightDecayer(es), tester, tester2, tester3, EpsilonDecayer(Shifted(Triangle(λ0 = 0.2, λ1 = 0.05, period = 2*stop_iterations), stop_iterations))), show_progress = false);
            #test on each forecast
            println("$agent_id done in $time seconds")
            println(lineplot(first.(tester.log)[22:end]))
            println("Benchmarking...")
            
            ppo_df = DataFrame(leadtime = Int[], shortage = Float64[], setup = Int[], lostsales = Bool[], CV = Float64[],horizon = Int[], ADI = Bool[], policy = String[], mu_dist_bounds = String[], avg_cost = Float64[], MC_std = [], train_time_s = Float64[], forecast_id = Int[], agent_id=Int[])
            Random.seed!(agent_id*1000) #In case we must reevaluate agents but not retrain
            for (f_ID, forecast) in collect(enumerate(forecasts))
                test_env = sl_sip(holding, shortage, setup,0, forecast, leadtime*μ, leadtime, lostsales = lostsale, horizon = forecast_horizon, policy = policy, periods = test_periods,d_type=CVNormal{CV})
                cost, std = test_agent(agent_d, test_env, 1000)
                push!(ppo_df, [leadtime, shortage, setup, lostsale, CV, forecast_horizon, false, String(Symbol(policy))[1:end-2], "($(μ_distribution.a), $(μ_distribution.b))", -cost, std, time, f_ID, agent_id])
            end
            CSV.write("data/single-item/ppo_testbed.csv", ppo_df, append = true)
            agent = cpu(agent_d)
            BSON.@save "data/single-item/agents/sdi/ppo_agent_$agent_id.bson" agent
        end  
    end
end
ppo_testbed()
