using Distributed, InteractiveUtils
@everywhere include("experiment_parameters_lostsales.jl")

versioninfo()
println("\nCUDA\n----")
CUDA.versioninfo()

CSV.write("data/single-item/ppo_testbed_lostsales.csv", DataFrame(leadtime = Int[], shortage = Float64[], setup = Int[], lostsales = Bool[], CV = Float64[], horizon = Int[], ADI = Bool[], policy = String[], mu_dist_bounds = String[], avg_cost = Float64[], MC_std = [], train_time_s = Float64[], forecast_id = Int[], agent_id=Int[],stockoutrate=[]))

#train N agents per setup
function ppo_testbed()
    N = 10 # agents trained per environment
    for a in 1:N
        @sync @distributed for (id, (leadtime, shortage, setup, lostsale)) in collect(enumerate(Iterators.product([2,4,8],[5,10],[10,20,30], [true])))
            CV, forecast_horizon = 0.2, 32
	    d_type = Poisson
	    println("Solving LT= $leadtime, b=$shortage, K=$setup, CV = $CV, lostsales = $lostsale, policy = $policy, μ_distribution = $μ_distribution, horizon = $forecast_horizon")            
            agent_id = (a-1)*n_instances + id
            Random.seed!(agent_id) #to reprocude correctly if restarted
            devi = myid()%length(CUDA.devices())
            CUDA.device!(devi) #one GPU per worker if nprocs = ndevices
            println("agent $agent_id / $(N*n_instances) using device $devi")
            env = sl_sip(holding, shortage, setup, 0.,Uniform(0,20), Uniform(-5*(leadtime+1)*(1-lostsale),10*(leadtime + 1)), leadtime, lostsales = lostsale, horizon = forecast_horizon, periods = steps_per_episode, policy = policy,d_type=d_type)
            agent_d = PPOPolicy(env, actor_optimiser = Scheduler(actor_schedule, ADAM()),  critic_optimiser = Scheduler(critic_schedule, ADAM()), n_hidden = 128,
                    γ = 0.99f0,λ = 0.90f0, clip_range=0.2f0, entropy_weight = 1f-2, n_actors = n_actors, n_epochs = n_epochs, batch_size = batch_size,
                        target_function = TD1_target, device = gpu)
            tester = TestEnvironment(sl_sip(holding, shortage, setup, 0, forecasts[2], leadtime*μ, leadtime, lostsales = lostsale, horizon = forecast_horizon, periods = test_periods, d_type=d_type), 100, 100)
            tester2 = TestEnvironment(sl_sip(holding, shortage, setup, 0, forecasts[6], leadtime*μ, leadtime, lostsales = lostsale, horizon = forecast_horizon, periods = test_periods,d_type=d_type), 100, 100)
            tester3 = TestEnvironment(sl_sip(holding, shortage, setup, 0, forecasts[1], leadtime*μ, leadtime, lostsales = lostsale, horizon = forecast_horizon, periods = test_periods, d_type=d_type), 100, 100)
            es = Sequence(1f-2 => stop_iterations, Exp(1f-2,(1f0/10)^(1f0/(stop_iterations-warmup_iterations))) => (stop_iterations - warmup_iterations)) 
            
            time = @elapsed run(agent_d, env, stop_iterations = stop_iterations, hook = Hook(EntropyWeightDecayer(es), tester, tester2, tester3, EpsilonDecayer(Shifted(Triangle(λ0 = 0.2, λ1 = 0.05, period = 2*stop_iterations), stop_iterations))), show_progress = false);
            #test on each forecast
            println("$agent_id done in $time seconds")
            println(lineplot(first.(tester.log)[22:end]))
            println("Benchmarking...")
            
            ppo_df = DataFrame(leadtime = Int[], shortage = Float64[], setup = Int[], lostsales = Bool[], CV = Float64[],horizon = Int[], ADI = Bool[], policy = String[], mu_dist_bounds = String[], avg_cost = Float64[], MC_std = [], train_time_s = Float64[], forecast_id = Int[], agent_id=Int[],shortrate =Float64[])
            Random.seed!(agent_id*1000) #In case we must reevaluate agents but not retrain
            for (f_ID, forecast) in collect(enumerate(forecasts))
                test_env = TestEnvironment(sl_sip(holding, shortage, setup, 0, forecast, leadtime*μ, leadtime, lostsales = lostsale, horizon = forecast_horizon, policy = policy, periods = test_periods,d_type=d_type),1000)
                cost = test_agent(agent_d, test_env)
		std = last(test_env.log)[2]
		dfa = test_env.logger.logs["product"]
		fillrate = sum(dfa.market_backorder)/sum(dfa.market_demand)
                push!(ppo_df, [leadtime, shortage, setup, lostsale, CV, forecast_horizon, false, String(Symbol(policy))[1:end-2], "($(μ_distribution.a), $(μ_distribution.b))", -cost, std, time, f_ID, agent_id,fillrate])
            end
            CSV.write("data/single-item/ppo_testbed_lostsales.csv", ppo_df, append = true)
            agent = cpu(agent_d)
            BSON.@save "data/single-item/agents/sdi/ppo_lostsales_agent_$agent_id.bson" agent
        end  
    end
end
ppo_testbed()
