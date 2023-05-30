using Distributed, InteractiveUtils

@everywhere include("experiment_parameters_lostsales.jl")

function solve_DP()
    println("Solving Lost Sales with DP")
    CSV.write("data/single-item/scarf_testbed_DP_lostsales10.csv", DataFrame(leadtime = Int[], shortage = Float64[], setup = Int[], lostsales = Bool[], CV = Float64[], horizon = Int[], forecast_id = Int[], opt_cost = Float64[], opt_MC_std = Float64[], solve_time_s = Float64[], opt_shortrate = []))
    #p = Progress(length(forecasts)*n_instances)
    it = Iterators.product([2,4,8],[5,10],[10,20,30], [true])
    for (id, (leadtime, shortage, setup, lostsale)) in collect(enumerate(it))
        CV, forecast_horizon = 0.2, 32
        d_type = Poisson
        println("Solving LT= $leadtime, b=$shortage, K=$setup, CV = $CV, lostsales = $lostsale, policy = $policy, μ_distribution = $μ_distribution, horizon = $forecast_horizon")      
        println("Solving $id/$(length(it))")
        Random.seed!(id)
        scarf_df = DataFrame(leadtime = Int[], shortage = Float64[], setup = Int[], lostsales = Bool[], CV = Float64[], horizon = Int[], forecast_id = Int[], avg_cost = Float64[], MC_std = Float64[], solve_time_s = Float64[], shortrate = Float64[])
        for (f_ID, forecast) in collect(enumerate(forecasts))
            test_env = TestEnvironment(sl_sip(holding, shortage, setup, 0, forecast, leadtime*μ, leadtime, lostsales = lostsale, horizon = forecast_horizon, policy = policy, periods = test_periods, d_type = d_type),100,100)
            cost, std, time = test_rolling_ss_policy(test_env, 1000)
            dfa = test_env.logger.logs["product"]
            shortrate = sum(dfa.market_backorder)/sum(dfa.market_demand)
            push!(scarf_df, [leadtime, shortage, setup, lostsale, CV, forecast_horizon, f_ID, cost, std, time, shortrate])
            #ProgressMeter.next!(p)
        end
        CSV.write("data/single-item/scarf_testbed_DP_lostsales10.csv", scarf_df, append = true)
    end
end
solve_DP()

function solve_simple()
    println("Solving SDI with simple (s,S)")
    CSV.write("data/single-item/scarf_testbed_simple_lostsales10.csv", DataFrame(leadtime = Int[], shortage = Float64[], setup = Int[], lostsales = Bool[], CV = Float64[], horizon = Int[], forecast_id = Int[], simple_cost = Float64[], simple_MC_std = Float64[], simple_solve_time_s = Float64[], shortrate = []))
    #p = Progress(length(forecasts)*n_instances)
    it = Iterators.product([2,4,8],[5,10],[10,20,30], [true])
    for (id, (leadtime, shortage, setup, lostsale)) in collect(enumerate(it))
        CV, forecast_horizon = 0.2, 32
        d_type = CVNormal{CV}
        println("Solving LT= $leadtime, b=$shortage, K=$setup, CV = $CV, lostsales = $lostsale, policy = $policy, μ_distribution = $μ_distribution, horizon = $forecast_horizon")      
        println("Solving $id/$(length(it))")
        Random.seed!(id)
        scarf_df = DataFrame(leadtime = Int[], shortage = Float64[], setup = Int[], lostsales = Bool[], CV = Float64[], horizon = Int[], forecast_id = Int[], avg_cost = Float64[], MC_std = Float64[], solve_time_s = Float64[], shortrate = Float64[])
        for (f_ID, forecast) in collect(enumerate(forecasts))
            test_env = TestEnvironment(sl_sip(holding, shortage, setup, 0, forecast, leadtime*μ, leadtime, lostsales = lostsale, horizon = forecast_horizon, policy = policy, periods = test_periods, d_type = d_type),100,100)
            cost, std, time = test_simple_ss_policy(test_env, 1000)
            dfa = test_env.logger.logs["product"]
            shortrate = sum(dfa.market_backorder)/sum(dfa.market_demand)
            push!(scarf_df, [leadtime, shortage, setup, lostsale, CV, forecast_horizon, f_ID, cost, std, time, shortrate])
            #ProgressMeter.next!(p)
        end
        CSV.write("data/single-item/scarf_testbed_simple_lostsales10.csv", scarf_df, append = true)
    end
end
solve_simple()
