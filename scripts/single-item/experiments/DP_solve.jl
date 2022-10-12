using Distributed, InteractiveUtils

@everywhere include("experiment_parameters.jl")

function solve_DP()
    println("Solving SDI with DP")
    CSV.write("data/single-item/scarf_testbed_DP.csv", DataFrame(leadtime = Int[], shortage = Float64[], setup = Int[], lostsales = Bool[], CV = Float64[], horizon = Int[], forecast_id = Int[], opt_cost = Float64[], opt_MC_std = Float64[], solve_time_s = Float64[]))
    #p = Progress(length(forecasts)*n_instances)
    @sync @distributed for (id, (leadtime, shortage, setup, CV, lostsale, forecast_horizon)) in collect(enumerate(zip(i_leadtimes, i_shortages, i_setups, i_CVs, i_lostsales, i_horizons)))
        println("Solving LT= $leadtime, b=$shortage, K=$setup, CV = $CV, lostsales = $lostsale, policy = $policy, μ_distribution = $μ_distribution, horizon = $forecast_horizon")      
        println("Solving $id/$(length(i_horizons))")
        Random.seed!(id)
        for (f_ID, forecast) in collect(enumerate(forecasts))
            if f_ID != 1
                continue
            end
            scarf_df = DataFrame(leadtime = Int[], shortage = Float64[], setup = Int[], lostsales = Bool[], CV = Float64[], horizon = Int[], forecast_id = Int[], avg_cost = Float64[], MC_std = Float64[], solve_time_s = Float64[])
            test_env = sl_sip(holding, shortage, setup, CV, 0, forecast, leadtime*μ, leadtime, lostsales = lostsale, horizon = forecast_horizon, policy = policy, periods = test_periods)
            cost, std, time = test_rolling_ss_policy(test_env, 1000, horizon = forecast_horizon)
            push!(scarf_df, [leadtime, shortage, setup, lostsale, CV, forecast_horizon, f_ID, cost, std, time])
            CSV.write("data/single-item/scarf_testbed_DP.csv", scarf_df, append = true)
            #ProgressMeter.next!(p)
        end
    end
end
solve_DP()

function solve_simple()
    println("Solving SDI with simple (s,S)")
    CSV.write("data/single-item/scarf_testbed_simple.csv", DataFrame(leadtime = Int[], shortage = Float64[], setup = Int[], lostsales = Bool[], CV = Float64[], horizon = Int[], forecast_id = Int[], opt_cost = Float64[], opt_MC_std = Float64[], solve_time_s = Float64[]))
    #p = Progress(length(forecasts)*n_instances)
    @sync @distributed for (id, (leadtime, shortage, setup, CV, lostsale, forecast_horizon)) in collect(enumerate(zip(i_leadtimes, i_shortages, i_setups, i_CVs, i_lostsales, i_horizons)))
        println("Solving $id/$(length(i_horizons))")
        Random.seed!(id)
        for (f_ID, forecast) in collect(enumerate(forecasts))
            scarf_df = DataFrame(leadtime = Int[], shortage = Float64[], setup = Int[], lostsales = Bool[], CV = Float64[], horizon = Int[], forecast_id = Int[], avg_cost = Float64[], MC_std = Float64[], solve_time_s = Float64[])
            test_env = sl_sip(holding, shortage, setup, CV, 0, forecast, leadtime*μ, leadtime, lostsales = lostsale, horizon = forecast_horizon, policy = policy, periods = test_periods)
            cost, std, time = test_simple_ss_policy(test_env, 1000, horizon = forecast_horizon)
            push!(scarf_df, [leadtime, shortage, setup, lostsale, CV, forecast_horizon, f_ID, cost, std, time])
            CSV.write("data/single-item/scarf_testbed_simple.csv", scarf_df, append = true)
            #ProgressMeter.next!(p)
        end
    end
end
#solve_simple()

function solve_adi_DP()
    println("Solving ADI with DP")
    mmfe_update = exp_multiplicative_mmfe(first_var, var_discount)
    CSV.write("data/single-item/scarf_testbed_adi_DP.csv", DataFrame(leadtime = Int[], shortage = Float64[], setup = Int[], lostsales = Bool[], CV = Float64[], horizon = Int[], forecast_id = Int[], opt_cost = Float64[], opt_MC_std = Float64[], solve_time_s = Float64[]))
    #p = Progress(length(forecasts)*n_instances)
    @sync @distributed for (id, (leadtime, shortage, setup, CV, lostsale, forecast_horizon)) in collect(enumerate(zip(i_leadtimes, i_shortages, i_setups, i_CVs, i_lostsales, i_horizons)))
        println("Solving $id/$(length(i_horizons))")
        Random.seed!(id)
        for (f_ID, forecast) in collect(enumerate(forecasts))
            scarf_df = DataFrame(leadtime = Int[], shortage = Float64[], setup = Int[], lostsales = Bool[], CV = Float64[], horizon = Int[], forecast_id = Int[], avg_cost = Float64[], MC_std = Float64[], solve_time_s = Float64[])
            test_env = sl_sip(holding, shortage, setup, CV, 0, forecast, leadtime*μ, leadtime, lostsales = lostsale, horizon = forecast_horizon, policy = policy, periods = test_periods) |> e -> SingleItemMMFE(e, mmfe_update)
            cost, std, time = test_rolling_ss_policy(test_env, 1000, horizon = forecast_horizon)
            push!(scarf_df, [leadtime, shortage, setup, lostsale, CV, forecast_horizon, f_ID, cost, std, time])
            CSV.write("data/single-item/scarf_testbed_adi_DP.csv", scarf_df, append = true)
            #ProgressMeter.next!(p)
        end
    end
end
#solve_adi_DP()

function solve_adi_simple()
    println("Solving ADI with simple (s,S)")
    mmfe_update = exp_multiplicative_mmfe(first_var, var_discount)
    CSV.write("data/single-item/scarf_testbed_adi_simple.csv", DataFrame(leadtime = Int[], shortage = Float64[], setup = Int[], lostsales = Bool[], CV = Float64[], horizon = Int[], forecast_id = Int[], opt_cost = Float64[], opt_MC_std = Float64[], solve_time_s = Float64[]))
    #p = Progress(length(forecasts)*n_instances)
    @sync @distributed for (id, (leadtime, shortage, setup, CV, lostsale, forecast_horizon)) in collect(enumerate(zip(i_leadtimes, i_shortages, i_setups, i_CVs, i_lostsales, i_horizons)))
        println("Solving $id/$(length(i_horizons))")
        Random.seed!(id)
        for (f_ID, forecast) in collect(enumerate(forecasts))
            scarf_df = DataFrame(leadtime = Int[], shortage = Float64[], setup = Int[], lostsales = Bool[], CV = Float64[], horizon = Int[], forecast_id = Int[], avg_cost = Float64[], MC_std = Float64[], solve_time_s = Float64[])
            test_env = sl_sip(holding, shortage, setup, CV, 0, forecast, leadtime*μ, leadtime, lostsales = lostsale, horizon = forecast_horizon, policy = policy, periods = test_periods) |> e -> SingleItemMMFE(e, mmfe_update)
            cost, std, time = test_simple_ss_policy(test_env, 1000, horizon = forecast_horizon)
            push!(scarf_df, [leadtime, shortage, setup, lostsale, CV, forecast_horizon, f_ID, cost, std, time])
            CSV.write("data/single-item/scarf_testbed_adi_simple.csv", scarf_df, append = true)
            #ProgressMeter.next!(p)
        end
    end
end
#solve_adi_simple()
