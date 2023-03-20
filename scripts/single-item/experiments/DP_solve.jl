using Distributed, InteractiveUtils

@everywhere include("experiment_parameters.jl")
#=
function solve_DP()
    println("Solving SDI with DP")
    CSV.write("data/single-item/scarf_testbed_DP.csv", DataFrame(leadtime = Int[], shortage = Float64[], setup = Int[], lostsales = Bool[], CV = Float64[], horizon = Int[], forecast_id = Int[], opt_cost = Float64[], opt_MC_std = Float64[], solve_time_s = Float64[]))
    #p = Progress(length(forecasts)*n_instances)
    @sync @distributed for (id, (leadtime, shortage, setup, CV, lostsale, forecast_horizon)) in collect(enumerate(zip(i_leadtimes, i_shortages, i_setups, i_CVs, i_lostsales, i_horizons)))
        println("Solving LT= $leadtime, b=$shortage, K=$setup, CV = $CV, lostsales = $lostsale, policy = $policy, μ_distribution = $μ_distribution, horizon = $forecast_horizon")      
        println("Solving $id/$(length(i_horizons))")
        Random.seed!(id)
        scarf_df = DataFrame(leadtime = Int[], shortage = Float64[], setup = Int[], lostsales = Bool[], CV = Float64[], horizon = Int[], forecast_id = Int[], avg_cost = Float64[], MC_std = Float64[], solve_time_s = Float64[])
        for (f_ID, forecast) in collect(enumerate(forecasts))
            test_env = sl_sip(holding, shortage, setup, 0, forecast, leadtime*μ, leadtime, lostsales = lostsale, horizon = forecast_horizon, policy = policy, periods = test_periods, d_type = CVNormal{CV})
            cost, std, time = test_rolling_ss_policy(test_env, 1000)
            push!(scarf_df, [leadtime, shortage, setup, lostsale, CV, forecast_horizon, f_ID, cost, std, time])
            #ProgressMeter.next!(p)
        end
        CSV.write("data/single-item/scarf_testbed_DP.csv", scarf_df, append = true)
    end
end
solve_DP()

function solve_simple()
    println("Solving SDI with simple (s,S)")
    CSV.write("data/single-item/scarf_testbed_simple.csv", DataFrame(leadtime = Int[], shortage = Float64[], setup = Int[], lostsales = Bool[], CV = Float64[], horizon = Int[], forecast_id = Int[], simple_cost = Float64[], simple_MC_std = Float64[], simple_solve_time_s = Float64[]))
    #p = Progress(length(forecasts)*n_instances)
    @sync @distributed for (id, (leadtime, shortage, setup, CV, lostsale, forecast_horizon)) in collect(enumerate(zip(i_leadtimes, i_shortages, i_setups, i_CVs, i_lostsales, i_horizons)))
        println("Solving $id/$(length(i_horizons))")
        Random.seed!(id)
        scarf_df = DataFrame(leadtime = Int[], shortage = Float64[], setup = Int[], lostsales = Bool[], CV = Float64[], horizon = Int[], forecast_id = Int[], avg_cost = Float64[], MC_std = Float64[], solve_time_s = Float64[])
        for (f_ID, forecast) in collect(enumerate(forecasts))
            test_env = sl_sip(holding, shortage, setup, CV, 0, forecast, leadtime*μ, leadtime, lostsales = lostsale, horizon = forecast_horizon, policy = policy, periods = test_periods, d_type = CVNormal{CV})
            cost, std, time = test_simple_ss_policy(test_env, 1000, horizon = forecast_horizon)
            push!(scarf_df, [leadtime, shortage, setup, lostsale, CV, forecast_horizon, f_ID, cost, std, time])
            #ProgressMeter.next!(p)
        end
        CSV.write("data/single-item/scarf_testbed_simple.csv", scarf_df, append = true)
    end
end
solve_simple()=#

function solve_adi_DP()
    println("Solving ADI with DP")
    CSV.write("data/single-item/scarf_testbed_adi_DP.csv", DataFrame(lostsales = Bool[], first_var = Float64[], var_discount = Float64[], forecast_id = Int[], opt_cost = Float64[], opt_MC_std = Float64[], solve_time_s = Float64[]))
    leadtime = 2 
    setup = 320 
    CV = 0.2 
    forecast_horizon = 32
    first_vars = [0.1, 0.05, 0.03]
    var_discounts = [0.7, 0.8, 0.9, 0.95, 0.99]
    lostsales = [true, false] 
    d_type = CVNormal{CV}
    it = Iterators.product(lostsales, first_vars, var_discounts)
    n_instances = length(it)
    @sync @distributed for (id, (lostsale, first_var, var_discount)) in collect(enumerate(it))
        println("Solving $id/$(n_instances), first_var = $first_var, var_discount = $var_discount")
        mmfe_update = exp_multiplicative_mmfe(first_var, var_discount)
        shortage =  lostsale ? 75. : 25.
        Random.seed!(id)
        scarf_df = DataFrame(lostsales = Bool[], first_var = Float64[], var_discount = Float64[], forecast_id = Int[], opt_cost = Float64[], opt_MC_std = Float64[], solve_time_s = Float64[])
        for (f_ID, forecast) in collect(enumerate(forecasts))
            test_env = sl_sip(holding, shortage, setup, 0, forecast, leadtime*μ, leadtime, lostsales = lostsale, horizon = forecast_horizon, policy = policy, periods = test_periods, d_type = d_type) |> e -> SingleItemMMFE(e, mmfe_update)
            cost, std, time = test_rolling_ss_policy(test_env, 10)
            push!(scarf_df, [lostsale, first_var, var_discount, f_ID, cost, std, time])
        end
        CSV.write("data/single-item/scarf_testbed_adi_DP.csv", scarf_df, append = true)
    end
end
solve_adi_DP()

function solve_adi_simple()
    println("Solving ADI with simple (s,S)")
    CSV.write("data/single-item/scarf_testbed_adi_simple.csv", DataFrame(lostsales = Bool[], first_var = Float64[], var_discount = Float64[], forecast_id = Int[], opt_cost = Float64[], opt_MC_std = Float64[], solve_time_s = Float64[]))
    leadtime = 2 
    setup = 320 
    CV = 0.2 
    forecast_horizon = 32
    first_vars = [0.1, 0.05, 0.03]
    var_discounts = [0.7, 0.8, 0.9, 0.95, 0.99]
    lostsales = [true, false] 
    d_type = CVNormal{CV}
    it = Iterators.product(lostsales, first_vars, var_discounts)
    n_instances = length(it)
    @sync @distributed for (id, (lostsale, first_var, var_discount)) in collect(enumerate(it))
        println("Solving $id/$(n_instances), first_var = $first_var, var_discount = $var_discount")
        mmfe_update = exp_multiplicative_mmfe(first_var, var_discount)
        shortage =  lostsale ? 75. : 25.
        Random.seed!(id)
        scarf_df = DataFrame(lostsales = Bool[], first_var = Float64[], var_discount = Float64[], forecast_id = Int[], opt_cost = Float64[], opt_MC_std = Float64[], solve_time_s = Float64[])
        for (f_ID, forecast) in collect(enumerate(forecasts))
            test_env = sl_sip(holding, shortage, setup, 0, forecast, leadtime*μ, leadtime, lostsales = lostsale, horizon = forecast_horizon, policy = policy, periods = test_periods, d_type = d_type) |> e -> SingleItemMMFE(e, mmfe_update)
            cost, std, time = test_simple_ss_policy(test_env, 1000)
            push!(scarf_df, [lostsale, first_var, var_discount, f_ID, cost, std, time])
        end
        CSV.write("data/single-item/scarf_testbed_adi_simple.csv", scarf_df, append = true)
    end
end
solve_adi_simple()

