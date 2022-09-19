include("experiment_parameters.jl")

function solve()
    CSV.write("data/single-item/scarf_testbed.csv", DataFrame(leadtime = Int[], shortage = Float64[], setup = Int[], lostsales = Bool[], CV = Float64[], horizon = Int[], forecast_id = Int[], opt_cost = Float64[], opt_MC_std = Float64[], solve_time_s = Float64[]))
    p = Progress(length(forecasts)*n_instances)
    Threads.@threads for (leadtime, shortage, setup, CV, lostsale) in collect(zip(i_leadtimes, i_shortages, i_setups, i_CVs, i_lostsales))
        for (f_ID, forecast) in collect(enumerate(forecasts))
            scarf_df = DataFrame(leadtime = Int[], shortage = Float64[], setup = Int[], lostsales = Bool[], CV = Float64[], forecast_id = Int[], avg_cost = Float64[], MC_std = Float64[], solve_time_s = Float64[])
            env = sl_sip(holding, shortage, setup, CV, 0, forecast, leadtime*μ, leadtime, lostsales = lostsale, horizon = forecast_horizon, periods = test_periods)
            instance = Scarf.Instance(env, 0.99)
            instance.backlog = true
            time = @elapsed Scarf.DP_sS(instance, 1., zero_boundary = false)
            (cost, std) = test_ss_policy(env, instance.s, instance.S)
            push!(scarf_df, [leadtime, shortage, setup, lostsale, CV, test_periods, f_ID, cost, std, time])
            for horizon in horizons
                test_env = sl_sip(holding, shortage, setup, CV, 0, forecasts[i], leadtime*μ, leadtime, lostsales = lostsale, horizon = forecast_horizon, policy = policy, periods = test_periods)
                instance = Scarf.Instance(test_env, 0.99)
                instance.backlog = true
                cost, std = test_dynamic_ss_policy(test_env, 1000, horizon = horizon)
                push!(scarf_df, [leadtime, shortage, setup, lostsale, CV, horizon, f_ID, cost, std, time])
                CSV.write("data/single-item/scarf_testbed.csv", scarf_df, append = true)
            end
            ProgressMeter.next!(p)
        end
    end
end

solve()
