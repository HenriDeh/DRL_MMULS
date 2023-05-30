using DRL_MMULS
#using .InventoryModels
using Distributions, BSON, CSV, DataFrames, Random, DataFramesMeta, Plots

function test_penalties()
    #CSV.write("penalty.csv", DataFrame(shortage_cost=[], inventory_cost = [], setup_cost = [], total = [], p = [], K = [], L = [], opt = [], degenerate = [], shortrate = []))
    for (shortage, setup, leadtime) in Iterators.product(50:-2.5:5, 1:4, 1:4) #[5:5:45; 50:50:1200], 1:4)
        lostsale = true
        holding = 1
        #shortage = lostsale ? 8. : 25.
        #setup = 320
        d_type = CVNormal{0.2}
        μ_distribution = Uniform(0,20)
        #leadtime = 2
        policy = sSPolicy()
        μ = 10
        forecast_horizon = 32
        test_periods = 104

        forecast_df = CSV.read("data/single-item/forecasts.csv", DataFrame)
        forecasts = Vector{Float64}[]
        for forecast_string in forecast_df.forecast
            fs = eval(Meta.parse(forecast_string))
            push!(forecasts, fs)
        end
        fc = forecasts[2] #11000

        base_tester = TestEnvironment(sl_sip(holding, shortage, setup, 0, fc, leadtime*μ, leadtime, lostsales = lostsale, horizon = forecast_horizon, periods = test_periods, d_type = d_type), 100, 1000)
        #base_tester = TestEnvironment(SingleItemMMFE(sl_sip(holding, shortage, setup, 0, fc, leadtime*μ, leadtime, lostsales = lostsale, horizon = forecast_horizon, periods = test_periods, d_type = d_type),mmfe_update), 100, 100)
        final_tester = deepcopy(base_tester)
        stupid_tester = deepcopy(base_tester)

        opt, std, _ = test_rolling_ss_policy(base_tester, 1000)
        #println("opt = ", opt, "+-", 1.95*std/sqrt(1000))
        df = base_tester.logger.logs["product"]
        shortrate = sum(df.market_backorder)/sum(df.market_demand)
        stupid, _, _ = test_ss_policy(stupid_tester, zeros(104), zeros(104))

        sumdf = @combine groupby(df, :simulation_id) begin
            :shortage_cost = sum(:market_cost)
            :inventory_cost = sum(:inventory_cost)
            :setup_cost = sum(:supplier_cost)
            :total_cost = sum(:market_cost)+ sum(:inventory_cost) + sum(:supplier_cost)
        end
        meandf = @combine sumdf begin
            :shortage_cost = mean(:shortage_cost)
            :inventory_cost = mean(:inventory_cost)
            :setup_cost = mean(:setup_cost)
            :total_cost = mean(:total_cost)
        end

        @transform! meandf begin
            :p = shortage
            :K = setup
            :L = leadtime
            :opt = opt
            :degenerate = stupid
            :shortrate = shortrate
            :is_degenerate = stupid < opt
        end

        CSV.write("penalty.csv", meandf, append = true)
    end
end

df = CSV.read("penalty.csv", DataFrame)
@transform! df begin
    :prop_shortage = :shortage_cost ./ :total
    :is_degenerate = (:setup_cost .== 0 .|| (:degenerate .< :opt))
    :is_basestock = :setup_cost .>= :K .* (104 .- :L)
end

df = filter(df) do r
    r.shortrate < 0.95 &&
    #!r.is_degenerate &&
    !r.is_basestock
end
sort!(df, :prop_shortage)
filter!(df) do r
    r.K <=50
end
