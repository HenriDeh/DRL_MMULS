using DRL_MMULS

using Distributions, Flux, BSON, CSV, DataFrames, CairoMakie, LaTeXStrings
using Random, ProgressMeter
begin
    forecast_df = CSV.read("data/single-item/forecasts.csv", DataFrame)
    forecasts = Vector{Float64}[]
    for forecast_string in forecast_df.forecast
        f = eval(Meta.parse(forecast_string))
        push!(forecasts, f)
    end
    begin 
        df_ppo = CSV.read("data/single-item/ppo_testbed.csv", DataFrame)
        df_opt = CSV.read("data/single-item/scarf_testbed_DP.csv", DataFrame)
        df_simple = CSV.read("data/single-item/scarf_testbed_simple.csv", DataFrame)
        forecast_df = CSV.read("data/single-item/forecasts.csv", DataFrame)
        df_opt = innerjoin(df_opt, forecast_df, on =(:forecast_id=> :ID))[:, Not(:forecast)]
        df = innerjoin(df_ppo, df_opt, df_simple, on = [:leadtime, :shortage, :setup, :lostsales, :CV, :forecast_id, :horizon]) #row = instance+forecast
        df.gap = -df.avg_cost ./ df.opt_cost .-1
        df.simple_gap = -df.avg_cost ./ df.simple_cost .-1
        gdf  = groupby(df, [:leadtime, :shortage, :setup, :lostsales, :CV, :policy, :agent_id, :horizon])
        df2 = combine(gdf, :gap .=> [mean,median], :simple_gap .=> [mean,median]) #row = agent
        df3 = filter(r->r.horizon==32 && r.lostsales == false,df2)
        shuffle!(df3)
    end
    set_theme!(Theme(Axis = (titlesize = 20f0, titlegap = -26.0, titlefont = "DejaVu Sans Bold"), linewidth = 2.5))
end
for r in eachrow(df3)[1:20]
    agent_id = r.agent_id
    println(agent_id)
    BSON.@load "data/single-item/agents/sdi/ppo_agent_$agent_id.bson" agent #median gap = 1.00%
    
    h = 1
    b = r.shortage
    K = r.setup
    CV = r.CV
    c = 0
    LT = r.leadtime
    lostsale = r.lostsales

    actor = agent.actor
    f = forecasts[4]
    fig = Figure(resolution = (800,800))
    a = Axis(fig[1,1], aspect = AxisAspect(1), xticks = 0:52:104)
    xlims!(a, (0, nothing))
    a.ylabel = L"\text{Output }(s_t,S_t)\text{ policy value}"
    a.xlabel = L"\text{Period }t"
    gap = round(r.gap_median, digits = 3)
    a.title = "Agent #$agent_id, gap = $gap"
    s_dp = Float64[]
    S_dp = Float64[]
    ss = Float64[]
    Ss = Float64[]
    color = :blue
    @showprogress for it in 1:104
        d = f[0+it:0+it+31]
        test_env = sl_sip(h, b, K, c, d, 0.0, LT, lostsales = false, horizon = 32, periods = 104, d_type = CVNormal{CV})
        ins = to_instance(test_env,0.99, 32)
        Scarf.DP_sS(ins, 1.)
        push!(s_dp, ins.s[1])
        push!(S_dp, ins.S[1])
        ip = 0  
        state = [-min(0,ip);d ;zeros(LT); max(0, ip)]    
        action, _ = actor(state)
        s, S = action
        err = s - ip
        count = 0
        while err > 0.1 && count < 100
            count += 1
            ip += 1
            state = [-min(0,ip);d ;zeros(LT); max(0, ip)]                
            action, _ = actor(state)
            s, S = action
            err = s - ip
        end
        if count == 100
            #print(err)
        end
        push!(ss, s)
        push!(Ss, S)
    end
    lines!(a, s_dp, color = :red, linestyle = :dash)
    lines!(a, S_dp, color = :red, linestyle = :dashdot)
    lines!(a, ss, linestyle = :dash, color = color)
    lines!(a, Ss, color = color, linestyle = :dashdot)
    save("plots/sspolicy_$agent_id.png", fig)
end

