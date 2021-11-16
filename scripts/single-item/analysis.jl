using CSV, DataFrames, Statistics, GLMakie

df_ppo = CSV.read("data/single-item/ppo_testbed.csv", DataFrame)
df_opt = CSV.read("data/single-item/scarf_testbed.csv", DataFrame)
forecast_df = CSV.read("data/single-item/forecasts.csv", DataFrame)
df_opt = innerjoin(df_opt, forecast_df, on =(:forecast_id=> :ID))[:, Not(:forecast)]
df = innerjoin(df_ppo, df_opt, on = [:leadtime, :shortage, :setup, :lostsales, :CV, :forecast_id])
df.gap = df.avg_cost ./ df.opt_cost .-1
gdf  = groupby(df, [:leadtime, :shortage, :setup, :lostsales, :CV, :policy, :agent_id])
df2 = combine(gdf, :gap .=> [mean, maximum, minimum])
show(df2, allrows = true)
gdf = groupby(df2, [:leadtime, :shortage, :setup, :lostsales, :CV, :policy])

begin
    f = Figure()
    a = Axis(f[1,1])
    g = groupby(df, )
end



begin
    f = Figure()
    a=Axis(f[1,1])
    g = groupby(df, [:trend, :deviation])
    for data in g
        density!(data.gap, label = "$(first(data.trend)) $(first(data.deviation))")
    end
    axislegend(a)
    f
end