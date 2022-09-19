#script used to generate forecasts.csv
using Distributions, CSV, DataFrames
CSV.write("data/single-item/forecasts.csv", DataFrame(ID = [],trend = [], forecast = []))
T = 52
H = 136
μ = 10
ID = 0
"""Constant trend"""

shifts = [0.5,1.,1.5]
for s in shifts
    global ID += 1
    fc = [s*μ for i in 1:H]
    CSV.write("data/single-item/forecasts.csv", DataFrame(ID = ID, trend = "Constant_$s", forecast = [fc]), append = true)
end

"""Seasonal trends"""

fs = [1, 2, 4]
for f in fs
    global ID += 1
    fc = [(1 + 0.5*sin(f*t*π/T))*μ for t in 1:H]
    CSV.write("data/single-item/forecasts.csv", DataFrame(ID = ID, trend = "Seasonnal_$(f)", forecast = [fc]), append = true)
end


"""Growth trends"""

s = 0.5
global ID += 1
fc = LinRange(s*μ, (1 + s)*μ, H) |> collect
CSV.write("data/single-item/forecasts.csv", DataFrame(ID = ID, trend = "Growth_$s", forecast = [fc]), append = true)




"""Decline trends"""

global ID += 1
fc = LinRange((2-s)*μ, s*μ, H) |> collect
CSV.write("data/single-item/forecasts.csv", DataFrame(ID = ID, trend = "Decline_$s", forecast = [fc]), append = true)
