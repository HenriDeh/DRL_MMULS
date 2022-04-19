#script used to generate forecasts.csv
using Distributions, CSV, DataFrames
CSV.write("data/single-item/forecasts.csv", DataFrame(ID = [],trend = [], forecast = []))
T = 104
H = 52
μ = 10
ID = 0
"""Constant trend"""

global ID += 1
fc = [μ for i in 1:T]
CSV.write("data/single-item/forecasts.csv", DataFrame(ID = ID, trend = "Constant", forecast = [fc]), append = true)


"""Seasonal trends"""

fs = [0.5, 1, 2]

for f in fs
    global ID += 1
    fc = [(1 + 0.5*sin(2f*t*π/T))*μ for t in 1:T]
    CSV.write("data/single-item/forecasts.csv", DataFrame(ID = ID, trend = "Seasonnal$f", forecast = [fc]), append = true)
end


"""Growth trends"""

global ID += 1
fc = [(0.8+0.4*t/T)*μ for t in 1:T]
CSV.write("data/single-item/forecasts.csv", DataFrame(ID = ID, trend = "Growth", forecast = [fc]), append = true)



"""Decline trends"""

global ID += 1
fc = [(1.2-0.4*t/T)*μ for t in 1:T]
CSV.write("data/single-item/forecasts.csv", DataFrame(ID = ID, trend = "Decline", forecast = [fc]), append = true)
