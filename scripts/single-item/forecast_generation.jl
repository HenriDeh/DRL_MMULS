#script used to generate forecasts.csv
using Distributions, CSV, DataFrames
CSV.write("data/single-item/forecasts.csv", DataFrame(ID = [],trend = [], forecast = []))
T = 104
H = 52
μs = (5,10,15)
ID = 0
"""Constant trends"""

for μ in μs
    global ID += 1
    fc = [μ for i in 1:T]
    CSV.write("data/single-item/forecasts.csv", DataFrame(ID = ID, trend = "Constant", forecast = [fc]), append = true)
end

"""Seasonal trends"""

fs = [0.5, 1, 2]

for f in fs
    for μ in μs
        global ID += 1
        fc = [(1 + 0.5*sin(2f*t*π/T))*μ for t in 1:T]
        CSV.write("data/single-item/forecasts.csv", DataFrame(ID = ID, trend = "Seasonnal$f", forecast = [fc]), append = true)
    end
end


"""Growth trends"""

for μ in μs
    global ID += 1
    fc = [(0.8+0.4*t/T)*μ for t in 1:T]
    CSV.write("data/single-item/forecasts.csv", DataFrame(ID = ID, trend = "Growth", forecast = [fc]), append = true)
end


"""Decline trends"""

for μ in μs
    global ID += 1
    fc = [(1.2-0.4*t/T)*μ for t in 1:T]
    CSV.write("data/single-item/forecasts.csv", DataFrame(ID = ID, trend = "Decline", forecast = [fc]), append = true)
end
