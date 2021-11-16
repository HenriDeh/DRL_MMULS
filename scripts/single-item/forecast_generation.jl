#script used to generate forecasts.csv
using Distributions, CSV, DataFrames
CSV.write("data/single-item/forecasts.csv", DataFrame(ID = [],trend = [], deviation = [], forecast = []))
T = 52
H = 52
μ = 10
n = 50
ϱs = [2, 4]
ID = 0
"""Constant trends"""

for ϱ in ϱs
    δ = Uniform(-ϱ, ϱ)
    for i in 1:n
        global ID += 1
        fc = [μ + rand(δ) for i in 1:T]
        CSV.write("data/single-item/forecasts.csv", DataFrame(ID = ID, trend = "Constant", varrho = ϱ, forecast = [fc]), append = true)
    end
end

"""Seasonal trends"""

fs = [1, 2]
for ϱ in ϱs
    δ = Uniform(-ϱ, ϱ)
    for f in fs
        for i in 1:n
            global ID += 1
            fc = [(1 + 0.5*sin(2f*t*π/H))*μ + rand(δ) for t in 1:T]
            CSV.write("data/single-item/forecasts.csv", DataFrame(ID = ID, trend = "Seasonnal$f", varrho = ϱ, forecast = [fc]), append = true)
        end
    end
end

"""Growth trends"""

for ϱ in ϱs
    δ = Uniform(-ϱ, ϱ)
    for i in 1:n
        global ID += 1
        fc = [(0.8+0.4*t/T)*μ + rand(δ) for t in 1:T]
        CSV.write("data/single-item/forecasts.csv", DataFrame(ID = ID, trend = "Growth", varrho = ϱ, forecast = [fc]), append = true)
    end
end

"""Decline trends"""

for ϱ in ϱs
    δ = Uniform(-ϱ, ϱ)
    for i in 1:n
        global ID += 1
        fc = [(1.2-0.4*t/T)*μ + rand(δ) for t in 1:T]
        CSV.write("data/single-item/forecasts.csv", DataFrame(ID = ID, trend = "Decline", varrho = ϱ, forecast = [fc]), append = true)
    end
end
