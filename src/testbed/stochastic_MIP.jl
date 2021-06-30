using CSV, DataFrames, JuMP, Distributions, Gurobi, OffsetArrays

df = CSV.read("data/instances.csv", DataFrame)

function load_stochastic_MIP(ins::DataFrameRow; forecasts, init_inv = :auto, prevQ = :auto, scenarios = 1000)
    EP = ins.EndProducts
    RM = ins.RawMaterials
    R = ins.Requirements |> Meta.parse |> eval |> first
    CV = ins.CV
    LT = ins.leadtimes |> Meta.parse |> eval |> first
    h = ins.holdings |> Meta.parse |> eval |> first
    b = ins.stockouts |> Meta.parse |> eval |> first
    K = ins.setups |> Meta.parse |> eval |> first
    μ = ins.mu |> Meta.parse |> eval |> first
    I = size(R)[1]
    α = ins.ressources |> Meta.parse |> eval |> first
    C = ins.capacities |> Meta.parse |> eval |> first
    Nres = length(C)
    H = size(forecasts, 2)
    @assert size(forecasts,1) == (EP) "forecasts must have $EP rows"
    T = 9
    H += T
    d = [zeros(EP, T) forecasts]
    
    M = zeros(I, H)
    M[1:EP,:] .= reverse(cumsum(reverse(d, dims=2), dims=2), dims=2)
    for i in EP + 1:I
        M[i,:] = [sum(R[i,j] * M[j,t] for j in 1:I) for t in 1:H]
    end
    
    s0 = init_inv == :auto ? init_inv = fill(0,I) : init_inv

    x0 = prevQ == :auto ? OffsetArray(fill(0.0, I, maximum(LT)), 1:I, 1-maximum(LT):0) : OffsetArray(prevQ, 1:I, 1-maximum(LT):0)
    
    σ = CV .* forecasts
    d = reduce((x,y) -> cat(x,y,dims = 3), [rand.(Normal.(forecasts, σ)) for i in 1:scenarios])
    
    #= compute SS
    ratio = (b ./ (b .+ h))[1:EP]
    SS = quantile.(Normal(), ratio) .* σ
    SS = [zeros(EP, T) SS]
    =#
    model = Model()
    @variable(model, s[i=1:I, t=0:H] >= 0)
    @variable(model, x[i=1:I, t=1-LT[i]:H] >= 0)
    @variable(model, y[i=1:I, t=1:H], Bin)
    
    @constraint(model, prevQ[i=1:I, t=1-LT[i]:0], x[i,t] == x0[i,t])
    
    @constraint(model, 
                SS[i=1:EP, t=T:H], 
                s[i, t] >= SS[i,t]
                )
    @constraint(model, 
                init_inv[i=1:I], 
                s[i,0] == s0[i]
    )
    @constraint(model, 
                balance_ep[i=1:EP, t=1:H], 
                s[i,t - 1] + x[i, t-LT[i]] == d[i, t] + s[i, t]
    )
    @constraint(model, 
                balance_comp[i=EP+1:I, t=1:H], 
                s[i,t - 1] + x[i, t-LT[i]] == sum(R[i,j] * x[j,t] for j in 1:I) + s[i, t]
    )
    @constraint(model, 
                bigM[i=1:I, t=1:H], 
                x[i,t] <= M[i,t] * y[i,t]
    )
    @constraint(model, 
                capacities[k=1:Nres, t=1:H], 
                sum(α[i,k] * x[i,t] for i in 1:I) <= C[k]
    )
    @objective(model, Min, sum(K[i] * y[i,t] + h[i] * s[i,t] for i in 1:I, t in 1:H))
    return model
end

#=
model = load_deterministic_MIP(df[1,:], forecasts=fill(10, 4, 32))
set_optimizer(model, optimizer_with_attributes(Gurobi.Optimizer, "TimeLimit" => 60))

optimize!(model)

solution_summary(model)
relative_gap(model)
objective_value(model)
value.(model[:s])[:,1] |> collect
value.(model[:y])
xs = [value(model[:x][i,t]) for i = 1:10, t = 1:32]
=#