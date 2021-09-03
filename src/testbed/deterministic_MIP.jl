using CSV, DataFrames, JuMP, Distributions, OffsetArrays
using .Gurobi

#df = CSV.read("data/instances.csv", DataFrame)

function load_deterministic_MIP(ins::DataFrameRow; forecasts, init_inv = :auto, prevQ = :auto, backorders = zeros(ins.EndProducts))
    EP = ins.EndProducts #number of end products
    RM = ins.RawMaterials #number of raw materials
    R = ins.Requirements |> Meta.parse |> eval |> first #matrix of BOM
    CV = ins.CV #CV of demand
    LT = ins.leadtimes |> Meta.parse |> eval |> first #LT of each item
    h = ins.holdings |> Meta.parse |> eval |> first #holding cost of each item
    b = ins.stockouts |> Meta.parse |> eval |> first #stockout cost of each item
    K = ins.setups |> Meta.parse |> eval |> first #setup cost of each item
    μ = ins.mu |> Meta.parse |> eval |> first #average per period demand of each item
    I = size(R)[1] #number of items
    α = ins.ressources |> Meta.parse |> eval |> first #matrix of capactity constraints' factors 
    C = ins.capacities |> Meta.parse |> eval |> first #capactity of each ressource
    Nres = length(C) #number of capactity constraints
    H = size(forecasts, 2) #forecast horizon
    @assert size(forecasts,1) == (EP) "forecasts must have $EP rows" #
    T = 9 #startup time of the system: longest possible time needed to make an endproduct from scratch 
    #H += T 
    d = forecasts #demand refactor 
    
    M = zeros(I, H) #bigM of each product at each period
    M[1:EP,:] .= reverse(cumsum(reverse(d, dims=2), dims=2), dims=2) #end products use cumulative demand
    for i in (EP + 1):I
        M[i,:] = [sum(R[i,j] * M[j,t] for j in 1:I) for t in 1:H] #other items use internal cumulative demand
    end
    
    s0 = init_inv == :auto ? init_inv = fill(0,I) : init_inv #initial inventories are 0 if not provided

    x0 = prevQ == :auto ? OffsetArray(fill(0.0, I, maximum(LT)), 1:I, 1-maximum(LT):0) : OffsetArray(prevQ, 1:I, 1-maximum(LT):0)
    reverse!(x0, dims = 2)
    #on-order quantities. 0 if not provided.

    # compute SS
    σ = CV .* forecasts #demand standard deviation
    ratio = (b ./ (b .+ h))[1:EP] #critical ratio of end products
    SS = quantile.(Normal(), ratio) .* σ #compute SS based on critical ratio
    #SS = [zeros(EP, T) SS] #SS is 0 during startup time.

    model = Model()
    @variable(model, s[i=1:I, t=0:H] >= 0) #stock
    @variable(model, m[i=1:I, t=0:H] >= 0) #backorders
    @variable(model, x[i=1:I, t=1-LT[i]:H] >= 0) #quantities
    @variable(model, y[i=1:I, t=1:H], Bin) #setups
    @variable(model, p[i=1:I, t=1:H] >=0) # missing SS
    
    @objective(model, Min, sum(K[i] * y[i,t] + h[i] * s[i,t] + 1.5h[i]*p[i,t] for i in 1:I, t in 1:H) + sum(m[i,t]*b[i] for i in 1:EP, t in 1:H))

    @constraint(model, #on order quantities
                prevQ[i=1:I, t=1-LT[i]:0], 
                x[i,t] == x0[i,t]
    )    
    @constraint(model, #initial backorders
                prevB[i=1:EP],
                m[i,0] == backorders[i]
    )
    @constraint(model, #safety stock penalty
                SS[i=1:EP, t=T:H], 
                s[i,t] + p[i,t] >= SS[i,t]
    )
    @constraint(model, #initial inventories
                init_inv[i=1:I], 
                s[i,0] == s0[i]
    )
    @constraint(model, #balance equalities EndProducts
                balance_ep[i=1:EP, t=1:H], 
                s[i,t - 1] + x[i, t-LT[i]] == d[i, t] + s[i, t] - m[i, t]
    )
    @constraint(model, #balance equalities other items
                balance_comp[i=EP+1:I, t=1:H], 
                s[i,t - 1] + x[i, t-LT[i]] == sum(R[i,j] * x[j,t] for j in 1:I) + s[i, t]
    )
    @constraint(model, #BigM
                bigM[i=1:I, t=1:H], 
                x[i,t] <= M[i,t] * y[i,t]
    )
    @constraint(model, #capacities
                capacities[k=1:Nres, t=1:H], 
                sum(α[i,k] * x[i,t] for i in 1:I) <= C[k]
    )
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