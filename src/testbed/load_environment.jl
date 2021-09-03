using InventoryModels, DataFrames, CSV, Distributions

#df = CSV.read("data/instances.csv", DataFrame)


function load_environment(ins::DataFrameRow; train = false, forecasts = fill(Uniform(1,19), ins.EndProducts), policy = sSPolicy())
    EP = ins.EndProducts
    RM = ins.RawMaterials
    Rij = ins.Requirements |> Meta.parse |> eval |> first
    CV = ins.CV
    LT = ins.leadtimes |> Meta.parse |> eval |> first
    h = ins.holdings |> Meta.parse |> eval |> first
    b = ins.stockouts |> Meta.parse |> eval |> first
    K = ins.setups |> Meta.parse |> eval |> first
    μ = ins.mu |> Meta.parse |> eval |> first
    I = size(Rij)[1]
    i = I

    if train
        init_inv = Uniform.(-μ, 2μ)
    else
        init_inv = [fill(100.0,EP); zeros(I-EP)] #utiliser L*μ (+ SS) à la place ?
    end

    bom = Dict{Int,Any}()
    #Raw Materials
    while i > I - RM
        item = Item(
            Inventory(h[i], init_inv[i]),
            Supplier(K[i], 0.0, leadtime = LeadTime(LT[i], 0.0)),
            policy = policy,
            name = "RM$i"
            )
        bom[i] = item
        i -= 1
    end

    #Sub Assemblies
    while i > EP
        components = [bom[idx] => r for (idx, r) in enumerate(Rij[:,i]) if r !=0]
        item = Item(
            Inventory(h[i], init_inv[i]),
            Assembly(K[i], 0.0, components... ,leadtime = LeadTime(LT[i], 0.0)),
            policy = policy,
            name = "SA$i"
        )
        bom[i] = item
        i-=1
    end

    #End products
    while i > 0
        components = [bom[idx] => r for (idx, r) in enumerate(Rij[:,i]) if r !=0]
        ep = EndProduct(
            Market(b[i], CVNormal{CV}, ins.H, 0.0, forecasts[i,:]),
            Inventory(h[i], init_inv[i]),
            Assembly(K[i], 0.0, components... , leadtime = LeadTime(LT[i], 0.0)),
            policy = policy,
            name = "EP$i"
        )
        bom[i] = ep
        i-=1 
    end

    #Ressource Constraints
    Cs = ins.capacities |> Meta.parse |> eval |> first
    α = ins.ressources |> Meta.parse |> eval |> first

    constraints = Dict{Int, Any}()
    for (k, C) in enumerate(Cs)
        components = [only(bom[idx].sources) => r for (idx, r) in enumerate(α[:,k]) if r !=0]
        co = RessourceConstraint(C, components, name = "constraint $k")
        constraints[k] = co
    end

    #Instanciate
    T = train ? 104 : 20
    return InventorySystem(T, [el for el in values(bom)], [el for el in values(constraints)]), bom
end