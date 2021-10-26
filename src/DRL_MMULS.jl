module DRL_MMULS
using CUDA
using Flux
using InventoryModels
using Requires

export PPOPolicy, TD1_target, TDλ_target
export Hook, EntropyAnnealing, TestEnvironment, Kscheduler
export test_ss_policy
function __init__()
    @require Gurobi="2e9cd046-0924-5485-92f1-d5272153d98b" include("testbed/deterministic_MIP.jl")
end

include("trajectory.jl")
include("agent.jl")
include("ppo.jl")
include("hooks.jl")
include("testbed/load_environment.jl")
include("dashboard/dashboard.jl")
include("testbed/sspolicy_test.jl")

mutable struct Kscheduler
    n::Int
    Ktarget::Float64
    range::UnitRange{Int}
end

function (ks::Kscheduler)(agent, env) 
    ks.n += 1
    if ks.n in ks.range
        env.bom[1].sources[1].order_cost.K += ks.Ktarget/length(ks.range)
    end
end
end