module DRL_MMULS
using CUDA
using Flux
using InventoryModels
using Requires

export PPOPolicy, TD1_target, TDλ_target
export Hook, TestEnvironment, Kscheduler
export test_ss_policy
function __init__()
    @require Gurobi="2e9cd046-0924-5485-92f1-d5272153d98b" include("testbed/deterministic_MIP.jl")
end

include("trajectory.jl")
include("agent.jl")
include("ppo.jl")
include("hooks.jl")
include("testbed/single-item/sspolicy_test.jl")


end