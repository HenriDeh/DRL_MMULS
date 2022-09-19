module DRL_MMULS
using CUDA
using Flux
using Requires, Distributions, Reexport

include("InventoryModels/src/InventoryModels.jl")
@reexport using .InventoryModels

export PPOPolicy, TD1_target, TDλ_target
export Hook, TestEnvironment, Kscheduler, Normalizer
export test_ss_policy, test_agent
include("trajectory.jl")
include("agent.jl")
include("ppo.jl")
include("hooks.jl")
include("interface.jl")
include("testbed/single-item/sspolicy_test.jl")


end