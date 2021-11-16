module DRL_MMULS
using CUDA
using Flux
using InventoryModels
using Requires, Distributions

export PPOPolicy, TD1_target, TDλ_target
export Hook, TestEnvironment, Kscheduler, Normalizer
export test_ss_policy, test_agent
include("trajectory.jl")
include("agent.jl")
include("ppo.jl")
include("hooks.jl")
include("testbed/single-item/sspolicy_test.jl")


end