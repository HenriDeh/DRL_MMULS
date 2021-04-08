module DRL_MMULS
using CUDA, Flux
using InventoryModels

export PPOPolicy, TD1_target, TDλ_target
export Hook, EntropyAnnealing, TestEnvironment

include("trajectory.jl")
include("ppo.jl")
include("agent.jl")
include("hooks.jl")

end