
mutable struct PPOPolicy{A, AO, C, CO,T,D}
    actor::A
    actor_optimiser::AO
    critic::C
    critic_optimiser::CO
    γ::Float32
    λ::Float32
    clip_range::Float32
    entropy_weight::Float32
    n_actors::Int
    n_steps::Int
    n_epochs::Int
    batch_size::Int
    target_function::T
    loss_actor::Vector{Float32}
    loss_critic::Vector{Float32}
    loss_entropy::Vector{Float32}
    device::D
end

struct Split{T}
    paths::T
end
  
Split(paths...) = Split(paths)
  
Flux.@functor Split
  
(m::Split)(x::AbstractArray) = map(f -> f(x), m.paths)

"""
    PPOPolicy(env; <keyword arguments>)

Instantiate a PPO agent. Although not specific, it follows a custom API and is probably not compatible with other environments than those of InventoryModels.
PPOPolicy logs the loss of each training epoch in the `loss_actor`, `loss_critic`, and `loss_entropy` fields.

#Arguments
`env`: Environment on which the agent will be trained, or another one but with similar spaces. Used only to get the state and action dimensions.
`actor_optimiser`: Optimiser from the Flux library used to update the actor network.
`critic_optimiser`: Optimiser from the Flux library used to update the critic network.
`γ`: Discount factor, should be ∈ ]0,1[.
`λ`: Weight of the Generalized Advantage Estimation. 
`clip_range`:PPO clip range parameter. 
`entropy_weight`: Weight factor for the entropy bonus in the actor loss function.
`n_steps::Int = env.T`: Number of steps performed by the agent on each environment at each iteration.
`n_actors::Int`: Number of parallel environments during training.
`n_epochs::Int`: Number of training epochs performed on the critic at each iteration. 
`batch_size::Int`: Size of minibatches sampled for gradient descent.
`target_function = TD1_target`. Value target function. TD1_target is the TD(1) value function. Alternativelly, TDλ_target will the GAE value estimation. 
`n_hidden::Int`: Number of neurons per hidden layer.
`device = cpu`: Device on which the agent weights are stored. Accepts the two following functions from Flux.jl: `cpu` or `gpu`. Use `gpu` if you have an Nvidia compatible graphics card.
"""
function PPOPolicy(env; actor_optimiser, critic_optimiser, γ, λ, clip_range, entropy_weight, n_steps = env.T,
    n_actors::Int, n_epochs::Int, batch_size::Int, target_function = TD1_target, n_hidden::Int, device = cpu)
    actor = Chain(
        Dense(state_size(env), n_hidden, gelu, init=Flux.orthogonal),
        Dense(n_hidden, n_hidden, gelu, init=Flux.orthogonal),
        Dense(n_hidden, n_hidden, gelu, init=Flux.orthogonal),
        Split(
            Dense(n_hidden, action_size(env), identity, bias = fill(0f0, action_size(env)),  init= Flux.orthogonal),
            Dense(n_hidden, action_size(env), softplus, bias = fill(0f0, action_size(env)), init=Flux.orthogonal)
            )
    ) |> device

    critic = Chain(
        Dense(state_size(env), n_hidden, gelu, init=Flux.orthogonal),
        Dense(n_hidden, n_hidden, gelu, init=Flux.orthogonal),
        Dense(n_hidden, n_hidden, gelu, init=Flux.orthogonal),
        Dense(n_hidden, 1, init=Flux.orthogonal)
    ) |> device

    PPOPolicy(
        actor,
        actor_optimiser,
        critic,
        critic_optimiser,
        γ,
        λ,
        clip_range,
        entropy_weight,
        n_actors,
        n_steps,
        n_epochs,
        batch_size,
        target_function,
        Float32[],
        Float32[],
        Float32[],
        device
    )
end

function Flux.cpu(p::PPOPolicy)
    PPOPolicy(
        p.actor |> cpu,
        p.actor_optimiser,
        p.critic |> cpu,
        p.critic_optimiser,
        p.γ,
        p.λ,
        p.clip_range,
        p.entropy_weight,
        p.n_actors,
        p.n_steps,
        p.n_epochs,
        p.batch_size,
        p.target_function,
        p.loss_actor,
        p.loss_critic,
        p.loss_entropy,
        cpu
    )
end

function Flux.gpu(p::PPOPolicy)
    PPOPolicy(
        p.actor |> gpu,
        p.actor_optimiser,
        p.critic |> gpu,
        p.critic_optimiser,
        p.γ,
        p.λ,
        p.clip_range,
        p.entropy_weight,
        p.n_actors,
        p.n_steps,
        p.n_epochs,
        p.batch_size,
        p.target_function,
        p.loss_actor,
        p.loss_critic,
        p.loss_entropy,
        gpu
    )
end

 