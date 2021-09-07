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

function PPOPolicy(env; actor_optimiser, critic_optimiser, γ, λ, clip_range, entropy_weight, n_steps = env.T,
    n_actors::Int, n_epochs::Int, batch_size::Int, target_function = TD1_target, n_hidden::Int, device = cpu)
    actor = Chain(
        Dense(state_size(env), n_hidden, relu),
        Dense(n_hidden, n_hidden, relu),
        Dense(n_hidden, n_hidden, relu),
        Split(
            Dense(n_hidden, action_size(env)),
            Dense(n_hidden, action_size(env), softplus)
            )
    ) |> device

    critic = Chain(
        Dense(state_size(env), n_hidden, relu),
        Dense(n_hidden, n_hidden, relu),
        Dense(n_hidden, n_hidden, relu),
        Dense(n_hidden, 1)
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