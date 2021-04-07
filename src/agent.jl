struct Split{T}
    paths::T
end
  
Split(paths...) = Split(paths)
  
Flux.@functor Split
  
(m::Split)(x::AbstractArray) = map(f -> f(x), m.paths)

function PPOPolicy(
    env; 
    actor_optimiser, 
    critic_optimiser, 
    γ, 
    λ, 
    clip_range, 
    entropy_weight, 
    n_actors::Int, 
    n_epochs::Int, 
    batch_size::Int, 
    target_function = TD1_target,
    n_hidden::Int)
    actor = Chain(
        Dense(state_size(env), n_hidden, relu),
        Dense(n_hidden, n_hidden, relu),
        Dense(n_hidden, n_hidden, relu),
        Split(
            Dense(n_hidden, action_size(env)),
            Dense(n_hidden, action_size(env), softplus)
            )
    )

    critic = Chain(
        Dense(state_size(env), n_hidden, relu),
        Dense(n_hidden, n_hidden, relu),
        Dense(n_hidden, n_hidden, relu),
        Dense(n_hidden, 1)
    )

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
        env.T,
        n_epochs,
        batch_size,
        target_function,
        Float32[],
        Float32[],
        Float32[]
    )
end