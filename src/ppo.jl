#import ReinforcementLearningZoo: PPOTrajectory, CircularArraySARTTrajectory, Trajectory, CircularArrayTrajectory
using Statistics, Flux, ProgressMeter
import Flux: mse

const log2π =  Float32(log(2π))
const nentropyconst = Float32((log2π + 1)/2)

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

function compute_advantages!(A, δ, λ, γ)
    T = size(δ)[2]
    ls = typeof(δ)(getindex.(CartesianIndices(δ), 2))
    A .= δ
    reverse!((λ*γ) .^(ls) .* cumsum((λ*γ) .^(.- ls) .* reverse!(δ, dims = 2), dims = 2), dims= 2)
end

function TD1_target(trajectory, agent)
    trajectory.traces[:reward] .+ agent.critic(trajectory.traces[:next_state]) .- agent.critic(trajectory.traces[:state])
end

function TDλ_target(trajectory, agent)
    trajectory.traces[:advantage] .+ agent.critic(trajectory.traces[:state])
end

function normlogpdf(μ, σ, x; ϵ = eps(Float32))
    -(((x .- μ) ./ (σ .+ ϵ)) .^ 2 .+ log2π) / 2.0f0 .- log.(σ .+ ϵ)
end

normentropy(σ, ϵ = eps(Float32)) = nentropyconst + log(σ + ϵ)

function L_clip(agent, state, action, advantage, log_prob_old)
    μ, σ = agent.actor(state)
    log_prob_new = normlogpdf(μ, σ, action)
    ratio = exp.(log_prob_new .- log_prob_old)
    loss_actor = -mean(min.(ratio .* advantage, clamp.(ratio, 1f0-agent.clip_range, 1f0+agent.clip_range) .* advantage))
    loss_entropy = -mean(normentropy.(σ))*agent.entropy_weight
    Flux.Zygote.ignore() do
        push!(agent.loss_actor, loss_actor)
        push!(agent.loss_entropy, loss_entropy)
    end
    return loss_actor + loss_entropy
end

function L_value(agent, state, target_value)
    loss_critic = Flux.mse(target_value, agent.critic(state))
    Flux.Zygote.ignore() do
        push!(agent.loss_critic, loss_critic)
    end
    return loss_critic
end

function Base.run(agent::PPOPolicy, env; stop_iterations::Int, hook = (x...) -> nothing)
    N = agent.n_actors
    T = agent.n_steps
    device = agent.device
    envs = [deepcopy(env) for _ in 1:N]
    #Memory preallocations
    advantages = zeros(N, T) |> device # Agent x Periods -> flatten corresponds to trajectory
    δ = similar(advantages)
    states = zeros(state_size(env), N) |> device
    actions = zeros(action_size(env), N) |> device
    rewards = zeros(1, N) |> device
    next_states = similar(states)
    μ = similar(actions)
    σ = similar(actions)
    z = similar(actions)
    action_log_probs = similar(actions)
    trajectory = PPOTrajectory(N*T, state_size(env), action_size(env), device = device)
    prog = Progress(stop_iterations)
    for it in 1:stop_iterations
        reset!.(envs)
        for t in 1:agent.n_steps
            states .= reduce(hcat, state.(envs)) |> device
            μ, σ = agent.actor(states)
            z .= randn(size(σ)) |> device
            actions .= μ .+ σ .* z 
            action_log_probs .= normlogpdf(μ, σ, actions)
            push!(trajectory, states, :state)
            push!(trajectory, actions, :action)
            push!(trajectory, action_log_probs, :action_log_prob)
            for (env, action) in zip(envs, eachcol(cpu(actions)))
                env(collect(action))
            end
            rewards .= reshape(reward.(envs), 1, :) |> device
            push!(trajectory, rewards, :reward)
            next_states .= reduce(hcat, state.(envs)) |> device
            push!(trajectory, next_states, :next_state)
            δ[:,t] = reshape((agent.γ .* agent.critic(next_states) .+ rewards .- agent.critic(states)), :, 1)
        end
        compute_advantages!(advantages, δ, agent.γ, agent.λ)
        push!(trajectory, reshape(advantages, 1, :), :advantage) 
        push!(trajectory, agent.target_function(trajectory, agent), :target_value)
        for i in 1:agent.n_epochs
            s, a, alp, ad, tv = rand(trajectory, agent.batch_size)
            psa = Flux.params(agent.actor)
            gsa = gradient(psa) do
                L_clip(agent, s, a, ad, alp)    
            end
            Flux.update!(agent.actor_optimiser, psa, gsa)
            psc = Flux.params(agent.critic)
            gsc = gradient(psc) do
                L_value(agent, s, tv)    
            end
            Flux.update!(agent.critic_optimiser, psc, gsc)
        end
        hook(agent, env)
        empty!(trajectory)
        next!(prog, showvalues = [("Iteration ", it), show_value(hook)..., ("Actor loss ", last(agent.loss_actor)), ("Entropy loss ", last(agent.loss_entropy)), ("Critic loss", last(agent.loss_critic))])
    end
end