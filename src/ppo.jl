#import ReinforcementLearningZoo: PPOTrajectory, CircularArraySARTTrajectory, Trajectory, CircularArrayTrajectory
using Statistics, Flux, ProgressMeter
import ReinforcementLearningCore.Agent
import CircularArrayBuffers: CircularArrayBuffer
import Flux: mse

const log2π =  Float32(log(2π))
const nentropyconst = Float32((log2π + 1)/2)

mutable struct PPOPolicy{A, AO, C, CO,T}
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
end

struct PPOTrajectory
    state::CircularArrayBuffer{Float32,2}
    action::CircularArrayBuffer{Float32,2}
    reward::CircularArrayBuffer{Float32,2}
    next_state::CircularArrayBuffer{Float32,2}
    action_log_prob::CircularArrayBuffer{Float32,2}
    advantage::CircularArrayBuffer{Float32,2}
    target_value::CircularArrayBuffer{Float32,2}
end

function PPOTrajectory(N, state_size, action_size)
    PPOTrajectory(
        CircularArrayBuffer{Float32}(state_size, N),
        CircularArrayBuffer{Float32}(action_size, N),
        CircularArrayBuffer{Float32}(1, N),
        CircularArrayBuffer{Float32}(state_size, N),
        CircularArrayBuffer{Float32}(action_size, N),
        CircularArrayBuffer{Float32}(1, N),
        CircularArrayBuffer{Float32}(1, N)
    )
end

function Base.rand(t::PPOTrajectory, n::Int)
    idx = rand(1:size(t.action)[2], n)
    t.state[:,idx], t.action[:,idx], t.action_log_prob[:,idx], t.advantage[:,idx], t.target_value[:,idx]
end

function Base.empty!(t::PPOTrajectory)
    empty!(t.state)
    empty!(t.action)
    empty!(t.reward)
    empty!(t.next_state)
    empty!(t.action_log_prob)
    empty!(t.advantage)
    empty!(t.target_value)
end

function compute_advantages!(A, δ, λ, γ)
    T = size(δ)[2]
    for t in 1:T
        for l in 0:T-t
            @inbounds A[:, t] += (λ*γ)^l .* δ[:,t+l]
        end
    end
end

function TD1_target(trajectory, agent)
    push!(trajectory.target_value, (trajectory.reward.buffer .+ agent.critic(trajectory.next_state.buffer) .- agent.critic(trajectory.state.buffer))...)
end

function TDλ_target(trajectory, agent)
    push!(trajectory.target_value, (trajectory.advantage.buffer .+ agent.critic(trajectory.state.buffer))...)
end

function normlogpdf(μ, σ, x; ϵ = eps(Float32))
    z = (x .- μ) ./ (σ .+ ϵ)
    -(z .^ 2 .+ log2π) / 2.0f0 .- log.(σ .+ ϵ)
end

normentropy(σ, ϵ = eps(Float32)) = nentropyconst + log(σ + ϵ)

function L_clip(agent, state, action, advantage, log_prob_old)
    μ, σ = agent.actor(state)
    log_prob_new = normlogpdf.(μ, σ, action)
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
    envs = [deepcopy(env) for _ in 1:agent.n_actors]
    advantages = Float32.(zeros(agent.n_actors, agent.n_steps)) # Agent x Periods -> flatten corresponds to trajectory
    δ = similar(advantages)
    trajectory = PPOTrajectory(N*T, state_size(env), action_size(env))
    prog = Progress(stop_iterations)
    showtest = () -> "No test environment"
    if hook.hooks isa Tuple
        for h in hook.hooks
            if h isa TestEnvironment
                showtest = () -> string("Test environment return: ", last(h.log))
            end
        end
    elseif hook.hooks isa TestEnvironment
        showtest = () -> string("Test environment return = ", last(hook.hooks.log))
    end
    for it in 1:stop_iterations
        advantages .= 0f0
        reset!.(envs)
        for t in 1:agent.n_steps
            states = reduce(hcat, state.(envs))
            μ, σ = agent.actor(states)
            z = randn(size(σ))
            actions = μ .+ σ .* z
            action_log_probs = normlogpdf.(μ, σ, actions)
            push!(trajectory.state, eachcol(states)...)
            push!(trajectory.action, eachcol(actions)...)
            push!(trajectory.action_log_prob, eachcol(action_log_probs)...)
            for (env, action) in zip(envs, eachcol(actions))
                env(collect(action))
            end
            rewards = reshape(reward.(envs), 1, :)
            push!(trajectory.reward, eachcol(rewards)...)
            next_states = reduce(hcat, state.(envs))
            push!(trajectory.next_state, eachcol(next_states)...)
            δ[:,t] = reshape(agent.γ .* agent.critic(next_states) .+ rewards .- agent.critic(states), :, 1)
        end
        compute_advantages!(advantages, δ, agent.γ, agent.λ)
        push!(trajectory.advantage, reshape(advantages, 1, :)...)
        agent.target_function(trajectory, agent)
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
        next!(prog, showvalues = [("$it ", showtest()), ("Actor loss ", last(agent.loss_actor)), ("Entropy loss ", last(agent.loss_entropy)), ("Critic loss", last(agent.loss_critic))])
    end
end