#import ReinforcementLearningZoo: PPOTrajectory, CircularArraySARTTrajectory, Trajectory, CircularArrayTrajectory
using Statistics, Flux, ProgressMeter, LinearAlgebra
import Flux: mse, cpu, gpu

const log2π =  Float32(log(2π))
const nentropyconst = Float32((log2π + 1)/2)

function normalize!(M)
    m = mean(M)
    s = std(M)
    M .-= m
    M ./= (s + eps(s)) 
end

"""
    compute_advantages!(A, δ, λ, γ)

Compute the GAE in-place in the A matrix given an error matrix of δ where each row represents the errors of the trajectory of a single simulation.
λ and γ are the discount parameters.
"""
function compute_advantages!(A, δ, λ, γ)
    ls = typeof(δ)(getindex.(CartesianIndices(δ), 2))
    α = γ*λ
    reverse!(δ,dims = 2)
    @. δ *= α^(-ls)
    cumsum!(A, δ, dims= 2)
    @. A *= α^ls
    reverse!(A,dims = 2)
end

function TD1_target(trajectory, agent)
    trajectory.traces[:reward] .+ agent.γ * agent.critic(trajectory.traces[:next_state]) 
end

function TDλ_target(trajectory, agent)
    trajectory.traces[:advantage] .+ agent.critic(trajectory.traces[:state])
end

function normlogpdf(μ, σ, x; ϵ = eps(Float32))
   @. -(((x - μ)/(σ + ϵ))^2 + log2π) / 2.0f0 - log(σ + ϵ)
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
        push!(agent.loss_entropy, loss_entropy)#/agent.entropy_weight)
    end
    return loss_actor + loss_entropy
end

function L_value(agent, state, target_value)
    loss_critic = Flux.mse(target_value, agent.critic(state))
    Flux.Zygote.ignore() do
        push!(agent.loss_critic, sqrt(loss_critic))
    end
    return loss_critic
end

"""
    Base.run(agent::PPOPolicy, env; stop_iterations::Int, hook = (x...) -> nothing)

Main PPO training function. Run the PPO algorithm for `stop_iterations` to train `agent` at solving `env`. Hyperparameters are encoded in the `agent` object. 
Use hook to run a `hook(agent, env)` function every iteration. For example, see TestEnvironment to evaluate periodically agent. Use the `Hook` object to input multiple hooks.
"""
function Base.run(agent::PPOPolicy, env; stop_iterations::Int, hook = (x...) -> nothing)
    #clip_anneal_step = agent.clip_range/stop_iterations
    critic_anneal_step = agent.critic_optimiser.eta/stop_iterations
    actor_anneal_step = agent.actor_optimiser.eta/stop_iterations
    N = agent.n_actors
    T = agent.n_steps
    device = agent.device
    env_step_count = 0
    rewards_running_mean = 0f0
    rewards_running_M2 = 1f0
    rewards_running_std = 1f0
    #Memory preallocations
    advantages = zeros(N, T) # Agent x Periods -> flatten corresponds to trajectory
    δ = similar(advantages)
    states = zeros(state_size(env), N)
    actions = zeros(action_size(env), N)
    rewards = zeros(1, N)
    next_states = similar(states)
    μ = similar(actions)
    σ = similar(actions)
    z = similar(actions)
    action_log_probs = similar(actions)
    trajectory = PPOTrajectory(N*T, state_size(env), action_size(env), device = cpu)
    trajectory_d = PPOTrajectory(N*T, state_size(env), action_size(env), device = device)
    prog = Progress(stop_iterations)
    for it in 1:stop_iterations
        envs = [deepcopy(env) for _ in 1:N]
        for t in 1:agent.n_steps
            states .= reduce(hcat, map(state, envs))
            μ, σ = agent.actor(states |> device) |> cpu
            z .= randn(size(σ))
            actions .= μ .+ σ .* z 
            action_log_probs .= normlogpdf(μ, σ, actions)
            push!(trajectory, states, :state)
            push!(trajectory, actions, :action)
            push!(trajectory, action_log_probs, :action_log_prob)
            Threads.@threads for (env, action) in collect(zip(envs, eachcol(actions)))
                env(collect(action))
            end
            env_step_count += N
            rewards .= (reshape(map(reward, envs), 1, :))
            tmp_mean = rewards_running_mean
            rewards_running_mean = (env_step_count-N)/env_step_count*rewards_running_mean + sum(rewards)/env_step_count
            rewards_running_M2 += sum((rewards .- tmp_mean).*(rewards .- rewards_running_mean))
            rewards_running_std = max(sqrt(rewards_running_M2/(env_step_count-1)), 1f-6)
            rewards ./= rewards_running_std
            push!(trajectory, rewards, :reward)
            next_states .= reduce(hcat, map(state, envs))
            push!(trajectory, next_states, :next_state)
            δ[:,t] = reshape((agent.γ .* cpu(agent.critic(next_states |> device)) .+ rewards .- cpu(agent.critic(states |> device))), :, 1)
        end
        compute_advantages!(advantages, δ, agent.γ, agent.λ)
        push!(trajectory, reshape(advantages, 1, :), :advantage) 
        trajectory_d.traces = device(trajectory.traces)
        targets = agent.target_function(trajectory_d, agent)
        trajectory_d.traces.target_value .= targets
        dataloader = Flux.Data.DataLoader(trajectory_d, batchsize = agent.batch_size, shuffle = true, partial = false)
        for i in 1:1
            for (s, a, r, ns, alp, ad, tv) in dataloader
                psa = Flux.params(agent.actor)
                gsa = gradient(psa) do
                    L_clip(agent, s, a, ad, alp)    
                end
                Flux.update!(agent.actor_optimiser, psa, gsa)
            end
        end
        for i in 1:agent.n_epochs
            for (s, a, r, ns, alp, ad, tv) in dataloader
                psc = Flux.params(agent.critic)
                gsc = gradient(psc) do
                    L_value(agent, s, tv)    
                end
                Flux.update!(agent.critic_optimiser, psc, gsc)
            end
        end
        #agent.clip_range = max(0, agent.clip_range - clip_anneal_step)
        agent.actor_optimiser.eta = max(0, agent.actor_optimiser.eta - actor_anneal_step)
        agent.critic_optimiser.eta = max(0, agent.critic_optimiser.eta - critic_anneal_step)
        hook(agent, env)
        empty!(trajectory)
        next!(prog, showvalues = [  ("Iteration ", it), 
                                    show_value(hook.hooks[1]), 
                                    ("Actor loss ", last(agent.loss_actor)), 
                                    ("Entropy loss ", last(agent.loss_entropy)), 
                                    ("√Critic loss", last(agent.loss_critic))])
    end
end