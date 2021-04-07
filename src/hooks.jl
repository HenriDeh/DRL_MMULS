struct Hook{H}
    hooks::H
end

Hook(hooks...) = Hook(hooks)

(h::Hook)(agent, env) = h.hooks(agent,env)
(h::Hook{<:Tuple})(agent, env) = for hook in h.hooks hook(agent,env) end

struct EntropyAnnealing
    step::Float32
    target::Float32
end

EntropyAnnealing(lr::LinRange) = EntropyAnnealing(lr[1] - lr[2], last(lr))

function (ea::EntropyAnnealing)(agent::PPOPolicy, env) 
    if agent.entropy_weight > ea.target
        agent.entropy_weight -= ea.step
    end
end

mutable struct TestEnvironment{E}
    env::E
    n_sim::Int
    log::Vector{Float64}
    every::Int
    count::Int
end

TestEnvironment(env, n_sim, every = 1) = TestEnvironment(env, n_sim, [-Inf], every, 0)

function (te::TestEnvironment)(agent::PPOPolicy, envi; stochastic = false)
    te.count += 1
    totreward = 0.
    te.count % te.every == 0 || return totreward
    reset!(envi)
    envs = [deepcopy(te.env) for _ in 1:te.n_sim]
    while all(map(!, is_terminated.(envs)))
        s = reduce(hcat, state.(envs))
        a, σ = agent.actor(s)
        if stochastic
            a .+= σ .* randn(size(σ)...)
        end
        for (env, action) in zip(envs, eachcol(a))
            env(collect(action))
        end
        totreward += sum(reward.(envs))
    end
    reset!(envi)
    push!(te.log, totreward/te.n_sim)
    return totreward/te.n_sim
end
