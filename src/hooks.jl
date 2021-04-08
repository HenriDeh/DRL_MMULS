struct Hook{H<:Tuple}
    hooks::H
end

Hook(hooks...) = Hook{typeof(hooks)}(hooks)
Hook(hooks::Tuple) = Hook{typeof(hooks)}(hooks)

(h::Hook)(agent, env) = for hook in h.hooks hook(agent,env) end

show_value(h::Hook{<:Tuple}) = [show_value(hook) for hook in h.hooks] 

struct EntropyAnnealing
    step::Float32
    target::Float32
end

EntropyAnnealing(lr::LinRange) = EntropyAnnealing(lr[1] - lr[2], last(lr))

show_value(ea::EntropyAnnealing) = ("Entropy annealing to ", ea.target)

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
    stochastic::Bool
    batchsize::Int
end

TestEnvironment(env, n_sim, every = 1; stochastic = false, batchsize = 1000) = TestEnvironment(env, n_sim, [-Inf], every, 0, stochastic, batchsize)

function (te::TestEnvironment)(agent::PPOPolicy, envi)
    te.count += 1
    totreward = 0.
    te.count % te.every == 0 || return totreward
    reset!(te.env)
    envs_part = Iterators.partition([deepcopy(te.env) for _ in 1:te.n_sim], te.batchsize)
    for envs in envs_part
        while all(map(!, is_terminated.(envs)))
            s = reduce(hcat, state.(envs)) |> agent.device
            a, σ = agent.actor(s)
            if te.stochastic
                a .+= σ .* (randn(size(σ)) |> agent.device)
            end
            for (env, action) in zip(envs, eachcol(cpu(a)))
                env(collect(action))
            end
            totreward += sum(reward.(envs))
        end
    end
    push!(te.log, totreward/te.n_sim)
    return totreward/te.n_sim
end

Base.getindex(te::TestEnvironment, n) = te.log[n]
show_value(te::TestEnvironment) = ("Test environment return ", last(te.log))