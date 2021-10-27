show_value(::Any) = [("No ", "hook")]

struct Hook{H<:Tuple}
    hooks::H
end

"""
    Hook(hooks...)

Use this struct to wrap multiple hooks.
"""
Hook(hooks...) = Hook{typeof(hooks)}(hooks)

(h::Hook)(agent, env) = for hook in h.hooks hook(agent,env) end

show_value(h::Hook{<:Tuple}) = [show_value(hook) for hook in h.hooks] 

mutable struct TestEnvironment{E}
    env::E
    n_sim::Int
    log::Vector{Tuple{Float64,Float64}}
    every::Int
    count::Int
    stochastic::Bool
    batchsize::Int
    logger
end
"""
    TestEnvironment(env, n_sim, every = 1; stochastic = false)

Hook object that will evaluate a training agent on the `env` environment for `n_sim` MC simulations every `every` iteration.
Set `stochastic = true` to use the stochastic policy, otherwise the actions will be the modes of the agent's policy.
Calling a test environment returns the mean and the std of the returns obtained for the `n_sim` runs. 
These values are stored in the `log` field of the struct.
"""
function TestEnvironment(env, n_sim, every = 1; stochastic = false, batchsize = 1000) 
    logger = ISLogger(env)
    TestEnvironment(env, n_sim, [(-Inf, -Inf)], every, 0, stochastic, batchsize, logger)
end

function (te::TestEnvironment)(agent::PPOPolicy, envi)
    te.count += 1
    te.count % te.every == 0 || return 0.
    return test_agent(agent, te)[1]
end

function test_agent(agent, te::TestEnvironment)
    totreward = 0.
    reset!(te.env)
    envs = [deepcopy(te.env) for _ in 1:te.n_sim]
    returns = zeros(te.n_sim)
    while all(map(env -> !is_terminated(env),envs))
        s = reduce(hcat, state.(envs)) |> agent.device
        a, σ = agent.actor(s)
        if te.stochastic
            a .+= σ .* (randn(size(σ)) |> agent.device)
        end
        for (env, action) in zip(envs, eachcol(cpu(a)))
            env(collect(action))
        end
        returns .+= map(reward, envs)
    end
    for env in envs
        te.logger(env, log_id = te.count)
    end
    push!(te.log, (mean(returns), std(returns)))
    return mean(returns)
end

Base.getindex(te::TestEnvironment, n) = te.log[n]
show_value(te::TestEnvironment) = ("Test environment return", "$(round(last(te.log)[1])) ± $(round(1.95*last(te.log)[2]/sqrt(te.n_sim)))")

"""
    Kscheduler(Ktarget::Float64, range::UnitRange{Int})

Hook object to linearly anneal the fixed order cost of a single item problem. Linearly anneal from a initial value to be increased by Ktarget over the course of the iterations in `range`. 
#Example
Kscheduler(1280, 1000:4000) will start increasing the fixed order cost of the learning environment at iteration 1000 by 1280/3000 every iteration. At iteration 4000, the fixed order cost will have increased by 1280.
"""
mutable struct Kscheduler
    n::Int
    Ktarget::Float64
    range::UnitRange{Int}
end

Kscheduler(Ktarget, range) = Kscheduler(0, Ktarget, range)

function (ks::Kscheduler)(agent, env) 
    ks.n += 1
    if ks.n in ks.range
        env.bom[1].sources[1].order_cost.K += ks.Ktarget/length(ks.range)
    end
end