using ParameterSchedulers
import ParameterSchedulers.Scheduler
export EntropyWeightDecayer, EpsilonDecayer

show_value(h::Any) = ("$(typeof(h))", "hook")

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
show_value(ks::Kscheduler) = ("Kscheduler", "Target $(ks.Ktarget) in $(ks.range)")

mutable struct Normalizer{T}
    mean::T
    m2::T
    std::T
    count::Int
end


function Normalizer(n::Int)
    Normalizer(zeros(Float32, n), ones(Float32, n), ones(Float32, n), 0)
end

function Normalizer()
    Normalizer(0f0, 1f0, 1f0, 0)
end

function update(n::Normalizer{<:Vector}, inputs::AbstractVecOrMat)
    for input in eachcol(inputs)
        n.count += 1
        tmp_mean = n.mean
        n.mean .= (n.count-1)/n.count .* n.mean .+ input ./ n.count
        n.m2 .+= (input .- tmp_mean) .* (input .- n.mean)
        n.std .= max.(sqrt.(n.m2 ./ (n.count-1)), 1f-6)
    end
end


function update(n::Normalizer{<:Number}, inputs)
    for input in inputs
        n.count += 1
        tmp_mean = n.mean
        n.mean = (n.count-1)/n.count * n.mean + input / n.count
        n.m2 += (input - tmp_mean) .* (input - n.mean)
        n.std = max(sqrt(n.m2 / (n.count-1)), 1e-6)
    end
end

function update(::Any, ::Any)
end

function (n::Normalizer)(input)
    input .-= n.mean 
    input ./= n.std
end

mutable struct EntropyWeightDecayer 
    schedule
    t::Int
end

EntropyWeightDecayer(schedule) = EntropyWeightDecayer(schedule, 1)

function (ewd::EntropyWeightDecayer)(agent, env)
    ewd.t += 1
    agent.entropy_weight = ewd.schedule(ewd.t)
end

show_value(ew::EntropyWeightDecayer) = ("Entropy weight: ", ew.schedule(ew.t-1))

mutable struct EpsilonDecayer
    schedule
    t::Int
end

EpsilonDecayer(schedule) = EpsilonDecayer(schedule, 1)

function (ed::EpsilonDecayer)(agent, env) 
    ed.t += 1
    agent.clip_range = ed.schedule(ed.t)
end

show_value(ed::EpsilonDecayer) = ("clip range ", ed.schedule(ed.t-1))

