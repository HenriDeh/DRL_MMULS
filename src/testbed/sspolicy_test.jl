function test_ss_policy(env::InventorySystem, s, S, n=1000)
    total_cost = 0.
    envs = [deepcopy(env) for i in 1:n]
    returns = zeros(size(envs))
    for (id,environment) in enumerate(envs)
        for t in 1:env.T
            environment([s[t], S[t]])
            returns[id] -= reward(environment)
        end
    end
    return mean(returns), std(returns)
end

function test_agent(agent, env, n_sims = 1000)
    RLBase.reset!(env)
    envs = [deepcopy(env) for _ in 1:n_sims]
    returns = zeros(size(envs)...)
    while all(map(env -> !RLBase.is_terminated(env),envs))
        s = reduce(hcat, map(state, envs)) |> s -> send_to_device(device(agent), s)
        a_t, σ = agent.policy.policy(s, is_sampling = false)
        a = dropdims(a_t, dims = 2)
        Threads.@threads for (env, action) in collect(zip(envs, eachcol(cpu(a))))
            env(collect(action))
        end
        returns .+= map(reward, envs)
    end
    return mean(returns), std(returns)
end