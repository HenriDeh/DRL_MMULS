function test_ss_policy(env::InventorySystem, s, S, n=1000)
    total_cost = 0.
    envs = [deepcopy(env) for i in 1:n]
    Threads.@threads for environment in envs
        for t in 1:env.T
            environment([s[t], S[t]])
            total_cost -= reward(environment)
        end
    end
    total_cost /= n
end

function test_agent(agent, env, n_sims = 1000)
    totreward = 0.0
    envs = [deepcopy(env) for _ in 1:n_sims]
    while all(map(env -> !is_terminated(env),envs))
        s = reduce(hcat, map(state, envs)) |> agent.device
        a, σ = agent.actor(s)
        Threads.@threads for (env, action) in collect(zip(envs, eachcol(cpu(a))))
            env(collect(action))
        end
        totreward += sum(reward.(envs))
    end
    return totreward/n_sims
end