export test_ss_policy, test_agent, test_rolling_ss_policy, test_simple_ss_policy

function test_ss_policy(env::InventorySystem, s, S, n=1000)
    instance = to_instance(env, 0.99)
    instance.backlog = true
    time = @elapsed Scarf.DP_sS(instance, 1., zero_boundary = true)
    envs = [deepcopy(env) for i in 1:n]
    returns = zeros(size(envs))
    Threads.@threads for (id,environment) in collect(enumerate(envs))
        for t in 1:env.T
            environment([s[t], S[t]])
            returns[id] -= reward(environment)
        end
    end
    return mean(returns), std(returns), time
end

function test_ss_policy(env::SingleItemMMFE, s, S, n=1000)
    envs = [deepcopy(env) for i in 1:n]
    returns = zeros(size(envs))
    time = 0.
    Threads.@threads for (id,environment) in collect(enumerate(envs))
        for t in 1:environment.T
            instance = to_instance(environment, 0.99)
            instance.backlog = true
            time += @elapsed Scarf.DP_sS(instance, 1., zero_boundary = true)
            environment([s[t], S[t]])
            returns[id] -= reward(environment)
        end
    end
    return mean(returns), std(returns), time/n
end

function test_rolling_ss_policy(env::InventorySystem, n=1000; horizon::Int)
    envs = [deepcopy(env) for i in 1:n]
    returns = zeros(size(envs))
    master_instance = to_instance(env, 0.99)
    time = 0.
    for t in 1:env.T
        instance = to_instance(env, 0.99, t:t+horizon-1)
        instance.backlog = true
        time += @elapsed Scarf.DP_sS(instance, 1., zero_boundary = false)
        master_instance.s[t] = instance.s[1]
        master_instance.S[t] = instance.S[1]
    end
    Threads.@threads for (id,environment) in collect(enumerate(envs))
        for t in 1:env.T
            environment([master_instance.s[t], master_instance.S[t]])
            returns[id] -= reward(environment)
        end
    end
    return mean(returns), std(returns), time/n
end

function test_rolling_ss_policy(env::SingleItemMMFE, n =10; horizon::Int)
    envs = [deepcopy(env) for i in 1:n]
    returns = zeros(size(envs))
    time = 0.
    #master_instance = to_instance(env, 0.99)
    Threads.@threads for (id,environment) in collect(enumerate(envs))
        for t in 1:env.T
            instance = to_instance(environment.env, 0.99, t:t+horizon-1)
            instance.backlog = true
            time += @elapsed Scarf.DP_sS(instance, 1., zero_boundary = false)
            environment([instance.s[1], instance.S[1]])
            returns[id] -= reward(environment)
        end
    end
    return mean(returns), std(returns), time/n
end

function test_agent(agent_d, env, n_sims = 1000)
    agent = cpu(agent_d)
    envs = [deepcopy(env) for _ in 1:n_sims]
    returns = zeros(size(envs))
    Threads.@threads for (id,environment) in collect(enumerate(envs))
        for t in 1:env.T
            s = state(environment)
            a, σ = agent.actor(s)
            environment(a)
            returns[id] -= reward(environment)
        end
    end
    return mean(returns), std(returns)
end

function test_simple_ss_policy(_env, n = 1000)
    if _env isa SingleItemMMFE
        env = _env.env
    else
        env = _env
    end
    envs = [deepcopy(env) for i in 1:n]
    returns = zeros(size(envs))
    time = 0.
    
    Threads.@threads for (id,environment) in collect(enumerate(envs))
        for t in 1:env.T
            market = environment.bom[1].market
            CV = InventoryModels.cv(market.demand_dist)
            fc = market.forecasts
            L = environment.bom[1].sources[1].leadtime.leadtime
            LTDmean = sum(fc[1:1+L])
            LTDstd = sqrt(sum((fc[i]*CV)^2 for i in 1:1+L))
            LTD = Normal(LTDmean, LTDstd)
            holding = environment.bom[1].inventory.holding_cost.h
            penalty = market.stockout_cost.b
            setup = environment.bom[1].sources[1].order_cost.K
            s = quantile(LTD, penalty/(penalty+holding))
            TBO = 0 
            while sqrt(2*mean(fc[1:1+TBO])*setup/(holding)) > sum(fc[1:1+TBO]) && TBO < length(fc)
                TBO += 1
            end
            EOQ = sqrt(2*mean(fc[1:1+TBO])*setup/(holding))
            S = s + EOQ
            environment([s, S])
            returns[id] -= reward(environment)
        end
    end 
    return mean(returns), std(returns)
end 

