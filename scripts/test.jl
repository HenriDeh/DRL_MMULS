using Revise
using InventoryModels
using DRL_MMULS
using Distributions, Flux, Plots, BSON

env = sl_sip(1,10,1280,0.4,0,Uniform(1,19), 52, Uniform(0,30))
test_env = sl_sip(1,10,1280,0.4,0,fill(10,52), 0, pad = false)
ins = Instance(test_env, .99)
scarf_SDP(ins)
test_policy(test_env, collect(Iterators.flatten(zip(ins.s, ins.S))), 10000)
reset!(test_env)

agent = PPOPolicy(env, actor_optimiser = ADAM(1f-4), critic_optimiser = ADAM(1f-1), n_hidden = 64,
γ = 0.99f0,λ = 0.95f0, clip_range = 0.2f0, entropy_weight = 1f-1, n_actors = 100, n_epochs = 40, batch_size = 16,
target_function = TDλ_target, device = gpu)

hook = Hook(
    TestEnvironment(test_env, 2000, 50),
    EntropyAnnealing(LinRange(1f-1,-1f-1, 3000))
)

run(agent, env; stop_iterations = 3000, hook = hook)


plot(plot(agent.loss_actor[1:end], label = "", title = "actor"), plot(agent.loss_entropy[1:end], label = "", title = "entropy"), 
    plot(sqrt.(agent.loss_critic[1:end]), label = "", title = "critic", yscale = :log10), plot(hook.hooks[1].log[50:end], label = "", title = "return"))

print(state(test_env)[[begin, end]], agent.actor(state(test_env)), agent.critic(state(test_env)))

begin
    function p(x)
        μ, _ = agent.actor([abs(min(x,0f0)); fill(10f0, 52); max(x,0f0)])
        s = μ[1]
        S = μ[2]
        return x < s ? S - x : zero(x)
    end

    function s(x) 
        μ, _ = agent.actor([abs(min(x,0f0)); fill(10f0, 52); max(x,0f0)])
        μ[1]
    end

    function S(x)
        μ, _ = agent.actor([abs(min(x,0f0)); fill(10f0, 52); max(x,0f0)])
        μ[2]
    end

    opt(x) = x < first(ins.s) ? first(ins.S) - x : 0.0

    plot([p, opt, s, S], -20,100, label = ["Agent" "Scarf" "̂s" "̂S"])
    hline!([first(ins.s) first(ins.S)], label = ["s" "S"], linestyle = :dash)
end

BSON.@save "1280_annealing1f-1to-1f-1.bson" agent env test_env hook