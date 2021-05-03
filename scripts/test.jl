using Revise
using InventoryModels
using DRL_MMULS
using Distributions, Flux, Plots, BSON

LT = 0
K = 1280
env = sl_sip(1,10,K,0.4,0,Uniform(1,19), 52, Uniform(0,30), LT)
test_env = sl_sip(1,10,K,0.4,0,fill(10,52), 0, LT, pad = false)

ins = Scarf.Instance(test_env)
Scarf.DP_sS(ins, 0.5)
opt_return = test_policy(test_env, collect(Iterators.flatten(zip(ins.s, ins.S))), 10000)
reset!(test_env)

#BSON.@load "$K_LT8.bson" agent env test_env hook

agent = PPOPolicy(env, actor_optimiser = ADAM(1f-4), critic_optimiser = ADAM(1f-3), n_hidden = 64,
γ = 0.99f0,λ = 0.95f0, clip_range = 0.1f0, entropy_weight = 1f-3, n_actors = 25, n_epochs = 40, batch_size = 32,
target_function = TDλ_target, device = cpu)


tester = TestEnvironment(test_env, 2000, 50)
entropy_annealer = EntropyAnnealing(LinRange(1f-3,-1f-3, 3000))

run(agent, env; stop_iterations = 2500, hook = Hook(tester))

begin 
    p1 = plot(agent.loss_actor[1:end], label = "", title = "actor");
    p2 = plot(agent.loss_entropy[1:end], label = "", title = "entropy");
    p3 = plot(sqrt.(agent.loss_critic[1:end]), label = "", title = "critic", yscale = :log10);
    p4 = plot(tester.log[1:end]./ opt_return .- 1, label = "", title = "gap");
    lens!(p4, [20,51],[0, 0.25], inset = (1, bbox(20/51,0.0,29.5/51,0.8)));
    plot(p1,p2,p3,p4)
end

begin
    println("Inventory position: ", InventoryModels.inventory_position(test_env.bom[1])) 
    println("Scarf (s,S) : (", ins.s[1],",", ins.S[1], ")")
    println("Agent (s,S) : ", agent.actor(state(test_env)))
    println("Net Present Value : ", agent.critic(state(test_env)))
end

begin
    function p(x)
        μ, _ = agent.actor([abs(min(x,0f0)); fill(10f0, 52) ; fill(0f0, LT); max(x,0f0)])
        s = μ[1]
        S = μ[2]
        return x < s ? S - x : zero(x)
    end

    function s(x) 
        μ, _ = agent.actor([abs(min(x,0f0)); fill(10f0, 52) ; fill(0f0, LT); max(x,0f0)])
        μ[1]
    end

    function S(x)
        μ, _ = agent.actor([abs(min(x,0f0)); fill(10f0, 52) ; fill(0f0, LT); max(x,0f0)])
        μ[2]
    end

    opt(x) = x < first(ins.s) ? first(ins.S) - x : 0.0

    plot([p, opt, s, S], -20,100, label = ["Agent" "Scarf" "̂s" "̂S"])
    hline!([first(ins.s) first(ins.S)], label = ["s" "S"], linestyle = :dash)
end

BSON.@save "1280_LT$LT.bson" agent env test_env hook