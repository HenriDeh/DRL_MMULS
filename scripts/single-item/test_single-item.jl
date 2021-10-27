using DRL_MMULS
using InventoryModels
using Distributions, Flux, BSON, CSV, DataFrames, GLMakie

h = 1
b = 25
K = 1280
CV = 0.5
c = 0
LT = 8
lostsales = false



test_env = sl_sip(h, b, K, CV, c, fill(10.0, 52), 0.0, LT, lostsales = lostsales, horizon = 32)

ins = Scarf.Instance(test_env)
ins.backlog = true
zero_return, std = test_ss_policy(test_env, ins.s, ins.S, 1000)
Scarf.DP_sS(ins, 0.5)
opt_return, std = test_ss_policy(test_env, ins.s, ins.S, 1000)
reset!(test_env)

#BSON.@load "K$(K)_LT$LT.bson" agent env test_env tester#hook
n_iterations = 5000
for _ in 1:9
    env = sl_sip(h, b, 0.0, CV, c, Uniform(1,19), Uniform(-10,20), LT, lostsales = lostsales, horizon = 32, periods = 52)
    agent = PPOPolicy(env, actor_optimiser = ADAM(3f-4), critic_optimiser = ADAM(3f-4), n_hidden = 128,
    γ = 0.99f0,λ = 0.95f0, clip_range = 0.2f0, entropy_weight = 1f-2, n_actors = 25, n_epochs = 10, batch_size = 128,
    target_function = TDλ_target, device = gpu)

    tester = TestEnvironment(test_env, 100, 100)
    run(agent, env, stop_iterations = n_iterations, hook = Hook(tester, DRL_MMULS.Kscheduler(0,K, 1000:9000) ))
end
stop




#=
begin
    Niter = 16
    fig = Figure()
    titles = ["Return" "Loss Actor"; "√loss critic" "entropy"]
    fig[1:2,1:2] = [Axis(fig, title = titles[i,j], xticks = LinearTicks(5)) for i in 1:2, j in 1:2]
    
    exp_smth(arr, fact=0.01) = accumulate((p,n) -> n*fact + p*(1-fact), arr)
    lines!(fig[1,1], 0..Niter, tester.log)
    #hlines!(first(contents(fig[1,1])), [-1.1805515462383565e6])
    #lines!(fig[1,2], 0..Niter, agent.loss_actor ); 
    lines!(fig[1,2], 0..Niter, agent.loss_actor |> exp_smth)
    #lines!(fig[2,1], 0..Niter, agent.loss_critic); 
    lines!(fig[2,1], 0..Niter, agent.loss_critic |> exp_smth)
    #lines!(fig[2,2], 0..Niter, agent.loss_entropy./ agent.entropy_weight); 
    lines!(fig[2,2], 0..Niter, agent.loss_entropy./ agent.entropy_weight |> exp_smth)
end
fig

dashboard(tester, agent)
save("ins40.png", fig)



BSON.@save "ins40.bson" agent env test_env tester
=#