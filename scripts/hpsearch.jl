using Revise
using InventoryModels
using DRL_MMULS
using Distributions, Flux, DataFrames, CSV

alr = [1f-5, 1f-4, 1f-3]
clr = [1f-3, 1f-2, 1f-1]
λs = [0.95f0]
clips = [0.1f0, 0.2f0]
entropy = [1f-1, 1f-3]
actors = [25,100]
epochs = [20,40]
bs = [32,128]

env = sl_sip(1,10,1280,0.4,0,Uniform(1,19), 52, Uniform(0,30))
test_env = sl_sip(1,10,1280,0.4,0,fill(10,52), 0, pad = false)
ins = Instance(test_env, .99)
scarf_SDP(ins)
test_policy(test_env, collect(Iterators.flatten(zip(ins.s, ins.S))), 10000)
reset!(test_env)

CSV.write("hpsearch.csv", DataFrame(alr = [], clr = [], λ = [], clip = [], entropy = [], n_actors = [], n_epochs = [], batchsize = [], returns = [], last_return = [], time = []))

for (i, (a, c, λ, clip, ent, n_actors, n_epochs, batch_size)) in enumerate(Iterators.product(alr,clr,λs,clips,entropy,actors,epochs,bs))
    println("test $i out of ", length(Iterators.product(alr,clr,λs,clips,entropy,actors,epochs,bs)))
    agent = PPOPolicy(env, actor_optimiser = ADAM(a), critic_optimiser = ADAM(c), n_hidden = 64,
    γ = 0.99f0,λ = λ, clip_range = clip, entropy_weight = ent, n_actors = n_actors, n_epochs = n_epochs, batch_size = batch_size,
    target_function = TDλ_target, device = cpu)

    hook = Hook(
        TestEnvironment(test_env, 2000, 30)
    )
    time = @elapsed run(agent, env; stop_iterations = 3000, hook = hook)
    CSV.write("hpsearch.csv", DataFrame(alr = a, clr = c, λ = λ, clip = clip, entropy = ent, n_actors = n_actors, n_epochs = n_epochs, batchsize = batch_size, 
    returns = [hook.hooks[1].log], last_return = last(hook.hooks[1].log), time = time), append = true)
end