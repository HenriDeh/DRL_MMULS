using GLMakie
using InventoryModels: make_pane!

function InventoryModels.dashboard(tester, agent)
    f = Figure(resolution = (1600,900))
    gl = f[1,1] = GridLayout()
    make_pane!(f, gl, tester.logger)
    gl2 = f[1,2] = GridLayout()
    make_tester_pane!(f, gl2, tester, agent)
    display(f)
end

exp_smth(arr) = accumulate((o,n)-> o*0.8+n*0.2, arr)

function make_tester_pane!(fig, gl, tester, agent)
    menu = gl[1,1] = Menu(fig, options = ["Return", "Actor loss", "√Critic loss", "Entropy"])
    menu.i_selected = 1
    maxx =  begin 
        m = maximum(tester.logger[first(keys(tester.logger.logs))].log_id)
        if m == 0 
            m = maximum(tester.logger[first(keys(tester.logger))].simulation_id)
        end
        m
    end
    n_tester = Node(tester)
    n_agent = Node(agent)
    ax = gl[2,1] = Axis(fig, xticks = LinearTicks(5), xzoomlock = true, xpanlock = true, xlabel = "Iterations (× 1000)")
    exp_tog = Toggle(fig)
    gl[3,1] = grid!([exp_tog Label(fig, "Exponential smoothing")], tellwidth = false)
    ys = @lift begin
        if $(menu.selection) == "Return"
            $(n_tester).log[2:end]
        elseif $(menu.selection) == "Actor loss"
            $(n_agent).loss_actor
        elseif $(menu.selection) == "√Critic loss"
            $(n_agent).loss_critic
        elseif $(menu.selection) == "Entropy"
            $(n_agent).loss_entropy./$(n_agent).entropy_weight
        else
            fill(Inf32, $maxx)  
        end
    end
    l = @lift(max(1, length($ys)÷maxx[]))
    ym = @lift mean.(Iterators.partition($ys, $l))
    y = @lift $(exp_tog.active) ? exp_smth($ym) : $ym
    on(menu.selection) do n
        autolimits!(ax)        
    end
    
    @lift begin 
        empty!(ax)
        lines!(ax, 0..(maxx[]/1000), $y, color = :red)
    end
    return nothing
end