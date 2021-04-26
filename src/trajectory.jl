using Flux
mutable struct PPOTrajectory{A <: AbstractArray}
    traces::NamedTuple{(:state, :action, :reward, :next_state, :action_log_prob, :advantage, :target_value), NTuple{7, A}}
    idx::Dict{Symbol, Int}
    N::Int
end

function PPOTrajectory(N, state_size, action_size; device = cpu)
    T = device == gpu ? Float32 : Float64
    PPOTrajectory(
        (state = zeros(T, state_size, N) |> device,
        action = zeros(T, action_size, N) |> device,
        reward = zeros(T, 1, N) |> device,
        next_state = zeros(T, state_size, N) |> device,
        action_log_prob = zeros(T, action_size, N) |> device,
        advantage = zeros(T, 1, N) |> device,
        target_value = zeros(T, 1, N) |> device),
        Dict(:state => 1, :action => 1, :reward => 1, :next_state => 1, :action_log_prob => 1, :advantage => 1, :target_value => 1),
        N
    )
end

function Base.rand(t::PPOTrajectory, n::Int)
    idx = rand(1:t.N, n)
    t.traces.state[:,idx], t.traces.action[:,idx], t.traces.action_log_prob[:,idx], t.traces.advantage[:,idx], t.traces.target_value[:,idx]
end

function Base.empty!(t::PPOTrajectory)
    t.traces.state .= 0f0
    t.traces.action .= 0f0
    t.traces.reward .= 0f0
    t.traces.next_state .= 0f0
    t.traces.action_log_prob .= 0f0
    t.traces.advantage .= 0f0
    t.traces.target_value .= 0f0
    for (k,v) in t.idx
        t.idx[k] = 1
    end
end

function Base.push!(t::PPOTrajectory{A}, data::B, trace::Symbol) where {A <: AbstractMatrix, B <: AbstractMatrix}
    T = size(data)[2]
    t.traces[trace][:,t.idx[trace]:t.idx[trace] + T - 1] .= A(data)
    t.idx[trace] += T
end