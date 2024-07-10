using Revise
using CUDA, KernelAbstractions, StaticArrays, GLMakie, Adapt, ProtoStructs
using SpatialHashTables

include("preprocess.jl")
using SpatialHashTables: dist²

const KA = KernelAbstractions

use_cpu = true

if use_cpu
    backend = CPU() # CUDA.CUDABackend()
    SVec = SVector{3,Float64}
    IntT = Int64 
    FloatT = Float64
else
    backend = CUDA.CUDABackend()
    SVec = SVector{3,Float32}
    IntT = Int32 
    FloatT = Float32    
end

p = (
    R = 0.1,
    N = 100_000,
    repulsion = 2.0,
    hardness = 2.5,
    #
    center = 0.1,
    #
    dt = 0.05, 
    t_end = 3.0,
    dt_safe = 0.2
)

p = preprocess(p; IntT, FloatT)

Base.@kwdef mutable struct State{FVT,FT}
    X::FVT
    V::FVT 
    F::FVT
    t::FT
end

Base.show(io::IO, s::State) = print(io, "State(N = ", length(s.X), " cells, t = ", s.t, ")")

function Adapt.adapt_structure(to, s::State) 
    return State(;
        X = adapt(to, s.X),
        V = adapt(to, s.V),
        F = adapt(to, s.F),
        t = adapt(to, s.t),
    )
end

function savecopy(s::State)
    return State(;
        X = Vector(s.X),
        V = Vector(s.V),
        F = Vector(s.F),
        t = s.t
    )
end

function init(p)

    X = randn(SVector{3,FloatT}, p.N)
    V = similar(X)
    F = similar(X)
    t = FloatT(0.0)

    return State(X,V,F,t)
end

function initplot(s, p; fig = Figure(), ax = Axis3(fig[1,1], aspect = :data), showfig = true)

    s_obs = Observable(s)
    pos = @lift $s_obs.X
    meshscatter!(ax, pos, markersize = p.R, space = :data)

    showfig && display(fig)

    return fig, s_obs 
end

updateplot!(fig, s_obs, s, p) = s_obs[] = s

function addslider!(fig_pos, s_obs, states)
    sl = Slider(fig_pos, range = 1:length(states), startvalue = 1)
    on(sl.value) do i
        updateplot!(fig, s_obs, states[i], p)
    end
    return sl
end

@kernel function interaction_kernel!(F, @Const(X), V, @Const(p), grid)
    i = @index(Global)
    FT = eltype(eltype(X))
    
    Xi = X[i]
    Fi = zero(Xi)

    Fi -= p.center * Xi

    # parameters
    R² = p.R*p.R

    for j in neighbours(grid, Xi, FT(2)*p.R)
        Xj = X[j]
        Xij = Xi - Xj 
        d² = dist²(Xij)

        if FT(0) < d² < FT(4)*R²  
            d = sqrt(d²)
            δ = FT(2)*p.R - d 

            Fi += p.repulsion * δ^(p.hardness-FT(2))/d * Xij
        end
    end
    F[i] = Fi 
end

function simulate(s, p, callback)
    backend = KA.get_backend(s.X)

    FT = eltype(eltype(s.X))
    grid = HashGrid(FloatT(2)*p.R, p.N, s.X, FT == Float32 ? CuVector{Int32} : Vector{Int64}; nthreads = 32)

    t_safe_last = 0.0
    states = [savecopy(s)]

    n_steps = ceil(Int, p.t_end / p.dt)

    ik = interaction_kernel!(backend, 32, length(s.X))

    for _ in 1:n_steps

        updatecells!(grid, s.X)
        KA.synchronize(backend)

        ik(s.F, s.X, s.V, p, grid)
        KA.synchronize(backend)

        @. s.X += p.dt * s.F

        if t_safe_last > p.dt_safe 
            push!(states, savecopy(s))
            isnothing(callback) || callback(states[end], p)

            t_safe_last = 0.0
        else
            t_safe_last += p.dt_safe
        end

    end
    return states
end

begin
    s = init(p)
    fig, s_obs = initplot(s, p)

    s = adapt(backend, s)
    
    states = simulate(s, p, (s, p) -> updateplot!(fig, s_obs, s, p))

    addslider!(fig[2,1], s_obs, states)
end

