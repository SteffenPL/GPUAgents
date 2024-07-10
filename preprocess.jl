
preprocess(x; kwargs...) = x

function preprocess(d::Dict; kwargs...)
    return preprocess(NamedTuple(Symbol(k) => preprocess(v; kwargs...) for (k,v) in d); kwargs...)
end

function preprocess(d::NamedTuple; kwargs...)
    return NamedTuple(Symbol(k) => preprocess(v; kwargs...) for (k,v) in pairs(d))
end

intT(;IntT = Int64, kwargs...) = IntT 
floatT(;FloatT = Float64, kwargs...) = FloatT 

preprocess(x::Vector{Union{Float64, Int64}}; kwargs...) = preprocess(floatT(;kwargs...).(x); kwargs...)
preprocess(x::Vector{Float64}; kwargs...) = SVector{length(x), floatT(;kwargs...)}(x)
preprocess(x::Vector{Int64}; kwargs...) = SVector{length(x), intT(;kwargs...)}(x)
preprocess(x::Vector; kwargs...) = Tuple(preprocess.(x; kwargs...))

preprocess(x::Float64; kwargs...) = floatT(;kwargs...)(x)
preprocess(x::Int64; kwargs...) = intT(;kwargs...)(x)

# domains:
function preprocess(x::@NamedTuple{center::SVecT, size::SVecT}; kwargs...) where SVecT
    c = SVector(preprocess(x.center; kwargs...))
    s = SVector(preprocess(x.size; kwargs...))
    min = c .- eltype(c)(0.5) .* s
    max = c .+ eltype(c)(0.5) .* s
    inv_size = inv.(s)
    return (center = c, size = s, min = min, max = max, inv_size = inv_size)
end