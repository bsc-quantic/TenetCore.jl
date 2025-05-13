using QuantumTags
using QuantumTags: Tag
using Serialization
using Random
using Base: IdSet

@kwdef struct GenericTensorNetwork <: AbstractTensorNetwork
    tn::SimpleTensorNetwork = SimpleTensorNetwork()
    tags::TagMixin = TagMixin()
end

GenericTensorNetwork(tn::SimpleTensorNetwork; kwargs...) = GenericTensorNetwork(; tn, kwargs...)

# TODO Find a way to remove the `unsafe` keyword argument from the constructor
function GenericTensorNetwork(tensors; kwargs...)
    tn = SimpleTensorNetwork(tensors; kwargs...)
    tags = TagMixin()
    return GenericTensorNetwork(; tn, tags)
end

Base.copy(tn::GenericTensorNetwork) = GenericTensorNetwork(copy(tn.tn), copy(tn.tags))

# delegation
DelegatorTrait(::Network, ::GenericTensorNetwork) = DelegateTo{:tn}()
DelegatorTrait(::UnsafeScopeable, ::GenericTensorNetwork) = DelegateTo{:tn}()
DelegatorTrait(::TensorNetwork, ::GenericTensorNetwork) = DelegateTo{:tn}()
DelegatorTrait(::Taggable, ::GenericTensorNetwork) = DelegateTo{:tags}()

# effects
function handle!(tn::GenericTensorNetwork, @nospecialize(e::RemoveTensorEffect))
    # notify the mixin that a tensor was deleted
    handle!(tn.tags, e)

    # notify the tensor network that a tensor was deleted
    handle!(tn.tn, e)
end

function handle!(tn::GenericTensorNetwork, @nospecialize(e::ReplaceEffect{<:Tensor,<:Tensor}))
    # notify the mixin that a tensor was deleted
    handle!(tn.tags, e)

    # notify the tensor network that a tensor was deleted
    handle!(tn.tn, e)
end

function handle!(tn::GenericTensorNetwork, @nospecialize(e::ReplaceEffect{<:Index,<:Index}))
    # notify the mixin that a tensor was deleted
    handle!(tn.tags, e)

    # notify the tensor network that a tensor was deleted
    handle!(tn.tn, e)
end

function handle!(tn::GenericTensorNetwork, @nospecialize(e::ReplaceEffect{<:Tag,<:Tag}))
    # notify the mixin that a tensor was deleted
    handle!(tn.tags, e)
end

# derived methods
Base.:(==)(a::GenericTensorNetwork, b::GenericTensorNetwork) = all(splat(==), zip(tensors(a), tensors(b)))
function Base.isapprox(a::GenericTensorNetwork, b::GenericTensorNetwork; kwargs...)
    return all(((x, y),) -> isapprox(x, y; kwargs...), zip(tensors(a), tensors(b)))
end

"""
    rand(TensorNetwork, n::Integer, regularity::Integer; out = 0, dim = 2:9, seed = nothing, globalind = false)

Generate a random tensor network.

# Arguments

  - `n` Number of tensors.
  - `regularity` Average number of indices per tensor.
  - `out` Number of open indices.
  - `dim` Range of dimension sizes.
  - `seed` If not `nothing`, seed random generator with this value.
  - `globalind` Add a global 'broadcast' dimension to every tensor.
"""
function Base.rand(
    rng::Random.AbstractRNG,
    ::Type{TensorNetwork},
    n::Integer,
    regularity::Integer;
    out=0,
    dim=2:9,
    seed=nothing,
    globalind=false,
    eltype=Float64,
)
    !isnothing(seed) && Random.seed!(rng, seed)

    inds = letter.(randperm(n * regularity ÷ 2 + out))
    size_dict = Dict(ind => rand(dim) for ind in inds)

    outer_inds = collect(Iterators.take(inds, out))
    inner_inds = collect(Iterators.drop(inds, out))

    candidate_inds = shuffle(
        collect(Iterators.flatten([outer_inds, Iterators.flatten(Iterators.repeated(inner_inds, 2))]))
    )

    inputs = map(x -> [x], Iterators.take(candidate_inds, n))

    for ind in Iterators.drop(candidate_inds, n)
        i = rand(1:n)
        while ind in inputs[i]
            i = rand(1:n)
        end

        push!(inputs[i], ind)
    end

    if globalind
        ninds = length(size_dict)
        ind = letter(ninds + 1)
        size_dict[ind] = rand(dim)
        push!(outer_inds, ind)
        push!.(inputs, (ind,))
    end

    tensors = Tensor[Tensor(rand(eltype, [size_dict[ind] for ind in input]...), tuple(input...)) for input in inputs]
    return GenericTensorNetwork(tensors)
end

function Base.rand(::Type{TensorNetwork}, n::Integer, regularity::Integer; kwargs...)
    return rand(Random.default_rng(), TensorNetwork, n, regularity; kwargs...)
end
