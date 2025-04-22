using QuantumTags
using QuantumTags: Tag
using Muscle: ImmutableVector
using BijectiveDicts: BijectiveDict, BijectiveIdDict
using Serialization
using Random

@kwdef struct GenericTensorNetwork <: AbstractTensorNetwork
    indmap::Dict{Index,Vector{Tensor}} = Dict{Index,Vector{Tensor}}()
    tensors::IdSet{Tensor} = IdSet{Tensor}()

    unsafe::Ref{Union{Nothing,UnsafeScope}} = Ref{Union{Nothing,UnsafeScope}}(nothing)

    # TODO move to a more open and diverse cache?
    sorted_tensors::CachedField{Vector{Tensor}} = CachedField{Vector{Tensor}}()
end

# TODO Find a way to remove the `unsafe` keyword argument from the constructor
function GenericTensorNetwork(tensors; unsafe::Union{Nothing,UnsafeScope}=nothing)
    tensors = IdSet{Tensor}(tensors)

    indmap = reduce(tensors; init=Dict{Index,Vector{Tensor}}()) do dict, tensor
        for index in inds(tensor)
            # TODO use lambda? `Tensor[]` might be reused
            # avoid multiple references to the same tensor
            if isnothing(findfirst(x -> x === tensor, get!(dict, index, Tensor[])))
                push!(dict[index], tensor)
            end
        end
        dict
    end

    # Check index size consistency if not inside an `UnsafeScope`
    if isnothing(unsafe)
        for ind in keys(indmap)
            dims = map(tensor -> size(tensor, ind), indmap[ind])
            length(unique(dims)) == 1 || throw(DimensionMismatch("Index $(ind) has inconsistent dimension: $(dims)"))
        end
    end

    return GenericTensorNetwork(; indmap, tensors, unsafe=Ref{Union{Nothing,UnsafeScope}}(unsafe))
end

function Base.copy(tn::GenericTensorNetwork)
    # `indmap` needs special treatment due to potential ownership issues of the `Vector`s in the values
    indmap = Dict{Index,Vector{Tensor}}(ind => copy(tensors) for (ind, tensors) in tn.indmap)
    tensors = copy(tn.tensors)

    sorted_tensors = CachedField{Vector{Tensor}}()
    unsafe = Ref{Union{Nothing,UnsafeScope}}(tn.unsafe[])

    new_tn = GenericTensorNetwork(; indmap, tensors, sorted_tensors, unsafe)

    # register the new copy to the proper UnsafeScope
    if !isnothing(unsafe[])
        push!(unsafe[].refs, WeakRef(new_tn))
    end

    return new_tn
end

# UnsafeScopeable implementation
get_unsafe_scope(tn::GenericTensorNetwork) = tn.unsafe[]
set_unsafe_scope!(tn::GenericTensorNetwork, uc::Union{Nothing,UnsafeScope}) = tn.unsafe[] = uc

function checksizes(tn::GenericTensorNetwork)
    for (index, tensors) in tn.indmap
        if !allequal(tensor -> size(tensor, index), tensors)
            return false
        end
    end

    return true
end

# TensorNetwork implementation
function all_tensors(tn::GenericTensorNetwork)
    return get!(tn.sorted_tensors) do
        # TODO is okay to use `hash`? we sort to get a "stable" order
        sort!(collect(tn.tensors); by=(x) -> x |> inds .|> hash |> sort)
    end
end

all_inds(tn::GenericTensorNetwork) = collect(keys(tn.indmap))
all_tensors_iter(tn::GenericTensorNetwork) = keys(tn.tensors)
all_inds_iter(tn::GenericTensorNetwork) = keys(tn.indmap)

hastensor(tn::GenericTensorNetwork, tensor) = in(tensor, tn.tensors)
hasind(tn::GenericTensorNetwork, index) = haskey(tn.indmap, index)

ntensors(tn::GenericTensorNetwork) = length(tn.tensors)
ninds(tn::GenericTensorNetwork) = length(tn.indmap)

function size_inds(tn::GenericTensorNetwork)
    return Dict{Index,Int}(index => size(tn, index) for index in keys(tn.indmap))
end

size_ind(tn::GenericTensorNetwork, index::Index) = size(first(tn.indmap[index]), index)

function tensors_contain_inds(tn::GenericTensorNetwork, index::Index)
    @assert hasind(tn, index) "index $index not found in tensor network"
    copy(tn.indmap[index])
end

function tensors_contain_inds(tn::GenericTensorNetwork, indices)
    target_tensors = tensors(tn; contain=first(indices))
    filter!(target_tensors) do tensor
        indices ⊆ inds(tensor)
    end
    return target_tensors
end

## mutating methods
function addtensor_inner!(tn::GenericTensorNetwork, tensor::Tensor)
    hastensor(tn, tensor) && return tn

    # check index sizes if there isn't an active `UnsafeScope` in the Tensor Network
    if isnothing(get_unsafe_scope(tn))
        for i in Iterators.filter(i -> size(tn, i) != size(tensor, i), inds(tensor) ∩ inds(tn))
            throw(
                DimensionMismatch("size(tensor,$i)=$(size(tensor,i)) but should be equal to size(tn,$i)=$(size(tn,i))")
            )
        end
    end

    # do the actual push
    push!(tn.tensors, tensor)
    for index in unique(inds(tensor))
        push!(get!(tn.indmap, index, Tensor[]), tensor)
    end

    # tensors have changed, invalidate cache and reconstruct on next `tensors` call
    invalidate!(tn.sorted_tensors)

    return tn
end

handle!(::GenericTensorNetwork, @nospecialize(e::PushEffect{<:Tensor})) = nothing

function rmtensor_inner!(tn::GenericTensorNetwork, tensor::Tensor)
    # do the actual delete
    for index in unique(inds(tensor))
        filter!(Base.Fix1(!==, tensor), tn.indmap[index])
        tryprune!(tn, index)
    end
    delete!(tn.tensors, tensor)

    # TODO move to `TaggableTensorNetwork`
    # remove tensor tag
    # if haskey(tn.sitemap', tensor)
    #     untag_inner!(tn, tn.sitemap'[tensor])
    # end

    # tensors have changed, invalidate cache and reconstruct on next `tensors` call
    invalidate!(tn.sorted_tensors)

    return tn
end

handle!(::GenericTensorNetwork, @nospecialize(e::DeleteEffect{<:Tensor})) = nothing

function replace_tensor_inner!(tn::GenericTensorNetwork, old_tensor, new_tensor)
    old_tensor === new_tensor && return tn
    hastensor(tn, old_tensor) || throw(ArgumentError("old tensor not found"))
    hastensor(tn, new_tensor) && throw(ArgumentError("new tensor already exists"))

    # TODO check index sizes
    if !isscoped(tn)
        @argcheck issetequal(inds(new_tensor), inds(old_tensor)) "replacing tensor indices don't match"
    end

    # remove old tensor
    for index in unique(inds(old_tensor))
        filter!(Base.Fix1(!==, old_tensor), tn.indmap[index])
        if isempty(tn.indmap[index])
            delete!(tn.indmap, index)
        end
    end
    delete!(tn.tensors, old_tensor)

    # add new tensor
    for index in unique(inds(new_tensor))
        list = get!(tn.indmap, index, Tensor[])
        push!(list, new_tensor)
    end
    push!(tn.tensors, new_tensor)

    # tensors have changed, invalidate cache and reconstruct on next `tensors` call
    invalidate!(tn.sorted_tensors)

    return tn
end

handle!(::GenericTensorNetwork, @nospecialize(e::ReplaceEffect{<:Tensor,<:Tensor})) = nothing

handle!(tn::GenericTensorNetwork, @nospecialize(e::ReplaceEffect{<:Index,<:Index})) = tryprune!(tn, e.old)

function tryprune!(tn::GenericTensorNetwork, i::Index)
    if hasind(tn, i) && isempty(tn.indmap[i])
        delete!(tn.indmap, i)

        # remove index tag
        # if haskey(tn.linkmap', i)
        #     delete!(tn.linkmap, tn.linkmap[i])
        # end
    end

    return tn
end

# function handle!(tn::GenericTensorNetwork, e::ReplaceEffect{Ia,Ib}) where {Ia<:Index,Ib<:Index}
#     tag = tn.linkmap'[e.old]
#     delete!(tn.linkmap, tag)
#     tag_inner!(tn, e.new, tag)
#     return tn
# end

# function handle!(tn::GenericTensorNetwork, @nospecialize(e::ReplaceEffect{A,B})) where {A<:Tensor,B<:Tensor}
#     tag = tn.sitemap'[e.old]
#     delete!(tn.sitemap, tag)
#     tag_inner!(tn, e.new, tag)
#     return tn
# end

# do not allow replacing a tensor with a Tensor Network if we don't know how to handle the tags
# TODO allow it by copying the tags too if posible
# function canhandle(tn::GenericTensorNetwork, @nospecialize(e::ReplaceEffect{T,TN})) where {T<:Tensor,TN<:AbstractTensorNetwork}
#     haskey(tn.sitemap', e.f)
# end

# handle!(::GenericTensorNetwork, @nospecialize(e::ReplaceEffect{T,TensorNetwork}))
# handle!(::GenericTensorNetwork, @nospecialize(e::ReplaceEffect{Vector{A},B})) where {A<:Tensor,B<:Tensor} = nothing

# handle!(::GenericTensorNetwork, @nospecialize(e::TagEffect{S,T})) where {S<:Site,T<:Tensor} = nothing
# handle!(::GenericTensorNetwork, @nospecialize(e::TagEffect{L,I})) where {L<:Link,I<:Index} = nothing
# handle!(::GenericTensorNetwork, @nospecialize(e::UntagEffect{S})) where {S<:Site} = nothing
# handle!(::GenericTensorNetwork, @nospecialize(e::UntagEffect{L})) where {L<:Link} = nothing

# derived methods
Base.:(==)(a::GenericTensorNetwork, b::GenericTensorNetwork) = all(splat(==), zip(tensors(a), tensors(b)))
function Base.isapprox(a::GenericTensorNetwork, b::GenericTensorNetwork; kwargs...)
    return all(((x, y),) -> isapprox(x, y; kwargs...), zip(tensors(a), tensors(b)))
end

function Serialization.serialize(s::AbstractSerializer, obj::GenericTensorNetwork)
    Serialization.writetag(s.io, Serialization.OBJECT_TAG)
    serialize(s, GenericTensorNetwork)
    return serialize(s, all_tensors(obj))
    # TODO fix serialization of tensor tags by storing tensors with a number tag
    # return serialize(s, obj.linkmap)
end

function Serialization.deserialize(s::AbstractSerializer, ::Type{GenericTensorNetwork})
    ts = deserialize(s)
    # linkmap = deserialize(s)
    tn = GenericTensorNetwork(ts)
    # TODO fix deserialization of tensor tags
    # merge!(tn.linkmap, linkmap)
    return tn
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
