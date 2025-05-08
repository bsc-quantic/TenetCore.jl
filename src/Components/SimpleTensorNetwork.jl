using QuantumTags
using QuantumTags: Tag
using Muscle: ImmutableVector
using BijectiveDicts: BijectiveDict, BijectiveIdDict
using Serialization
using Random
using Base: IdSet

@kwdef struct SimpleTensorNetwork <: AbstractTensorNetwork
    indmap::Dict{Index,Vector{Tensor}} = Dict{Index,Vector{Tensor}}()
    tensors::IdSet{Tensor} = IdSet{Tensor}()

    unsafe::Ref{Union{Nothing,UnsafeScope}} = Ref{Union{Nothing,UnsafeScope}}(nothing)

    # TODO move to a more open and diverse cache?
    sorted_tensors::CachedField{Vector{Tensor}} = CachedField{Vector{Tensor}}()
end

# TODO Find a way to remove the `unsafe` keyword argument from the constructor
function SimpleTensorNetwork(tensors; unsafe::Union{Nothing,UnsafeScope}=nothing)
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

    return SimpleTensorNetwork(; indmap, tensors, unsafe=Ref{Union{Nothing,UnsafeScope}}(unsafe))
end

function Base.copy(tn::SimpleTensorNetwork)
    # `indmap` needs special treatment due to potential ownership issues of the `Vector`s in the values
    indmap = Dict{Index,Vector{Tensor}}(ind => copy(tensors) for (ind, tensors) in tn.indmap)
    tensors = copy(tn.tensors)

    sorted_tensors = CachedField{Vector{Tensor}}()
    unsafe = Ref{Union{Nothing,UnsafeScope}}(tn.unsafe[])

    new_tn = SimpleTensorNetwork(; indmap, tensors, sorted_tensors, unsafe)

    # register the new copy to the proper UnsafeScope
    if !isnothing(unsafe[])
        push!(unsafe[].refs, WeakRef(new_tn))
    end

    return new_tn
end

# UnsafeScopeable implementation
implements(::UnsafeScopeable, ::SimpleTensorNetwork) = Implements()

get_unsafe_scope(tn::SimpleTensorNetwork) = tn.unsafe[]
set_unsafe_scope!(tn::SimpleTensorNetwork, uc::Union{Nothing,UnsafeScope}) = tn.unsafe[] = uc

function checksizes(tn::SimpleTensorNetwork)
    for (index, tensors) in tn.indmap
        if !allequal(tensor -> size(tensor, index), tensors)
            return false
        end
    end

    return true
end

# TensorNetwork implementation
implements(::TensorNetwork, ::SimpleTensorNetwork) = Implements()

function all_tensors(tn::SimpleTensorNetwork)
    return get!(tn.sorted_tensors) do
        # TODO is okay to use `hash`? we sort to get a "stable" order
        sort!(collect(tn.tensors); by=(x) -> x |> inds .|> hash |> sort)
    end
end

all_tensors_iter(tn::SimpleTensorNetwork) = tn.tensors

all_inds(tn::SimpleTensorNetwork) = collect(keys(tn.indmap))
all_inds_iter(tn::SimpleTensorNetwork) = keys(tn.indmap)

hastensor(tn::SimpleTensorNetwork, tensor) = in(tensor, tn.tensors)
hasind(tn::SimpleTensorNetwork, index) = haskey(tn.indmap, index)

ntensors(tn::SimpleTensorNetwork) = length(tn.tensors)
ninds(tn::SimpleTensorNetwork) = length(tn.indmap)

function size_inds(tn::SimpleTensorNetwork)
    return Dict{Index,Int}(index => size(tn, index) for index in keys(tn.indmap))
end

size_ind(tn::SimpleTensorNetwork, index::Index) = size(first(tn.indmap[index]), index)

function tensors_contain_inds(tn::SimpleTensorNetwork, index::Index)
    @assert hasind(tn, index) "index $index not found in tensor network"
    copy(tn.indmap[index])
end

function tensors_contain_inds(tn::SimpleTensorNetwork, indices)
    target_tensors = tensors(tn; contain=first(indices))
    filter!(target_tensors) do tensor
        indices ⊆ inds(tensor)
    end
    return target_tensors
end

## mutating methods
function addtensor_inner!(tn::SimpleTensorNetwork, tensor::Tensor)
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

handle!(::SimpleTensorNetwork, @nospecialize(e::PushEffect{<:Tensor})) = nothing

function rmtensor_inner!(tn::SimpleTensorNetwork, tensor::Tensor)
    # do the actual delete
    for index in unique(inds(tensor))
        filter!(Base.Fix1(!==, tensor), tn.indmap[index])
        tryprune!(tn, index)
    end
    delete!(tn.tensors, tensor)

    # tensors have changed, invalidate cache and reconstruct on next `tensors` call
    invalidate!(tn.sorted_tensors)

    return tn
end

handle!(::SimpleTensorNetwork, @nospecialize(e::DeleteEffect{<:Tensor})) = nothing

function replace_tensor_inner!(tn::SimpleTensorNetwork, old_tensor, new_tensor)
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

handle!(::SimpleTensorNetwork, @nospecialize(e::ReplaceEffect{<:Tensor,<:Tensor})) = nothing

handle!(tn::SimpleTensorNetwork, @nospecialize(e::ReplaceEffect{<:Index,<:Index})) = tryprune!(tn, e.old)

function tryprune!(tn::SimpleTensorNetwork, i::Index)
    if hasind(tn, i) && isempty(tn.indmap[i])
        delete!(tn.indmap, i)
    end
    return tn
end

function checkeffect(tn::SimpleTensorNetwork, e::FuseEffect)
    @argcheck all(Base.Fix1(hasind, tn), e.inds)
end

# derived methods
Base.:(==)(a::SimpleTensorNetwork, b::SimpleTensorNetwork) = all(splat(==), zip(tensors(a), tensors(b)))
function Base.isapprox(a::SimpleTensorNetwork, b::SimpleTensorNetwork; kwargs...)
    return all(((x, y),) -> isapprox(x, y; kwargs...), zip(tensors(a), tensors(b)))
end

function Serialization.serialize(s::AbstractSerializer, obj::SimpleTensorNetwork)
    Serialization.writetag(s.io, Serialization.OBJECT_TAG)
    serialize(s, SimpleTensorNetwork)
    return serialize(s, all_tensors(obj))
    # TODO fix serialization of tensor tags by storing tensors with a number tag
    # return serialize(s, obj.linkmap)
end

function Serialization.deserialize(s::AbstractSerializer, ::Type{SimpleTensorNetwork})
    ts = deserialize(s)
    # linkmap = deserialize(s)
    tn = SimpleTensorNetwork(ts)
    # TODO fix deserialization of tensor tags
    # merge!(tn.linkmap, linkmap)
    return tn
end
