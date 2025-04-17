using QuantumTags
using QuantumTags: Tag
using Muscle: ImmutableVector
using BijectiveDicts: BijectiveDict, BijectiveIdDict
using Serialization
using Random

const LinkBiDict = BijectiveDict{Link,Index,Dict{Link,Index},Dict{Index,Link}}
const SiteBiDict = BijectiveDict{Site,Tensor,Dict{Site,Tensor},IdDict{Tensor,Site}}

@kwdef struct GenericTensorNetwork <: AbstractTensorNetwork
    indmap::Dict{Index,Vector{Tensor}} = Dict{Index,Vector{Tensor}}()
    tensors::IdSet{Tensor} = IdSet{Tensor}()

    # TODO move tag info to a `TaggedTensorNetwork` type?
    linkmap::LinkBiDict = LinkBiDict()
    sitemap::SiteBiDict = SiteBiDict()

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
    linkmap = copy(tn.linkmap)
    sitemap = copy(tn.sitemap)
    sorted_tensors = CachedField{Vector{Tensor}}()
    unsafe = Ref{Union{Nothing,UnsafeScope}}(tn.unsafe[])

    new_tn = GenericTensorNetwork(; indmap, tensors, linkmap, sitemap, sorted_tensors, unsafe)

    if !isnothing(unsafe[])
        push!(unsafe[].refs, WeakRef(new_tn)) # Register the new copy to the proper UnsafeScope
    end

    return new_tn
end

# UnsafeScope implementation
get_unsafe_scope(tn::GenericTensorNetwork) = tn.unsafe[]
set_unsafe_scope!(tn::GenericTensorNetwork, uc::Union{Nothing,UnsafeScope}) = tn.unsafe[] = uc

function checksizes(tn::GenericTensorNetwork)
    # Iterate through each index in the indmap
    for (index, tensors) in tn.indmap
        # Get the size of the first tensor for this index
        reference_size = size(tensors[1], index)

        # Compare the size of each subsequent tensor for this index
        for tensor in tensors
            if size(tensor, index) != reference_size
                return false
            end
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

tensors_contain_inds(tn::GenericTensorNetwork, index::Index) = copy(tn.indmap[index])
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

function rmtensor_inner!(tn::GenericTensorNetwork, tensor::Tensor)
    # do the actual delete
    for index in unique(inds(tensor))
        filter!(Base.Fix1(!==, tensor), tn.indmap[index])
        tryprune!(tn, index)
    end
    delete!(tn.tensors, tensor)

    # remove tensor tag
    if haskey(tn.sitemap', tensor)
        untag_inner!(tn, tn.sitemap'[tensor])
    end

    # tensors have changed, invalidate cache and reconstruct on next `tensors` call
    invalidate!(tn.sorted_tensors)

    return tn
end

# function replace_tensor_inner!(tn, )

function tryprune!(tn::GenericTensorNetwork, i::Index)
    if hasind(tn, i) && isempty(tn.indmap[i])
        delete!(tn.indmap, i)

        # remove index tag
        if haskey(tn.linkmap', i)
            delete!(tn.linkmap, tn.linkmap[i])
        end
    end

    return tn
end

# required for `canhandle` and thus, `push!`, `delete!` and `replace!`  to work
# TODO might be a good idea to move `addtensor_inner!` and `rmtensor_inner!` to this
handle!(::GenericTensorNetwork, @nospecialize(e::PushEffect{T})) where {T<:Tensor} = nothing
handle!(::GenericTensorNetwork, e::DeleteEffect{T}) where {T<:Tensor} = nothing

function handle!(tn::GenericTensorNetwork, e::ReplaceEffect{Ia,Ib}) where {Ia<:Index,Ib<:Index}
    tag = tn.linkmap'[e.old]
    delete!(tn.linkmap, tag)
    tag_inner!(tn, e.new, tag)
    return tn
end

function handle!(tn::GenericTensorNetwork, @nospecialize(e::ReplaceEffect{A,B})) where {A<:Tensor,B<:Tensor}
    tag = tn.sitemap'[e.old]
    delete!(tn.sitemap, tag)
    tag_inner!(tn, e.new, tag)
    return tn
end

# do not allow replacing a tensor with a Tensor Network if we don't know how to handle the tags
# TODO allow it by copying the tags too if posible
# function canhandle(tn::GenericTensorNetwork, @nospecialize(e::ReplaceEffect{T,TN})) where {T<:Tensor,TN<:AbstractTensorNetwork}
#     haskey(tn.sitemap', e.f)
# end

# handle!(::GenericTensorNetwork, @nospecialize(e::ReplaceEffect{T,TensorNetwork}))
# handle!(::GenericTensorNetwork, @nospecialize(e::ReplaceEffect{Vector{A},B})) where {A<:Tensor,B<:Tensor} = nothing

handle!(::GenericTensorNetwork, @nospecialize(e::TagEffect{S,T})) where {S<:Site,T<:Tensor} = nothing
handle!(::GenericTensorNetwork, @nospecialize(e::TagEffect{L,I})) where {L<:Link,I<:Index} = nothing
handle!(::GenericTensorNetwork, @nospecialize(e::UntagEffect{S})) where {S<:Site} = nothing
handle!(::GenericTensorNetwork, @nospecialize(e::UntagEffect{L})) where {L<:Link} = nothing

# Taggable implementation
all_sites(tn::GenericTensorNetwork) = collect(keys(tn.sitemap))
all_links(tn::GenericTensorNetwork) = collect(keys(tn.linkmap))
all_sites_iter(tn::GenericTensorNetwork) = keys(tn.sitemap)
all_links_iter(tn::GenericTensorNetwork) = keys(tn.linkmap)

sites(tn::GenericTensorNetwork) = collect(keys(tn.sitemap))
links(tn::GenericTensorNetwork) = collect(keys(tn.linkmap))
links_iter(tn::GenericTensorNetwork) = keys(tn.linkmap)

function tag_inner!(tn::GenericTensorNetwork, tensor::Tensor, site::Site)
    hastensor(tn, tensor) || throw(ArgumentError("Tensor not found in TensorNetwork"))
    hassite(tn, site) && throw(ArgumentError("Site $(site) already exists in TensorNetwork"))

    tn.sitemap[site] = tensor

    return tn
end

function tag_inner!(tn::GenericTensorNetwork, ind::Index, link::Link)
    hasind(tn, ind) || throw(ArgumentError("$ind not found in TensorNetwork"))
    haslink(tn, link) && throw(ArgumentError("Link $(link) already exists in TensorNetwork"))

    tn.linkmap[link] = ind

    return tn
end

function untag_inner!(tn::GenericTensorNetwork, site::Site)
    hassite(tn, site) || throw(ArgumentError("Site $(site) not found in TensorNetwork"))
    delete!(tn.sitemap, site)
    return tn
end

function untag_inner!(tn::GenericTensorNetwork, link::Link)
    haslink(tn, link) || throw(ArgumentError("Link $(link) not found in TensorNetwork"))
    delete!(tn.linkmap, link)
    return tn
end

hastag(tn::GenericTensorNetwork, tag) = haskey(tn.linkmap, tag) || haskey(tn.sitemap, tag)
ntags(tn::GenericTensorNetwork) = length(tn.linkmap) + length(tn.sitemap)

tensor_at(tn::GenericTensorNetwork, tag) = tn.sitemap[tag]
ind_at(tn::GenericTensorNetwork, tag) = tn.linkmap[tag]
tag_at(tn::GenericTensorNetwork, tensor::Tensor) = tn.sitemap'[tensor]
tag_at(tn::GenericTensorNetwork, index::Index) = tn.linkmap'[index]

# derived methods
Base.:(==)(a::GenericTensorNetwork, b::GenericTensorNetwork) = all(splat(==), zip(tensors(a), tensors(b)))
function Base.isapprox(a::GenericTensorNetwork, b::GenericTensorNetwork; kwargs...)
    return all(((x, y),) -> isapprox(x, y; kwargs...), zip(tensors(a), tensors(b)))
end

function Serialization.serialize(s::AbstractSerializer, obj::GenericTensorNetwork)
    Serialization.writetag(s.io, Serialization.OBJECT_TAG)
    serialize(s, GenericTensorNetwork)
    serialize(s, tensors(obj))
    # TODO fix serialization of tensor tags by storing tensors with a number tag
    return serialize(s, obj.linkmap)
end

function Serialization.deserialize(s::AbstractSerializer, ::Type{GenericTensorNetwork})
    ts = deserialize(s)
    linkmap = deserialize(s)
    tn = GenericTensorNetwork(ts)
    # TODO fix deserialization of tensor tags
    merge!(tn.linkmap, linkmap)
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
