using DelegatorTraits
using Base: AbstractVecOrTuple
using ArgCheck
using ValSplit
using Networks
using QuantumTags
using Muscle: Muscle

# interface object
"""
    TensorNetwork <: Interface

A singleton type that represents the basic interface of a Tensor Network.
A type implementing this interface should also implement the `Network` interface.
"""
struct TensorNetwork <: Interface end

# NOTE do not name it `copy` because it can break calls to `Base.copy`
# function copy_tn end

# query methods
## in reality, the only required methods are `all_*` and the mutating methods
function tensors end
# function inds end # WARN moved to `Operations/AbstractTensorNetwork.jl` to avoid type-piracy

function tensor end
function ind end

function all_tensors end
function all_inds end

function all_tensors_iter end
function all_inds_iter end

function hastensor end
function hasind end

function ntensors end
function ninds end

function tensors_with_inds end
function tensors_contain_inds end
function tensors_intersect_inds end

function inds_set end
function inds_parallel_to end

function size_inds end
function size_ind end

function tensor_at end
function ind_at end

# mutating methods
function addtensor! end
function rmtensor! end
function replace_tensor! end
function replace_ind! end

"""
    slice!(tn, index::Symbol, i)

In-place projection of `index` on dimension `i`.

See also: [`selectdim`](@ref), [`view`](@ref).
"""
function slice! end

"""
    fuse!(tn, ind)

Group indices parallel to `ind` and reshape the tensors accordingly.
"""
function fuse! end

# TODO contract!, split!

# implementation
## `tensors`
tensors(tn; kwargs...) = tensors(sort_nt(values(kwargs)), tn)
tensors(::@NamedTuple{}, tn) = all_tensors(tn)

# TODO fix grammar error on naming
tensors(kwargs::NamedTuple{(:contain,)}, tn) = tensors_contain_inds(tn, kwargs.contain)
tensors(kwargs::NamedTuple{(:intersect,)}, tn) = tensors_intersect_inds(tn, kwargs.intersect)
tensors(kwargs::NamedTuple{(:withinds,)}, tn) = tensors_with_inds(tn, kwargs.withinds)

@deprecate tensors(kwargs::NamedTuple{(:contains,)}, tn) tensors(tn; contain=kwargs.contains)
@deprecate tensors(kwargs::NamedTuple{(:intersects,)}, tn) tensors(tn; intersect=kwargs.intersects)

### singular version of `tensors`
tensor(tn; kwargs...) = tensor(sort_nt(values(kwargs)), tn)
tensor(kwargs::NamedTuple, tn) = only(tensors(kwargs, tn))
tensor(kwargs::NamedTuple{(:at,)}, tn) = tensor_at(tn, kwargs.at)

## `inds`
# NOTE moved to `Operations/AbstractTensorNetwork.jl` to avoid type-piracy

### singular version of `inds`
ind(tn; kwargs...) = ind(sort_nt(values(kwargs)), tn)
ind(kwargs::NamedTuple, tn) = only(inds(kwargs, tn))
ind(kwargs::NamedTuple{(:at,)}, tn) = ind_at(tn, kwargs.at)

## `all_tensors`
all_tensors(tn) = all_tensors(tn, DelegatorTrait(TensorNetwork(), tn))
all_tensors(tn, ::DelegateToField) = all_tensors(delegator(TensorNetwork(), tn))
all_tensors(tn, ::DontDelegate) = throw(MethodError(all_tensors, (tn,)))

## `all_inds`
all_inds(tn) = all_inds(tn, DelegatorTrait(TensorNetwork(), tn))
all_inds(tn, ::DelegateToField) = all_inds(delegator(TensorNetwork(), tn))
function all_inds(tn, ::DontDelegate)
    fallback(all_inds)
    mapreduce(inds, ∪, tensors(tn); init=Index[])
end

## `all_tensors_iter`
all_tensors_iter(tn) = all_tensors_iter(tn, DelegatorTrait(TensorNetwork(), tn))
all_tensors_iter(tn, ::DelegateToField) = all_tensors_iter(delegator(TensorNetwork(), tn))
function all_tensors_iter(tn, ::DontDelegate)
    fallback(all_tensors_iter)
    all_tensors(tn)
end

## `all_inds_iter`
all_inds_iter(tn) = all_inds_iter(tn, DelegatorTrait(TensorNetwork(), tn))
all_inds_iter(tn, ::DelegateToField) = all_inds_iter(delegator(TensorNetwork(), tn))
function all_inds_iter(tn, ::DontDelegate)
    fallback(all_inds_iter)
    all_inds(tn)
end

## `hastensor`
hastensor(tn, tensor) = hastensor(tn, tensor, DelegatorTrait(TensorNetwork(), tn))
hastensor(tn, tensor, ::DelegateToField) = hastensor(delegator(TensorNetwork(), tn), tensor)
function hastensor(tn, tensor, ::DontDelegate)
    fallback(hastensor)
    any(Base.Fix1(===, tensor), all_tensors(tn))
end

## `hasind`
hasind(tn, i) = hasind(tn, i, DelegatorTrait(TensorNetwork(), tn))
hasind(tn, i, ::DelegateToField) = hasind(delegator(TensorNetwork(), tn), i)
function hasind(tn, i, _)
    fallback(hasind)
    i ∈ all_inds(tn)
end

## `ntensors`
ntensors(tn; kwargs...) = ntensors(sort_nt(values(kwargs)), tn)

function ntensors(kwargs::NamedTuple, tn)
    fallback(ntensors)
    length(tensors(kwargs, tn))
end

### dispatch due to performance reasons: see implementation in src/GenericTensorNetwork.jl
ntensors(::@NamedTuple{}, tn) = ntensors((;), tn, DelegatorTrait(TensorNetwork(), tn))
ntensors(::@NamedTuple{}, tn, ::DelegateToField) = ntensors(delegator(TensorNetwork(), tn))
function ntensors(::@NamedTuple{}, tn, ::DontDelegate)
    fallback(ntensors)
    length(all_tensors(tn))
end

## `ninds`
ninds(tn; kwargs...) = ninds(sort_nt(values(kwargs)), tn)

function ninds(kwargs::NamedTuple, tn)
    fallback(ninds)
    length(inds(kwargs, tn))
end

### dispatch due to performance reasons: see implementation in src/GenericTensorNetwork.jl
ninds(::@NamedTuple{}, tn) = ninds((;), tn, DelegatorTrait(TensorNetwork(), tn))
ninds(::@NamedTuple{}, tn, ::DelegateToField) = ninds((;), delegator(TensorNetwork(), tn))
function ninds(::@NamedTuple{}, tn, ::DontDelegate)
    fallback(ninds)
    length(all_inds(tn))
end

## `tensors_with_inds`
function tensors_with_inds(tn, withinds::T) where {T<:AbstractVecOrTuple{<:Index}}
    filter(t -> issetequal(inds(t), withinds), tensors(tn; contain=withinds))
end

## `tensors_contain_inds`
tensors_contain_inds(tn, target) = tensors_contain_inds(tn, target, DelegatorTrait(TensorNetwork(), tn))
tensors_contain_inds(tn, target, ::DelegateToField) = tensors_contain_inds(delegator(TensorNetwork(), tn), target)
tensors_contain_inds(tn, target, ::DontDelegate) = filter(Base.Fix2(⊇, target) ∘ inds, tensors(tn))
tensors_contain_inds(tn, target::Index, ::DontDelegate) = tensors_contain_inds(tn, [target], DontDelegate())

## `tensors_intersect_inds`
tensors_intersect_inds(tn, target::Index) = tensors_intersect_inds(tn, [target])
function tensors_intersect_inds(tn, target::AbstractVecOrTuple)
    filter(t -> !isdisjoint(inds(t), target), tensors(tn))
end

## `inds_set`
@valsplit function inds_set(tn, Val(set::Symbol))
    throw(ArgumentError("Unknown query: set=$(set)"))
end

inds_set(tn, ::Val{:all}) = all_inds(tn)

inds_set(tn, ::Val{:open}) = inds_set_open(tn)
inds_set_open(tn) = inds_set_open(tn, DelegatorTrait(TensorNetwork(), tn))::Vector{<:Index}
inds_set_open(tn, ::DelegateToField) = inds_set_open(delegator(TensorNetwork(), tn))
function inds_set_open(tn, ::DontDelegate)
    fallback(inds_set_open)
    selected = Index[]
    histogram = hist(Iterators.flatten(Iterators.map(inds, tensors(tn))); init=Dict{Index,Int}())
    append!(selected, Iterators.map(first, Iterators.filter(((k, c),) -> c == 1, histogram)))
    return selected
end

inds_set(tn, ::Val{:inner}) = inds_set_inner(tn)
inds_set_inner(tn) = inds_set_inner(tn, DelegatorTrait(TensorNetwork(), tn))::Vector{<:Index}
inds_set_inner(tn, ::DelegateToField) = inds_set_inner(delegator(TensorNetwork(), tn))
function inds_set_inner(tn, ::DontDelegate)
    fallback(inds_set_inner)
    selected = Index[]
    histogram = hist(Iterators.flatten(Iterators.map(inds, tensors(tn))); init=Dict{Index,Int}())
    append!(selected, first.(Iterators.filter(((k, c),) -> c == 2, histogram)))
    return selected
end

inds_set(tn, ::Val{:hyper}) = inds_set_hyper(tn)
inds_set_hyper(tn) = inds_set_hyper(tn, DelegatorTrait(TensorNetwork(), tn))::Vector{<:Index}
inds_set_hyper(tn, ::DelegateToField) = inds_set_hyper(delegator(TensorNetwork(), tn))
function inds_set_hyper(tn, ::DontDelegate)
    fallback(inds_set_hyper)
    selected = Index[]
    histogram = hist(Iterators.flatten(Iterators.map(inds, tensors(tn))); init=Dict{Index,Int}())
    append!(selected, Iterators.map(first, Iterators.filter(((k, c),) -> c >= 3, histogram)))
    return selected
end

## `inds_parallel_to`
function inds_parallel_to(tn, parallel_to)
    candidates = filter!(!=(parallel_to), collect(mapreduce(inds, ∩, tensors(tn; contain=parallel_to))))
    return filter(candidates) do i
        length(tensors(tn; contain=i)) == length(tensors(tn; contain=parallel_to))
    end
end

## `size_inds`
size_inds(tn) = size_inds(tn, DelegatorTrait(TensorNetwork(), tn))
size_inds(tn, ::DelegateToField) = size_inds(delegator(TensorNetwork(), tn))
function size_inds(tn, ::DontDelegate)
    fallback(size_inds)
    sizes = Dict{Index,Int}()
    for tensor in tensors(tn)
        for ind in inds(tensor)
            sizes[ind] = size(tensor, ind)
        end
    end
    return sizes
end

## `size_ind`
size_ind(tn, i) = size_ind(tn, i, DelegatorTrait(TensorNetwork(), tn))
size_ind(tn, i, ::DelegateToField) = size_ind(delegator(TensorNetwork(), tn), i)
function size_ind(tn, i, ::DontDelegate)
    fallback(size_ind)
    _tensors = tensors(tn; contain=i)
    @argcheck !isempty(_tensors) "Index $i not found in the Tensor Network"
    return size(first(_tensors), i)
end

## `tensor_at`
tensor_at(tn, tensor) = tensor_at(tn, tensor, DelegatorTrait(TensorNetwork(), tn))
tensor_at(tn, tensor, ::DelegateToField) = tensor_at(delegator(TensorNetwork(), tn), tensor)

## `ind_at`
ind_at(tn, index) = ind_at(tn, index, DelegatorTrait(TensorNetwork(), tn))
ind_at(tn, index, ::DelegateToField) = ind_at(delegator(TensorNetwork(), tn), index)
ind_at(tn, index, ::DontDelegate) = throw(MethodError(ind_at, (tn, index)))

# `addtensor!`
# TODO check that tensor is not already present
#   hastensor(tn, e.f) && throw(ArgumentError("tensor already present"))
addtensor!(tn, tensor) = addtensor!(tn, tensor, DelegatorTrait(TensorNetwork(), tn))
addtensor!(tn, tensor, ::DelegateToField) = addtensor!(delegator(TensorNetwork(), tn), tensor)
addtensor!(tn, tensor, ::DontDelegate) = throw(MethodError(addtensor!, (tn, tensor)))

## `rmtensor!`
# TODO check that tensor is present
#   hastensor(tn, e.f) || throw(ArgumentError("tensor not found"))
rmtensor!(tn, tensor) = rmtensor!(tn, tensor, DelegatorTrait(TensorNetwork(), tn))
rmtensor!(tn, tensor, ::DelegateToField) = rmtensor!(delegator(TensorNetwork(), tn), tensor)
rmtensor!(tn, tensor, ::DontDelegate) = throw(MethodError(rmtensor!, (tn, tensor)))

## `replace_tensor!`
# TODO check that `old` is present, `new` is not present and that the indices match
#    hastensor(tn, e.old) || throw(ArgumentError("old tensor not found"))
#    hastensor(tn, e.new) && throw(ArgumentError("new tensor already exists"))
#    !isscoped(tn) && @argcheck issetequal(inds(e.new), inds(e.old)) "replacing tensor indices don't match"
replace_tensor!(tn, old, new) = replace_tensor!(tn, old, new, DelegatorTrait(TensorNetwork(), tn))
replace_tensor!(tn, old, new, ::DelegateToField) = replace_tensor!(delegator(TensorNetwork(), tn), old, new)
replace_tensor!(tn, old, new, ::DontDelegate) = throw(MethodError(replace_tensor!, (tn, old, new)))

## `replace_ind!`
# TODO check that `old` is present, `new` is not present
#    hasind(tn, e.old) || throw(ArgumentError("old index not found"))
#    hasind(tn, e.new) && throw(ArgumentError("new index already exists"))
replace_ind!(tn, old, new) = replace_ind!(tn, old, new, DelegatorTrait(TensorNetwork(), tn))
replace_ind!(tn, old, new, ::DelegateToField) = replace_ind!(delegator(TensorNetwork(), tn), old, new)
replace_ind!(tn, old, new, ::DontDelegate) = throw(MethodError(replace_ind!, (tn, old, new)))

replace_ind!(tn, old_new) = replace_ind!(tn, old_new, DelegatorTrait(TensorNetwork(), tn))
replace_ind!(tn, old_new, ::DelegateToField) = replace_ind!(delegator(TensorNetwork(), tn), old_new)
replace_ind!(tn, old_new, ::DontDelegate) = throw(MethodError(replace_ind!, (tn, old_new)))

## `slice!`
# TODO check that `ind` is present
slice!(tn, ind, i) = slice!(tn, ind, i, DelegatorTrait(TensorNetwork(), tn))
slice!(tn, ind, i, ::DelegateToField) = slice!(delegator(TensorNetwork(), tn), ind, i)
function slice!(tn, ind, i, ::DontDelegate)
    hasind(tn, ind) || throw(ArgumentError("Index $ind not found in tensor network"))
    target_edge = edge_at(tn, ind)

    # update tensors
    for old_tensor in tensors(tn; contain=ind)
        new_tensor = selectdim(old_tensor, ind, i)
        replace_tensor!(tn, old_tensor, new_tensor)
    end

    # update network: if `i` is an integer, the index disappears and the edge is removed
    if i isa Integer
        rmedge!(tn.network, target_edge)
        delete!(tn.indmap, target_edge)
    end

    return tn
end

## `fuse!`
fuse!(tn, i) = fuse!(tn, i, DelegatorTrait(TensorNetwork(), tn))
fuse!(tn, i, ::DelegateToField) = fuse!(DelegatorTrait(TensorNetwork(), tn), i)

# TODO replace ind for `Index(Fused(parinds))`?
# TODO should this be run on the lowest or the highest level of the delegation hierarchy?
function fuse!(tn, i, ::DontDelegate)
    fallback(fuse!)
    @argcheck hasind(tn, i) "Index $i not found in the Tensor Network"

    parinds = inds(tn; parallelto=i)
    length(parinds) == 0 && return tn

    parinds = (i,) ∪ parinds
    @unsafe_region tn for tensor in tensors(tn; intersect=parinds)
        # TODO maybe refactor this when we stop using `Tensors` as graph vertices?
        replace_tensor!(tn, tensor, Muscle.fuse(tensor, parinds))
    end
    return tn
end
