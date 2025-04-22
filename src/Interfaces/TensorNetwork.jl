using Base: AbstractVecOrTuple
using ArgCheck
using ValSplit
using QuantumTags
using Graphs: Graphs
using EinExprs: EinExprs

abstract type AbstractTensorNetwork end

# interface object
struct TensorNetwork <: Interface end

# NOTE do not name it `copy` because it can break calls to `Base.copy`
# function copy_tn end

# query methods
## in reality, the only required methods are `all_*` and the mutating methods
function tensors end
function inds end

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

function size_inds end
function size_ind end

function tensors_with_inds end
function tensors_contain_inds end
function tensors_intersect_inds end

function inds_set end
function inds_parallel_to end

# mutating methods
function addtensor_inner! end
function rmtensor_inner! end
function replace_tensor_inner! end
function replace_ind_inner! end

function addtensor! end
function rmtensor! end
function replace_tensor! end
function replace_ind! end

# TODO contract!, split!

# implementation
## `tensors`
tensors(tn; kwargs...) = tensors(sort_nt(values(kwargs)), tn)
tensors(::@NamedTuple{}, tn) = all_tensors(tn) # tensors((;), tn, delegates(TensorNetwork(), tn))

# TODO fix grammar error on naming
tensors(kwargs::NamedTuple{(:contain,)}, tn) = tensors_contain_inds(tn, kwargs.contain)
tensors(kwargs::NamedTuple{(:intersect,)}, tn) = tensors_intersect_inds(tn, kwargs.intersect)
tensors(kwargs::NamedTuple{(:withinds,)}, tn) = tensors_with_inds(tn, kwargs.withinds)

@deprecate tensors(kwargs::NamedTuple{(:contains,)}, tn) tensors(; contain=kwargs.contains, tn)
@deprecate tensors(kwargs::NamedTuple{(:intersects,)}, tn) tensors(; intersect=kwargs.intersects, tn)

### singular version of `tensors`
tensor(tn; kwargs...) = tensor(sort_nt(values(kwargs)), tn)
tensor(kwargs::NamedTuple, tn) = only(tensors(kwargs, tn))

## `inds`
inds(tn; kwargs...) = inds(sort_nt(values(kwargs)), tn)
inds(::@NamedTuple{}, tn) = all_inds(tn) # inds((;), tn, delegates(TensorNetwork(), tn))
inds(kwargs::@NamedTuple{set::Symbol}, tn) = inds_set(tn, kwargs.set)
inds(kwargs::NamedTuple{(:parallel_to,)}, tn) = inds_parallel_to(tn, kwargs.parallel_to)
inds(kwargs::NamedTuple{(:parallelto,)}, tn) = inds_parallel_to(tn, kwargs.parallelto)

ind(tn; kwargs...) = ind(sort_nt(values(kwargs)), tn)
ind(kwargs::NamedTuple, tn) = only(inds(kwargs, tn))

## `all_tensors`
all_tensors(tn) = all_tensors(tn, delegates(TensorNetwork(), tn))
all_tensors(tn, ::DelegateTo) = all_tensors(delegate(TensorNetwork(), tn))
all_tensors(tn, ::DontDelegate) = throw(MethodError(all_tensors, (tn,)))

## `all_inds`
all_inds(tn) = all_inds(tn, delegates(TensorNetwork(), tn))
all_inds(tn, ::DelegateTo) = all_tensors(delegate(TensorNetwork(), tn))
function all_inds(tn, ::DontDelegate)
    @debug "Falling back to default `all_inds` method"
    mapreduce(inds, ∪, tensors(tn); init=Index[])
end

## `hastensor`
hastensor(tn, tensor) = hastensor(tn, tensor, delegates(TensorNetwork(), tn))
hastensor(tn, tensor, ::DelegateTo) = hastensor(delegate(TensorNetwork(), tn), tensor)
function hastensor(tn, tensor, ::DontDelegate)
    @debug "Falling back to default `hastensor` method"
    any(Base.Fix1(===, tensor), all_tensors(tn))
end

## `hasind`
hasind(tn, i) = hasind(tn, i, delegates(TensorNetwork(), tn))
hasind(tn, i, ::DelegateTo) = hasind(delegate(TensorNetwork(), tn), i)
function hasind(tn, i, _)
    @debug "Falling back to default `hasind` method"
    i ∈ all_inds(tn)
end

## `ntensors`
ntensors(tn; kwargs...) = ntensors(sort_nt(values(kwargs)), tn)

function ntensors(kwargs::NamedTuple, tn)
    @debug "Falling back to default `ntensors` method"
    length(tensors(kwargs, tn))
end

### dispatch due to performance reasons: see implementation in src/GenericTensorNetwork.jl
ntensors(::@NamedTuple{}, tn) = ntensors((;), tn, delegates(TensorNetwork(), tn))
ntensors(::@NamedTuple{}, tn, ::DelegateTo) = ntensors(delegate(TensorNetwork(), tn))
function ntensors(::@NamedTuple{}, tn, ::DontDelegate)
    @debug "Falling back to default `ntensors` method"
    length(all_tensors(tn))
end

## `ninds`
ninds(tn; kwargs...) = ninds(sort_nt(values(kwargs)), tn)

function ninds(kwargs::NamedTuple, tn)
    @debug "Falling back to default `ninds` method"
    length(inds(kwargs, tn))
end

### dispatch due to performance reasons: see implementation in src/GenericTensorNetwork.jl
ninds(::@NamedTuple{}, tn) = ninds((;), tn, delegates(TensorNetwork(), tn))
ninds(::@NamedTuple{}, tn, ::DelegateTo) = ninds((;), delegate(TensorNetwork(), tn))
function ninds(::@NamedTuple{}, tn, ::DontDelegate)
    @debug "Falling back to default `ninds` method"
    length(all_inds(tn))
end

## `tensors_with_inds`
function tensors_with_inds(tn, withinds::T) where {T<:AbstractVecOrTuple{<:Index}}
    filter(t -> issetequal(inds(t), withinds), tensors(tn; contain=withinds))
end

## `tensors_contain_inds`
tensors_contain_inds(tn, target) = tensors_contain_inds(tn, target, delegates(TensorNetwork(), tn))
tensors_contain_inds(tn, target, ::DelegateTo) = tensors_contain_inds(delegate(TensorNetwork(), tn), target)
tensors_contain_inds(tn, target, ::DontDelegate) = filter(⊇(target) ∘ inds, tensors(tn))
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
inds_set_open(tn) = inds_set_open(tn, delegates(TensorNetwork(), tn))::Vector{<:Index}
inds_set_open(tn, ::DelegateTo) = inds_set_open(delegate(TensorNetwork(), tn))
function inds_set_open(tn, ::DontDelegate)
    @debug "Falling back to default `inds_set_open` method"
    selected = Index[]
    histogram = hist(Iterators.flatten(Iterators.map(inds, tensors(tn))); init=Dict{Index,Int}())
    append!(selected, Iterators.map(first, Iterators.filter(((k, c),) -> c == 1, histogram)))
    return selected
end

inds_set(tn, ::Val{:inner}) = inds_set_inner(tn)
inds_set_inner(tn) = inds_set_inner(tn, delegates(TensorNetwork(), tn))::Vector{<:Index}
inds_set_inner(tn, ::DelegateTo) = inds_set_inner(delegate(TensorNetwork(), tn))
function inds_set_inner(tn, ::DontDelegate)
    @debug "Falling back to default `inds_set_inner` method"
    selected = Index[]
    histogram = hist(Iterators.flatten(Iterators.map(inds, tensors(tn))); init=Dict{Index,Int}())
    append!(selected, first.(Iterators.filter(((k, c),) -> c == 2, histogram)))
    return selected
end

inds_set(tn, ::Val{:hyper}) = inds_set_hyper(tn)
inds_set_hyper(tn) = inds_set_hyper(tn, delegates(TensorNetwork(), tn))::Vector{<:Index}
inds_set_hyper(tn, ::DelegateTo) = inds_set_hyper(delegate(TensorNetwork(), tn))
function inds_set_hyper(tn, ::DontDelegate)
    @debug "Falling back to default `inds_set_hyper` method"
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
size_inds(tn) = size_inds(tn, delegates(TensorNetwork(), tn))
size_inds(tn, ::DelegateTo) = size_inds(delegate(TensorNetwork(), tn))
function size_inds(tn, ::DontDelegate)
    @debug "Falling back to default `size_inds` method"
    sizes = Dict{Index,Int}()
    for tensor in tensors(tn)
        for ind in inds(tensor)
            sizes[ind] = size(tensor, ind)
        end
    end
    return sizes
end

## `size_ind`
size_ind(tn, i) = size_ind(tn, i, delegates(TensorNetwork(), tn))
size_ind(tn, i, ::DelegateTo) = size_ind(delegate(TensorNetwork(), tn), i)
function size_ind(tn, i, ::DontDelegate)
    @debug "Falling back to default `size_ind` method"
    _tensors = tensors(tn; contain=i)
    @argcheck !isempty(_tensors) "Index $i not found in the Tensor Network"
    return size(first(_tensors), i)
end

# mutating methods
addtensor_inner!(tn, tensor) = addtensor_inner!(tn, tensor, delegates(TensorNetwork(), tn))
addtensor_inner!(tn, tensor, ::DelegateTo) = addtensor!(delegate(TensorNetwork(), tn), tensor)
addtensor_inner!(tn, tensor, ::DontDelegate) = throw(MethodError(addtensor_inner!, (tn, tensor)))

rmtensor_inner!(tn, tensor) = rmtensor_inner!(tn, tensor, delegates(TensorNetwork(), tn))
rmtensor_inner!(tn, tensor, ::DelegateTo) = rmtensor!(delegate(TensorNetwork(), tn), tensor)
rmtensor_inner!(tn, tensor, ::DontDelegate) = throw(MethodError(rmtensor_inner!, (tn, tensor)))

function replace_tensor_inner!(tn, old_tensor, new_tensor)
    replace_tensor_inner!(tn, old_tensor, new_tensor, delegates(TensorNetwork(), tn))
end
function replace_tensor_inner!(tn, old_tensor, new_tensor, ::DelegateTo)
    replace_tensor!(delegate(TensorNetwork(), tn), old_tensor, new_tensor)
end
function replace_tensor_inner!(tn, old_tensor, new_tensor, ::DontDelegate)
    @debug "Falling back to the default `replace_tensor_inner!` method"

    old_tensor === new_tensor && return tn
    hastensor(tn, old_tensor) || throw(ArgumentError("old tensor not found"))
    hastensor(tn, new_tensor) && throw(ArgumentError("new tensor already exists"))

    if !isscoped(tn)
        @argcheck issetequal(inds(new_tensor), inds(old_tensor)) "replacing tensor indices don't match"
    end

    rmtensor!(tn, old_tensor)
    addtensor!(tn, new_tensor)
end

replace_ind_inner!(tn, old_ind, new_ind) = replace_ind_inner!(tn, old_ind, new_ind, delegates(TensorNetwork(), tn))
function replace_ind_inner!(tn, old_ind, new_ind, ::DelegateTo)
    replace_ind!(delegate(TensorNetwork(), tn), old_ind, new_ind)
end
function replace_ind_inner!(tn, old_ind, new_ind, ::DontDelegate)
    @debug "Falling back to the default `replace_ind_inner!` method"

    @argcheck hasind(tn, old_ind) "index $old_ind does not exist"
    old_ind == new_ind && return tn
    @argcheck !hasind(tn, new_ind) "index $new_ind is already present"

    ############## legacy comment which might not be true anymore ##############
    # NOTE `copy` because collection underneath is mutated
    # for old_tensor in copy(tensors(tn; contain=old_ind))
    # ...
    # NOTE do not `delete!` before `push!` as indices can be lost due to `tryprune!`
    # tryprune!(tn, old_ind) => `tryprune!` should be called on `handle!`
    ############################################################################

    @unsafe_region tn for old_tensor in tensors_contain_inds(tn, old_ind)
        new_tensor = replace(old_tensor, old_ind => new_ind)
        replace_tensor!(tn, old_tensor, new_tensor)
    end
end

## `addtensor!`
function addtensor!(tn, tensor)
    checkeffect(tn, PushEffect(tensor))
    addtensor_inner!(tn, tensor)
    handle!(tn, PushEffect(tensor))
    return tn
end

checkeffect(tn, @nospecialize(e::PushEffect{<:Tensor})) = checkeffect(tn, e, delegates(TensorNetwork(), tn))
checkeffect(tn, @nospecialize(e::PushEffect{<:Tensor}), ::DelegateTo) = checkeffect(delegate(TensorNetwork(), tn), e)
function checkeffect(tn, @nospecialize(e::PushEffect{T}), ::DontDelegate) where {T<:Tensor}
    # TODO throw a custom EffectError
    hastensor(tn, e.f) && throw(ArgumentError("tensor already present"))
end

handle!(tn, @nospecialize(e::PushEffect{<:Tensor})) = handle!(tn, e, delegates(TensorNetwork(), tn))
handle!(tn, @nospecialize(e::PushEffect{<:Tensor}), ::DelegateTo) = handle!(delegate(TensorNetwork(), tn), e)
function handle!(tn, @nospecialize(e::PushEffect{T}), ::DontDelegate) where {T<:Tensor}
    throw(MissingEffectHandlerException(tn, e))
end

## `rmtensor!`
function rmtensor!(tn, tensor)
    checkeffect(tn, DeleteEffect(tensor))
    rmtensor_inner!(tn, tensor)
    handle!(tn, DeleteEffect(tensor))
    return tn
end

checkeffect(tn, @nospecialize(e::DeleteEffect{<:Tensor})) = checkeffect(tn, e, delegates(TensorNetwork(), tn))
checkeffect(tn, @nospecialize(e::DeleteEffect{<:Tensor}), ::DelegateTo) = checkeffect(delegate(TensorNetwork(), tn), e)
function checkeffect(tn, @nospecialize(e::DeleteEffect{T}), ::DontDelegate) where {T<:Tensor}
    hastensor(tn, e.f) || throw(ArgumentError("tensor not found"))
end

handle!(tn, @nospecialize(e::DeleteEffect{<:Tensor})) = handle!(tn, e, delegates(TensorNetwork(), tn))
handle!(tn, @nospecialize(e::DeleteEffect{<:Tensor}), ::DelegateTo) = handle!(delegate(TensorNetwork(), tn), e)
function handle!(tn, @nospecialize(e::DeleteEffect{T}), ::DontDelegate) where {T<:Tensor}
    throw(MissingEffectHandlerException(tn, e))
end

## `replace_tensor!`
function replace_tensor!(tn, old_tensor, new_tensor)
    checkeffect(tn, ReplaceEffect(old_tensor, new_tensor))
    replace_tensor_inner!(tn, old_tensor, new_tensor)
    handle!(tn, ReplaceEffect(old_tensor, new_tensor))
    return tn
end

checkeffect(tn, @nospecialize(e::ReplaceEffect{<:Tensor,<:Tensor})) = checkeffect(tn, e, delegates(TensorNetwork(), tn))
function checkeffect(tn, @nospecialize(e::ReplaceEffect{<:Tensor,<:Tensor}), ::DelegateTo)
    checkeffect(delegate(TensorNetwork(), tn), e)
end
function checkeffect(tn, @nospecialize(e::ReplaceEffect{Told,Tnew}), ::DontDelegate) where {Told<:Tensor,Tnew<:Tensor}
    hastensor(tn, e.old) || throw(ArgumentError("old tensor not found"))
    hastensor(tn, e.new) && throw(ArgumentError("new tensor already exists"))

    if !isscoped(tn)
        @argcheck issetequal(inds(e.new), inds(e.old)) "replacing tensor indices don't match"
    end
end

handle!(tn, @nospecialize(e::ReplaceEffect{<:Tensor,<:Tensor})) = handle!(tn, e, delegates(TensorNetwork(), tn))
function handle!(tn, @nospecialize(e::ReplaceEffect{<:Tensor,<:Tensor}), ::DelegateTo)
    handle!(delegate(TensorNetwork(), tn), e)
end
function handle!(_, @nospecialize(e::ReplaceEffect{Told,Tnew}), ::DontDelegate) where {Told<:Tensor,Tnew<:Tensor}
    throw(MissingEffectHandlerException(tn, e))
end

## `replace_ind!`
function replace_ind!(tn, old_ind, new_ind)
    checkeffect(tn, ReplaceEffect(old_ind, new_ind))
    replace_ind_inner!(tn, old_ind, new_ind)
    handle!(tn, ReplaceEffect(old_ind, new_ind))
    return tn
end

checkeffect(tn, @nospecialize(e::ReplaceEffect{<:Index,<:Index})) = checkeffect(tn, e, delegates(TensorNetwork(), tn))
function checkeffect(tn, @nospecialize(e::ReplaceEffect{<:Index,<:Index}), ::DelegateTo)
    checkeffect(delegate(TensorNetwork(), tn), e)
end
function checkeffect(tn, @nospecialize(e::ReplaceEffect{Iold,Inew}), ::DontDelegate) where {Iold<:Index,Inew<:Index}
    hasind(tn, e.old) || throw(ArgumentError("old index not found"))
    hasind(tn, e.new) && throw(ArgumentError("new index already exists"))
end

handle!(tn, @nospecialize(e::ReplaceEffect{<:Index,<:Index})) = handle!(tn, e, delegates(TensorNetwork(), tn))
handle!(tn, @nospecialize(e::ReplaceEffect{<:Index,<:Index}), ::DelegateTo) = handle!(delegate(TensorNetwork(), tn), e)
function handle!(_, @nospecialize(e::ReplaceEffect{Iold,Inew}), ::DontDelegate) where {Iold<:Index,Inew<:Index}
    throw(MissingEffectHandlerException(tn, e))
end

# derived methods
# TODO Base.copy ==> copy_tn

Base.summary(io::IO, tn::T) where {T<:AbstractTensorNetwork} = print(io, "$(ntensors(tn))-tensors $T")

function Base.show(io::IO, tn::T) where {T<:AbstractTensorNetwork}
    return print(io, "$T (#tensors=$(ntensors(tn)), #inds=$(ninds(tn)))")
end

Base.in(i::Index, tn::AbstractTensorNetwork) = hasind(tn, i)
Base.in(tensor::Tensor, tn::AbstractTensorNetwork) = hastensor(tn, tensor)

Base.size(tn::AbstractTensorNetwork) = size_inds(tn)
Base.size(tn::AbstractTensorNetwork, i::Index) = size_ind(tn, i)

Base.eltype(tn::AbstractTensorNetwork) = promote_type(eltype.(tensors(tn))...)

"""
    Base.collect(tn::AbstractTensorNetwork)

Return a list of the [`Tensor`](@ref)s in the Tensor Network. It is equivalent to `tensors(tn)`.
"""
Base.collect(tn::AbstractTensorNetwork) = tensors(tn)

"""
    Base.similar(tn::AbstractTensorNetwork)

Return a copy of the `TensorNetwork` with all [`Tensor`](@ref)s replaced by their `similar` version.
"""
function Base.similar(tn::AbstractTensorNetwork)
    tn = copy(tn)
    replace!(tn, tensors(tn) .=> similar.(tensors(tn)))
    return tn
end

"""
    Base.zero(tn::AbstractTensorNetwork)

Return a copy of the `TensorNetwork` with all [`Tensor`](@ref)s replaced by their `zero` version.
"""
function Base.zero(tn::AbstractTensorNetwork)
    tn = copy(tn)
    replace!(tn, tensors(tn) .=> zero.(tensors(tn)))
    return tn
end

"""
    conj(tn::AbstractTensorNetwork)

Return a copy of the [`AbstractTensorNetwork`](@ref) with all tensors conjugated.

See also: [`conj!`](@ref).
"""
function Base.conj(tn::AbstractTensorNetwork)
    tn = copy(tn)
    # WARN do not call `conj!(tn)` because it will mutate the arrays of the original `tn` too!
    replace!(tn, tensors(tn) .=> conj.(tensors(tn)))
    return tn
end

"""
    conj!(tn::AbstractTensorNetwork)

Conjugate all tensors in the [`AbstractTensorNetwork`](@ref) in-place.

See also: [`conj`](@ref).
"""
function Base.conj!(tn::AbstractTensorNetwork)
    foreach(conj!, tensors(tn))
    return tn
end

"""
    selectdim(tn, index::Symbol, i)

Return a copy of the Tensor Network where `index` has been projected to dimension `i`.

See also: [`view`](@ref), [`slice!`](@ref).
"""
Base.selectdim(tn::AbstractTensorNetwork, index::Index, i) = @view tn[index => i]

"""
    view(tn, index => i...)

Return a copy of the Tensor Network where each `index` has been projected to dimension `i`.
It is equivalent to a recursive call of [`selectdim`](@ref).

See also: [`selectdim`](@ref), [`slice!`](@ref).
"""
function Base.view(tn::AbstractTensorNetwork, slices::Pair{I}...) where {I<:Index}
    tn = copy(tn)

    for (label, i) in slices
        slice!(tn, label, i)
    end

    return tn
end

"""
    Graphs.neighbors(tn::AbstractTensorNetwork, tensor; open=true)

Return the neighboring [`Tensor`](@ref)s of `tensor` in the Tensor Network.
If `open=true`, the `tensor` itself is not included in the result.
"""
function Graphs.neighbors(tn::AbstractTensorNetwork, tensor::Tensor; open::Bool=true)
    @argcheck hastensor(tn, tensor) "Tensor not found in TensorNetwork"
    neigh_tensors = mapreduce(∪, inds(tensor)) do index
        tensors(tn; intersects=index)
    end
    open && filter!(x -> x !== tensor, neigh_tensors)
    return neigh_tensors
end

"""
    Graphs.neighbors(tn::AbstractTensorNetwork, ind; open=true)

Return the neighboring indices of `ind` in the Tensor Network.
If `open=true`, the `ind` itself is not included in the result.
"""
function Graphs.neighbors(tn::AbstractTensorNetwork, i::Index; open::Bool=true)
    @argcheck i ∈ tn "Index $i not found in TensorNetwork"
    neigh_inds = mapreduce(inds, ∪, tensors(tn; intersects=i))
    open && filter(x -> x !== i, neigh_inds)
    return neigh_inds
end

"""
    push!(tn::AbstractTensorNetwork, tensor)

Add a [`Tensor`](@ref) to the Tensor Network.
"""
Base.push!(tn::AbstractTensorNetwork, tensor::Tensor; kwargs...) = addtensor!(tn, tensor; kwargs...)

"""
    append!(tn::AbstractTensorNetwork, tensors)
    append!(tn::AbstractTensorNetwork, other::AbstractTensorNetwork)

Add a tensors to a Tensor Network from a list of [`Tensor`](@ref)s or from another Tensor Network.

See also: [`push!`](@ref).
"""
Base.append!(tn::AbstractTensorNetwork, tensors) = (foreach(Base.Fix1(push!, tn), tensors); tn)

# TODO how do we deal with the tags from the other Tensor Network?
# function Base.append!(tn::AbstractTensorNetwork, other::AbstractTensorNetwork)
#     (foreach(Base.Fix1(push!, tn), tensors(other)); tn)
# end

"""
    pop!(tn::AbstractTensorNetwork, tensor::Tensor)
    pop!(tn::AbstractTensorNetwork, i::Union{Symbol,AbstractVecOrTuple{Symbol}})

Remove and return the first tensor in `tn`` that satisfies _egality_ (i.e. `≡`or`===`) with `tensor`.

See also: [`push!`](@ref), [`delete!`](@ref).
"""
Base.pop!(tn::AbstractTensorNetwork, tensor::Tensor) = (delete!(tn, tensor); tensor)

"""
    delete!(tn::AbstractTensorNetwork, tensor)

Remove a [`Tensor`](@ref) from the Tensor Network.

!!! warning

    [`Tensor`](@ref)s are identified in a Tensor Network by their `objectid`, so you must pass the same object and not a copy.
"""
Base.delete!(tn::AbstractTensorNetwork, tensor::Tensor) = rmtensor!(tn, tensor)

"""
    replace!(tn::AbstractTensorNetwork, old => new...)
    replace(tn::AbstractTensorNetwork, old => new...)

Replace the element in `old` with the one in `new`. Depending on the types of `old` and `new`, the following behaviour is expected:

  - If `Symbol`s, it will correspond to a index renaming.
  - If `Tensor`s, first element that satisfies _egality_ (`≡` or `===`) will be replaced.
"""
Base.replace!(::AbstractTensorNetwork, ::Any...)

# rename index
function Base.replace!(tn::AbstractTensorNetwork, old_new::Pair{Ia,Ib}) where {Ia<:Index,Ib<:Index}
    replace_ind!(tn, old_new.first, old_new.second)
    return tn
end

# replace tensor
function Base.replace!(tn::AbstractTensorNetwork, old_new::Pair{<:Tensor,<:Tensor})
    replace_tensor!(tn, old_new.first, old_new.second)
    return tn
end

# rename a collection of indices
function Base.replace!(
    tn::AbstractTensorNetwork, old_new::Base.AbstractVecOrTuple{Pair{Ia,Ib}}
) where {Ia<:Index,Ib<:Index}
    replace_inds!(tn, old_new)
    return tn
end

function replace_inds!(tn, old_new)
    from, to = first.(old_new), last.(old_new)
    allinds = inds(tn)

    # condition: from ⊆ allinds
    @argcheck from ⊆ allinds "set of old indices must be a subset of current indices"

    # condition: from \ to ∩ allinds = ∅
    @argcheck isdisjoint(setdiff(to, from), allinds) """
        new indices must be either a element of the old indices or not an element of the TensorNetwork's indices
        """

    overlap = from ∩ to
    if isempty(overlap)
        # no overlap so easy replacement
        for (f, t) in zip(from, to)
            replace!(tn, f => t)
        end
    else
        # overlap between old and new indices => need a temporary name `replace!`
        tmp = Dict([i => gensym(i) for i in from])

        # replace old indices with temporary names
        # TODO maybe do replacement manually and call `handle!` once in the end?
        replace!(tn, tmp)

        # replace temporary names with new indices
        replace!(tn, [tmp[f] => t for (f, t) in zip(from, to)])
    end

    # return the final index mapping
    return tn
end

# replace tensor with a TensorNetwork
function Base.replace!(tn::AbstractTensorNetwork, old_new::Pair{<:Tensor,<:AbstractTensorNetwork})
    checkeffect(tn, ReplaceEffect(old_new))

    old, new = old_new
    @argcheck issetequal(inds(new; set=:open), inds(old)) "indices don't match"
    @argcheck isdisjoint(inds(new; set=:inner), inds(tn)) "overlapping inner indices"

    # manually perform `append!(tn, new)` to avoid calling `handle!` several times
    for tensor in tensors(new)
        addtensor_inner!(tn, tensor)
    end
    rmtensor_inner!(tn, old)
    handle!(tn, ReplaceEffect(old_new))

    return tn
end

function Base.replace!(tn::AbstractTensorNetwork, @nospecialize(old_new::Pair{<:Tensor,<:Vector{<:Tensor}}))
    replace!(tn, old_new.first => TensorNetwork(old_new.second))
end

# replace collection of tensors with a tensor (called on `contract!`)
function Base.replace!(tn::AbstractTensorNetwork, @nospecialize(old_new::Pair{<:Vector{<:Tensor},<:Tensor}))
    old, new = old_new

    checkeffect(tn, ReplaceEffect(old, new))
    @argcheck all(∈(tn), old)
    @argcheck new ∉ tn
    @argcheck inds(new) ⊆ collect(Iterators.flatmap(inds, old))
    # TODO check open and inner inds

    for tensor in old
        rmtensor_inner!(tn, tensor)
    end
    addtensor_inner!(tn, new)
    handle!(tn, ReplaceEffect(old, new))

    return tn
end

Base.replace!(tn::AbstractTensorNetwork) = tn
Base.replace!(tn::AbstractTensorNetwork, old_new::Pair) = throw(MethodError(replace!, (tn, old_new)))
@inline Base.replace!(tn::T, old_new::P...) where {T<:AbstractTensorNetwork,P<:Pair} = replace!(tn, old_new)
@inline Base.replace!(tn::AbstractTensorNetwork, old_new::Dict) = replace!(tn, collect(old_new))

function Base.replace!(tn::AbstractTensorNetwork, old_new::Base.AbstractVecOrTuple{Pair})
    for pair in old_new
        replace!(tn, pair)
    end
    return tn
end

function Base.rand(::Type{T}, args...; kwargs...) where {T<:AbstractTensorNetwork}
    return rand(Random.default_rng(), T, args...; kwargs...)
end

"""
    arrays(tn::AbstractTensorNetwork; kwargs...)

Return a list of the arrays of in the Tensor Network. It is equivalent to `parent.(tensors(tn; kwargs...))`.
"""
arrays(tn; kwargs...) = parent.(tensors(tn; kwargs...))

"""
    slice!(tn, index::Symbol, i)

In-place projection of `index` on dimension `i`.

See also: [`selectdim`](@ref), [`view`](@ref).
"""
function slice!(tn, ind, i)
    replacements = map(tensors(tn; contains=ind)) do tensor
        tensor => selectdim(tensor, ind, i)
    end

    @unsafe_region tn replace!(tn, replacements)

    return tn
end

"""
    einexpr(tn::AbstractTensorNetwork; optimizer = EinExprs.Greedy, output = inds(tn, :open), kwargs...)

Search a contraction path for the given [`AbstractTensorNetwork`](@ref) and return it as a `EinExpr`.

# Keyword Arguments

  - `optimizer` Contraction path optimizer. Check [`EinExprs`](https://github.com/bsc-quantic/EinExprs.jl) documentation for more info.
  - `outputs` Indices that won't be contracted. Defaults to open indices.
  - `kwargs` Options to be passed to the optimizer.

See also: [`contract`](@ref).
"""
function EinExprs.einexpr(
    tn::AbstractTensorNetwork; optimizer=EinExprs.Greedy(), output=inds(tn; set=:open), outputs=nothing, kwargs...
)
    if !isnothing(outputs)
        Base.depwarn("`outputs` keyword argument is deprecated, use output instead", :einexpr; force=true)
        output = outputs
    end

    #! format: off
    path = EinExprs.SizedEinExpr(
        EinExprs.EinExpr(
            output,
            EinExprs.EinExpr.(Iterators.map(inds, tensors(tn)))
        ),
        Dict(ind => size(tn, ind) for ind in inds(tn))
    )
    #! format: on

    # don't use `sum(::Vector{EinExpr})`: it's broken and takes x10 more time
    return einexpr(optimizer, path; kwargs...)
end

"""
    contract(tn; optimizer=Greedy(), path=einexpr(tn))

Contract a Tensor Network. If `path` is not specified, the contraction order will be computed by [`einexpr`](@ref).

See also: [`einexpr`](@ref), [`contract!`](@ref).
"""
function contract(tn; optimizer=EinExprs.Greedy(), path=EinExprs.einexpr(tn; optimizer))
    path::EinExprs.EinExpr = if path isa EinExprs.SizedEinExpr
        path.path
    else
        path
    end

    # copy `tn` and pop tensors to avoid conflicts between tensors with same indices
    tn = GenericTensorNetwork(tensors(tn))
    cache = IdDict{EinExprs.EinExpr,Tensor}()
    for leaf in leaves(path)
        selection = tensors(tn; withinds=head(leaf))
        if length(selection) > 1
            @warn "Found more than one tensor with index $(head(leaf))... Using first one"
        end
        selection = first(selection)
        cache[leaf] = selection
        delete!(tn, selection)
    end

    for intermediate in Branches(path)
        if EinExprs.nargs(intermediate) == 1
            a = only(args(intermediate))
            cache[intermediate] = contract(cache[a]; dims=EinExprs.suminds(intermediate))
            delete!(cache, a)
        elseif EinExprs.nargs(intermediate) == 2
            a, b = args(intermediate)
            cache[intermediate] = contract(cache[a], cache[b]; dims=EinExprs.suminds(intermediate))
            delete!(cache, a)
            delete!(cache, b)
        else
            # TODO we should fix this in EinExprs, this is a temporal fix meanwhile
            @warn "Found a contraction with $(EinExprs.nargs(intermediate)) arguments... Using reduction which might be sub-optimal"
            target_tensors = map(EinExprs.args(intermediate)) do branch
                pop!(cache, branch)
            end
            cache[intermediate] = foldl(target_tensors) do a, b
                contract(a, b; dims=EinExprs.suminds(intermediate))
            end
        end
    end
    return cache[path]
end

# TODO to add in the future
# """
#     gauge!(tn::AbstractTensorNetwork, ind, U[, Uinv])

# Perform a gauge transformation on index `ind`.
# """
# function gauge!(tn::AbstractTensorNetwork, ind::Symbol, U::AbstractMatrix, Uinv::AbstractMatrix=inv(U))
#     a, b = tensors(tn; contains=ind)
#     tmpind = gensym(ind)

#     tU = Tensor(U, [ind, tmpind])
#     tUinv = Tensor(Uinv, [tmpind, ind])

#     gauged_a = replace(contract(a, tU), tmpind => ind)
#     gauged_b = replace(contract(tUinv, b), tmpind => ind)

#     replace!(tn, [a => gauged_a, b => gauged_b])
# end

"""
    resetinds!(tn::AbstractTensorNetwork, method=:gensymnew; kwargs...)

Rename indices in the `TensorNetwork` to a new set of indices. It is mainly used to avoid index name conflicts when connecting Tensor Networks.
"""
function resetinds!(tn, method=:gensymclean; kwargs...)
    new_name_f = if method === :suffix
        (ind) -> Index(Symbol(ind, get(kwargs, :suffix, '\'')))
    elseif method === :gensymwrap
        (ind) -> Index(gensym(ind))
    elseif method === :gensymnew
        (_) -> Index(gensym(get(kwargs, :base, :i)))
    elseif method === :gensymclean
        (ind) -> Index(gensymclean(ind))
    elseif method === :characters
        gen = IndexCounter(get(kwargs, :init, 1))
        (_) -> Index(nextindex!(gen))
    else
        error("Invalid method: $(Meta.quot(method))")
    end

    _inds = if haskey(kwargs, :set)
        inds(tn; set=kwargs.set)
    else
        inds(tn)
    end

    for ind in _inds
        replace!(tn, ind => new_name_f(ind))
    end
end

"""
    fuse!(tn::AbstractTensorNetwork, i::Symbol)

Group indices parallel to `i` and reshape the tensors accordingly.
"""
function fuse!(tn, i)
    parinds = filter!(!=(i), inds(tn; parallelto=i))
    length(parinds) == 0 && return tn

    parinds = (i,) ∪ parinds
    newtensors = map(Base.Fix2(fuse, parinds), pop!(tn, parinds))

    append!(tn, newtensors)

    return tn
end
