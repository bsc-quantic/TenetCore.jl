using Base: AbstractVecOrTuple
using ArgCheck
using ValSplit
using QuantumTags
using EinExprs: EinExprs
using Muscle: Muscle

# interface object
"""
    TensorNetwork <: Interface

A singleton type that represents the basic interface of a Tensor Network.
"""
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

# effects
"""
    AddTensorEffect{F} <: Effect

Represents the effect of pushing an object.
"""
struct AddTensorEffect{F} <: Effect
    f::F
end

AddTensorEffect(@nospecialize(f::Tensor)) = AddTensorEffect{Tensor}(f)

"""
    RemoveTensorEffect{F} <: Effect

Represents the effect of deleting an object.
"""
struct RemoveTensorEffect{F} <: Effect
    f::F
end

RemoveTensorEffect(@nospecialize(f::Tensor)) = RemoveTensorEffect{Tensor}(f)

# TODO split into `ReplaceTensorEffect`, `ReplaceIndexEffect`, ...
"""
    ReplaceEffect{F} <: Effect

Represents the effect of replacing an object with a new one.
"""
struct ReplaceEffect{O,N} <: Effect
    old::O
    new::N
end

ReplaceEffect(old::Tensor, new::Tensor) = ReplaceEffect{Tensor,Tensor}(old, new)
ReplaceEffect(old::Tensor, new::TN) where {TN} = ReplaceEffect{Tensor,TN}(old, new)
ReplaceEffect(f::Pair) = ReplaceEffect(f.first, f.second)

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

## `inds`
inds(tn; kwargs...) = inds(sort_nt(values(kwargs)), tn)
inds(::@NamedTuple{}, tn) = all_inds(tn) # inds((;), tn, DelegatorTrait(TensorNetwork(), tn))
inds(kwargs::@NamedTuple{set::Symbol}, tn) = inds_set(tn, kwargs.set)
inds(kwargs::NamedTuple{(:parallel_to,)}, tn) = inds_parallel_to(tn, kwargs.parallel_to)
inds(kwargs::NamedTuple{(:parallelto,)}, tn) = inds_parallel_to(tn, kwargs.parallelto)

ind(tn; kwargs...) = ind(sort_nt(values(kwargs)), tn)
ind(kwargs::NamedTuple, tn) = only(inds(kwargs, tn))

## `all_tensors`
all_tensors(tn) = all_tensors(tn, DelegatorTrait(TensorNetwork(), tn))
all_tensors(tn, ::DelegateTo) = all_tensors(delegator(TensorNetwork(), tn))
all_tensors(tn, ::DontDelegate) = throw(MethodError(all_tensors, (tn,)))

## `all_inds`
all_inds(tn) = all_inds(tn, DelegatorTrait(TensorNetwork(), tn))
all_inds(tn, ::DelegateTo) = all_inds(delegator(TensorNetwork(), tn))
function all_inds(tn, ::DontDelegate)
    @debug "Falling back to default `all_inds` method"
    mapreduce(inds, ∪, tensors(tn); init=Index[])
end

## `all_tensors_iter`
all_tensors_iter(tn) = all_tensors_iter(tn, DelegatorTrait(TensorNetwork(), tn))
all_tensors_iter(tn, ::DelegateTo) = all_tensors_iter(delegator(TensorNetwork(), tn))
function all_tensors_iter(tn, ::DontDelegate)
    @debug "Falling back to default `all_tensors_iter` method"
    all_tensors(tn)
end

## `all_inds_iter`
all_inds_iter(tn) = all_inds_iter(tn, DelegatorTrait(TensorNetwork(), tn))
all_inds_iter(tn, ::DelegateTo) = all_inds_iter(delegator(TensorNetwork(), tn))
function all_inds_iter(tn, ::DontDelegate)
    @debug "Falling back to default `all_inds_iter` method"
    all_inds(tn)
end

## `hastensor`
hastensor(tn, tensor) = hastensor(tn, tensor, DelegatorTrait(TensorNetwork(), tn))
hastensor(tn, tensor, ::DelegateTo) = hastensor(delegator(TensorNetwork(), tn), tensor)
function hastensor(tn, tensor, ::DontDelegate)
    @debug "Falling back to default `hastensor` method"
    any(Base.Fix1(===, tensor), all_tensors(tn))
end

## `hasind`
hasind(tn, i) = hasind(tn, i, DelegatorTrait(TensorNetwork(), tn))
hasind(tn, i, ::DelegateTo) = hasind(delegator(TensorNetwork(), tn), i)
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
ntensors(::@NamedTuple{}, tn) = ntensors((;), tn, DelegatorTrait(TensorNetwork(), tn))
ntensors(::@NamedTuple{}, tn, ::DelegateTo) = ntensors(delegator(TensorNetwork(), tn))
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
ninds(::@NamedTuple{}, tn) = ninds((;), tn, DelegatorTrait(TensorNetwork(), tn))
ninds(::@NamedTuple{}, tn, ::DelegateTo) = ninds((;), delegator(TensorNetwork(), tn))
function ninds(::@NamedTuple{}, tn, ::DontDelegate)
    @debug "Falling back to default `ninds` method"
    length(all_inds(tn))
end

## `tensors_with_inds`
function tensors_with_inds(tn, withinds::T) where {T<:AbstractVecOrTuple{<:Index}}
    filter(t -> issetequal(inds(t), withinds), tensors(tn; contain=withinds))
end

## `tensors_contain_inds`
tensors_contain_inds(tn, target) = tensors_contain_inds(tn, target, DelegatorTrait(TensorNetwork(), tn))
tensors_contain_inds(tn, target, ::DelegateTo) = tensors_contain_inds(delegator(TensorNetwork(), tn), target)
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
inds_set_open(tn, ::DelegateTo) = inds_set_open(delegator(TensorNetwork(), tn))
function inds_set_open(tn, ::DontDelegate)
    @debug "Falling back to default `inds_set_open` method"
    selected = Index[]
    histogram = hist(Iterators.flatten(Iterators.map(inds, tensors(tn))); init=Dict{Index,Int}())
    append!(selected, Iterators.map(first, Iterators.filter(((k, c),) -> c == 1, histogram)))
    return selected
end

inds_set(tn, ::Val{:inner}) = inds_set_inner(tn)
inds_set_inner(tn) = inds_set_inner(tn, DelegatorTrait(TensorNetwork(), tn))::Vector{<:Index}
inds_set_inner(tn, ::DelegateTo) = inds_set_inner(delegator(TensorNetwork(), tn))
function inds_set_inner(tn, ::DontDelegate)
    @debug "Falling back to default `inds_set_inner` method"
    selected = Index[]
    histogram = hist(Iterators.flatten(Iterators.map(inds, tensors(tn))); init=Dict{Index,Int}())
    append!(selected, first.(Iterators.filter(((k, c),) -> c == 2, histogram)))
    return selected
end

inds_set(tn, ::Val{:hyper}) = inds_set_hyper(tn)
inds_set_hyper(tn) = inds_set_hyper(tn, DelegatorTrait(TensorNetwork(), tn))::Vector{<:Index}
inds_set_hyper(tn, ::DelegateTo) = inds_set_hyper(delegator(TensorNetwork(), tn))
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
size_inds(tn) = size_inds(tn, DelegatorTrait(TensorNetwork(), tn))
size_inds(tn, ::DelegateTo) = size_inds(delegator(TensorNetwork(), tn))
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
size_ind(tn, i) = size_ind(tn, i, DelegatorTrait(TensorNetwork(), tn))
size_ind(tn, i, ::DelegateTo) = size_ind(delegator(TensorNetwork(), tn), i)
function size_ind(tn, i, ::DontDelegate)
    @debug "Falling back to default `size_ind` method"
    _tensors = tensors(tn; contain=i)
    @argcheck !isempty(_tensors) "Index $i not found in the Tensor Network"
    return size(first(_tensors), i)
end

# mutating methods
addtensor_inner!(tn, tensor) = addtensor_inner!(tn, tensor, DelegatorTrait(TensorNetwork(), tn))
addtensor_inner!(tn, tensor, ::DelegateTo) = addtensor!(delegator(TensorNetwork(), tn), tensor)
addtensor_inner!(tn, tensor, ::DontDelegate) = throw(MethodError(addtensor_inner!, (tn, tensor)))

rmtensor_inner!(tn, tensor) = rmtensor_inner!(tn, tensor, DelegatorTrait(TensorNetwork(), tn))
rmtensor_inner!(tn, tensor, ::DelegateTo) = rmtensor!(delegator(TensorNetwork(), tn), tensor)
rmtensor_inner!(tn, tensor, ::DontDelegate) = throw(MethodError(rmtensor_inner!, (tn, tensor)))

function replace_tensor_inner!(tn, old_tensor, new_tensor)
    replace_tensor_inner!(tn, old_tensor, new_tensor, DelegatorTrait(TensorNetwork(), tn))
end
function replace_tensor_inner!(tn, old_tensor, new_tensor, ::DelegateTo)
    replace_tensor!(delegator(TensorNetwork(), tn), old_tensor, new_tensor)
end
function replace_tensor_inner!(tn, old_tensor, new_tensor, ::DontDelegate)
    @debug "Falling back to the default `replace_tensor_inner!` method"

    old_tensor === new_tensor && return tn
    hastensor(tn, old_tensor) || throw(ArgumentError("old tensor not found"))
    hastensor(tn, new_tensor) && throw(ArgumentError("new tensor already exists"))

    if !isscoped(tn)
        @argcheck issetequal(inds(new_tensor), inds(old_tensor)) "replacing tensor indices don't match"
    end

    # TODO shouldn't we call `*_inner!` methods instead? their effects might have side effects
    rmtensor!(tn, old_tensor)
    addtensor!(tn, new_tensor)
end

replace_ind_inner!(tn, old_ind, new_ind) = replace_ind_inner!(tn, old_ind, new_ind, DelegatorTrait(TensorNetwork(), tn))
function replace_ind_inner!(tn, old_ind, new_ind, ::DelegateTo)
    replace_ind!(delegator(TensorNetwork(), tn), old_ind, new_ind)
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
    checkeffect(tn, AddTensorEffect(tensor))
    addtensor_inner!(tn, tensor)
    handle!(tn, AddTensorEffect(tensor))
    return tn
end

checkeffect(tn, @nospecialize(e::AddTensorEffect)) = checkeffect(tn, e, DelegatorTrait(TensorNetwork(), tn))
checkeffect(tn, @nospecialize(e::AddTensorEffect), ::DelegateTo) = checkeffect(delegator(TensorNetwork(), tn), e)
function checkeffect(tn, @nospecialize(e::AddTensorEffect), ::DontDelegate)
    # TODO throw a custom EffectError
    hastensor(tn, e.f) && throw(ArgumentError("tensor already present"))
end

handle!(tn, @nospecialize(e::AddTensorEffect)) = handle!(tn, e, DelegatorTrait(TensorNetwork(), tn))
handle!(tn, @nospecialize(e::AddTensorEffect), ::DelegateTo) = handle!(delegator(TensorNetwork(), tn), e)
handle!(tn, @nospecialize(e::AddTensorEffect), ::DontDelegate) = throw(MethodError(tn, e))

## `rmtensor!`
function rmtensor!(tn, tensor)
    checkeffect(tn, RemoveTensorEffect(tensor))
    rmtensor_inner!(tn, tensor)
    handle!(tn, RemoveTensorEffect(tensor))
    return tn
end

checkeffect(tn, @nospecialize(e::RemoveTensorEffect)) = checkeffect(tn, e, DelegatorTrait(TensorNetwork(), tn))
checkeffect(tn, @nospecialize(e::RemoveTensorEffect), ::DelegateTo) = checkeffect(delegator(TensorNetwork(), tn), e)
function checkeffect(tn, @nospecialize(e::RemoveTensorEffect), ::DontDelegate)
    hastensor(tn, e.f) || throw(ArgumentError("tensor not found"))
end

handle!(tn, @nospecialize(e::RemoveTensorEffect)) = handle!(tn, e, DelegatorTrait(TensorNetwork(), tn))
handle!(tn, @nospecialize(e::RemoveTensorEffect), ::DelegateTo) = handle!(delegator(TensorNetwork(), tn), e)
handle!(tn, @nospecialize(e::RemoveTensorEffect), ::DontDelegate) = throw(MethodError(tn, e))

## `replace_tensor!`
function replace_tensor!(tn, old_tensor, new_tensor)
    checkeffect(tn, ReplaceEffect(old_tensor, new_tensor))
    replace_tensor_inner!(tn, old_tensor, new_tensor)
    handle!(tn, ReplaceEffect(old_tensor, new_tensor))
    return tn
end
replace_tensor!(tn, old_new::Pair) = replace_tensor!(tn, old_new.first, old_new.second)

function checkeffect(tn, @nospecialize(e::ReplaceEffect{<:Tensor,<:Tensor}))
    checkeffect(tn, e, DelegatorTrait(TensorNetwork(), tn))
end
function checkeffect(tn, @nospecialize(e::ReplaceEffect{<:Tensor,<:Tensor}), ::DelegateTo)
    checkeffect(delegator(TensorNetwork(), tn), e)
end
function checkeffect(tn, @nospecialize(e::ReplaceEffect{Told,Tnew}), ::DontDelegate) where {Told<:Tensor,Tnew<:Tensor}
    hastensor(tn, e.old) || throw(ArgumentError("old tensor not found"))
    hastensor(tn, e.new) && throw(ArgumentError("new tensor already exists"))

    if !isscoped(tn)
        @argcheck issetequal(inds(e.new), inds(e.old)) "replacing tensor indices don't match"
    end
end

handle!(tn, @nospecialize(e::ReplaceEffect{<:Tensor,<:Tensor})) = handle!(tn, e, DelegatorTrait(TensorNetwork(), tn))
function handle!(tn, @nospecialize(e::ReplaceEffect{<:Tensor,<:Tensor}), ::DelegateTo)
    handle!(delegator(TensorNetwork(), tn), e)
end
function handle!(_, @nospecialize(e::ReplaceEffect{Told,Tnew}), ::DontDelegate) where {Told<:Tensor,Tnew<:Tensor}
    throw(MethodError(tn, e))
end

## `replace_ind!`
function replace_ind!(tn, old_ind, new_ind)
    checkeffect(tn, ReplaceEffect(old_ind, new_ind))
    replace_ind_inner!(tn, old_ind, new_ind)
    handle!(tn, ReplaceEffect(old_ind, new_ind))
    return tn
end
replace_ind!(tn, old_new::Pair) = replace_ind!(tn, old_new.first, old_new.second)

function checkeffect(tn, @nospecialize(e::ReplaceEffect{<:Index,<:Index}))
    checkeffect(tn, e, DelegatorTrait(TensorNetwork(), tn))
end
function checkeffect(tn, @nospecialize(e::ReplaceEffect{<:Index,<:Index}), ::DelegateTo)
    checkeffect(delegator(TensorNetwork(), tn), e)
end
function checkeffect(tn, @nospecialize(e::ReplaceEffect{Iold,Inew}), ::DontDelegate) where {Iold<:Index,Inew<:Index}
    hasind(tn, e.old) || throw(ArgumentError("old index not found"))
    hasind(tn, e.new) && throw(ArgumentError("new index already exists"))
end

handle!(tn, @nospecialize(e::ReplaceEffect{<:Index,<:Index})) = handle!(tn, e, DelegatorTrait(TensorNetwork(), tn))
handle!(tn, @nospecialize(e::ReplaceEffect{<:Index,<:Index}), ::DelegateTo) = handle!(delegator(TensorNetwork(), tn), e)
function handle!(_, @nospecialize(e::ReplaceEffect{Iold,Inew}), ::DontDelegate) where {Iold<:Index,Inew<:Index}
    throw(MethodError(tn, e))
end
