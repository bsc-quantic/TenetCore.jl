using LinearAlgebra: LinearAlgebra
using QuantumTags: isdual, is_plug_equal
using ValSplit

# interface object
struct Pluggable <: Interface end

# keyword-dispatching methods
function plugs end
# function plug end
:(QuantumTags.plug)

# query methods
function all_plugs end
function all_plugs_iter end

function hasplug end
function nplugs end

function plug_at end

function plugs_like end
function plug_like end

function plugs_set end
function plugs_set_inputs end
function plugs_set_outputs end

# TODO move to Pluggable + TensorNetwork
function inds_set_physical end
function inds_set_virtual end
function inds_set_inputs end
function inds_set_outputs end

# mutating methods
function setplug! end
function unsetplug! end

# implementation
## `plugs`
plugs(tn; kwargs...) = plugs(sort_nt(values(kwargs)), tn)
plugs(::@NamedTuple{}, tn) = all_plugs(tn)
plugs(kwargs::NamedTuple{(:set,)}, tn) = plugs_set(tn, kwargs.set)

## `plug`
### NOTE in `Operations/AbstractTensorNetwork.jl` because `plug` belongs to `QuantumTags` and thus,
### it needs to use `AbstractTensorNetwork` to avoid piracy

## `all_plugs`
all_plugs(tn) = all_plugs(tn, DelegatorTrait(Pluggable(), tn))
all_plugs(tn, ::DelegateToField) = all_plugs(delegator(Pluggable(), tn))
all_plugs(tn, ::DontDelegate) = throw(MethodError(all_plugs, (tn,)))

## `all_plugs_iter`
all_plugs_iter(tn) = all_plugs_iter(tn, DelegatorTrait(Pluggable(), tn))
all_plugs_iter(tn, ::DelegateToField) = all_plugs_iter(delegator(Pluggable(), tn))
function all_plugs_iter(tn, ::DontDelegate)
    fallback(all_plugs_iter)
    all_plugs(tn)
end

## `hasplug`
hasplug(tn, plug) = hasplug(tn, plug, DelegatorTrait(Pluggable(), tn))
hasplug(tn, plug, ::DelegateToField) = hasplug(delegator(Pluggable(), tn), plug)
function hasplug(tn, plug, ::DontDelegate)
    fallback(hasplug)
    any(Base.Fix1(is_plug_equal, plug), all_plugs_iter(tn))
end

## `nplugs`
nplugs(tn) = nplugs(tn, DelegatorTrait(Pluggable(), tn))
nplugs(tn, ::DelegateToField) = nplugs(delegator(Pluggable(), tn))
function nplugs(tn, ::DontDelegate)
    fallback(nplugs)
    length(all_plugs(tn))
end

## `plug_at`
plug_at(tn, plug) = plug_at(tn, plug, DelegatorTrait(Pluggable(), tn))
plug_at(tn, plug, ::DelegateToField) = plug_at(delegator(Pluggable(), tn), plug)
plut_at(tn, plug, ::DontDelegate) = throw(MethodError(plug_at, (tn, plug)))
# plug_at(tn, plug) = first(Iterators.filter(Base.Fix1(is_plug_equal, plug), all_links_iter(tn)))

## `plugs_like`
plugs_like(tn, plug) = plugs_like(tn, plug, DelegatorTrait(Pluggable(), tn))
plugs_like(tn, plug, ::DelegateToField) = plugs_like(delegator(Pluggable(), tn), plug)
function plugs_like(tn, plug, ::DontDelegate)
    fallback(plugs_like)
    filter(Base.Fix1(is_plug_equal, plug), all_plugs(tn))
end

## `plug_like`
plug_like(tn, plug) = plug_like(tn, plug, DelegatorTrait(Pluggable(), tn))
plug_like(tn, plug, ::DelegateToField) = plug_like(delegator(Pluggable(), tn), plug)
function plug_like(tn, plug, ::DontDelegate)
    fallback(plug_like)
    first(Iterators.filter(Base.Fix1(is_plug_equal, plug), all_plugs_iter(tn)))
end

## `plugs_set`
@valsplit plugs_set(tn, Val(set::Symbol)) = throw(ArgumentError("invalid `set` values: $(set)"))

plugs_set(tn, ::Val{:all}) = plugs_set_all(tn)
plugs_set_all(tn) = all_plugs(tn)

plugs_set(tn, ::Val{:inputs}) = plugs_set_inputs(tn)
plugs_set_inputs(tn) = plugs_set_inputs(tn, DelegatorTrait(Pluggable(), tn))
plugs_set_inputs(tn, ::DelegateToField) = plugs_set_inputs(delegator(Pluggable(), tn))
function plugs_set_inputs(tn, ::DontDelegate)
    fallback(plugs_set_inputs)
    filter(t -> isdual(t), all_plugs(tn))
end

plugs_set(tn, ::Val{:outputs}) = plugs_set_outputs(tn)
plugs_set_outputs(tn) = plugs_set_outputs(tn, DelegatorTrait(Pluggable(), tn))
plugs_set_outputs(tn, ::DelegateToField) = plugs_set_outputs(delegator(Pluggable(), tn))
function plugs_set_outputs(tn, ::DontDelegate)
    fallback(plugs_set_outputs)
    filter(t -> !isdual(t), all_plugs(tn))
end

## `inds_set` extensions
inds_set(tn, ::Val{:physical}) = inds_set_physical(tn)
inds_set_physical(tn) = inds_set_physical(tn, DelegatorTrait(Pluggable(), tn))
inds_set_physical(tn, ::DelegateToField) = inds_set_physical(delegator(Pluggable(), tn))
function inds_set_physical(tn, ::DontDelegate)
    fallback(inds_set_physical)
    Index[ind_at(tn, i) for i in all_plugs(tn)]
end

inds_set(tn, ::Val{:virtual}) = inds_set_virtual(tn)
inds_set_virtual(tn) = inds_set_virtual(tn, DelegatorTrait(Pluggable(), tn))
inds_set_virtual(tn, ::DelegateToField) = inds_set_virtual(delegator(Pluggable(), tn))
function inds_set_virtual(tn, ::DontDelegate)
    fallback(inds_set_virtual)
    setdiff(all_inds(tn), inds_set_physical(tn))
end

inds_set(tn, ::Val{:inputs}) = inds_set_inputs(tn)
inds_set_inputs(tn) = inds_set_inputs(tn, DelegatorTrait(Pluggable(), tn))
inds_set_inputs(tn, ::DelegateToField) = inds_set_inputs(delegator(Pluggable(), tn))
function inds_set_inputs(tn, ::DontDelegate)
    fallback(inds_set_inputs)
    Index[ind_at(tn, i) for i in plugs_set_inputs(tn)]
end

inds_set(tn, ::Val{:outputs}) = inds_set_outputs(tn)
inds_set_outputs(tn) = inds_set_outputs(tn, DelegatorTrait(Pluggable(), tn))
inds_set_outputs(tn, ::DelegateToField) = inds_set_outputs(delegator(Pluggable(), tn))
function inds_set_outputs(tn, ::DontDelegate)
    fallback(inds_set_outputs)
    Index[ind_at(tn, i) for i in plugs_set_outputs(tn)]
end

## `setplug!`
setplug!(tn, x, plug) = setplug!(tn, x, plug, DelegatorTrait(Pluggable(), tn))
setplug!(tn, x, plug, ::DelegateToField) = setplug!(delegator(Pluggable(), tn), x, plug)
setplug!(tn, x, plug, ::DontDelegate) = throw(MethodError(setplug!, (tn, x, plug)))

## `unsetplug!`
unsetplug!(tn, plug) = unsetplug!(tn, plug, DelegatorTrait(Pluggable(), tn))
unsetplug!(tn, plug, ::DelegateToField) = unsetplug!(delegator(Pluggable(), tn), plug)
unsetplug!(tn, plug, ::DontDelegate) = throw(MethodError(unsetplug!, (tn, plug)))
