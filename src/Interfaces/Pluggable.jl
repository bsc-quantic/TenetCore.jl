using LinearAlgebra: LinearAlgebra
using QuantumTags: isdual, is_plug_equal
using ValSplit

# interface object
struct Pluggable <: Interface end

# TODO move to `Networks`? use `DelegateTo{Taggable}`?
struct DelegateToInterface{T} <: DelegatorTrait end

# keyword-dispatching methods
function plugs end
# function plug end
:(QuantumTags.plug)

# query methods
function all_plugs end
function all_plugs_iter end

function hasplug end
function nplugs end

function plugs_like end
function plug_like end
function ind_at_plug end
# function plug_at end

function plugs_set end
function plugs_set_inputs end
function plugs_set_outputs end

function inds_set_physical end
function inds_set_virtual end
function inds_set_inputs end
function inds_set_outputs end

# mutating methods
# function set_plug_inner! end
# function unset_plug_inner! end

# function set_plug! end
# function unset_plug! end

# effects
# TODO aren't these affected or treated already by the `Taggable` interface?
"""
    SetPlugEffect{Tag,Obj} <: Effect

Represents the effect of setting a mapping between a `Plug` `Tag` and an `Obj`ect.
"""
struct SetPlugEffect{T,O} <: Effect
    plug::T
    obj::O
end

SetPlugEffect(plug::T, @nospecialize(obj::Index)) where {T} = SetPlugEffect{T,Index}(plug, obj)

"""
    UnsetPlugEffect{Tag} <: Effect

Represents the effect of unsetting a mapping of a `Plug` `Tag`.
"""
struct UnsetPlugEffect{T} <: Effect
    plug::T
end

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
all_plugs(tn, ::DelegateTo) = all_plugs(delegator(Pluggable(), tn))
all_plugs(tn, ::DontDelegate) = throw(MethodError(all_plugs, (tn,)))
all_plugs(tn, ::DelegateToInterface{Taggable}) = filter(isplug, all_links_iter(tn))

## `all_plugs_iter`
all_plugs_iter(tn) = all_plugs_iter(tn, DelegatorTrait(Pluggable(), tn))
all_plugs_iter(tn, ::DelegateTo) = all_plugs_iter(delegator(Pluggable(), tn))
function all_plugs_iter(tn, ::DontDelegate)
    fallback(all_plugs_iter)
    all_plugs(tn)
end
all_plugs_iter(tn, ::DelegateToInterface{Taggable}) = Iterators.filter(isplug, all_links_iter(tn))

## `hasplug`
hasplug(tn, plug) = hasplug(tn, plug, DelegatorTrait(Pluggable(), tn))
hasplug(tn, plug, ::DelegateTo) = hasplug(delegator(Pluggable(), tn), plug)
hasplug(tn, plug, ::DontDelegate) = any(Base.Fix1(is_plug_equal, plug), all_plugs_iter(tn))
hasplug(tn, plug, ::DelegateToInterface{Taggable}) = any(Base.Fix1(is_plug_equal, plug), all_links_iter(tn)) # CONTINUE HERE

## `nplugs`
nplugs(tn) = nplugs(tn, DelegatorTrait(Pluggable(), tn))
nplugs(tn, ::DelegateTo) = nplugs(delegator(Pluggable(), tn))
nplugs(tn, ::DontDelegate) = length(all_plugs(tn))
nplugs(tn, ::DelegateToInterface{Taggable}) = length(all_plugs(tn))

## `plugs_like`
plugs_like(tn, plug) = plugs_like(tn, plug, DelegatorTrait(Pluggable(), tn))
plugs_like(tn, plug, ::DelegateTo) = plugs_like(delegator(Pluggable(), tn), plug)
plugs_like(tn, plug, ::DontDelegate) = filter(Base.Fix1(is_plug_equal, plug), all_plugs(tn))
plugs_like(tn, plug, ::DelegateToInterface{Taggable}) = links_like(is_plug_equal, tn, plug)

## `plug_like`
plug_like(tn, plug) = plug_like(tn, plug, DelegatorTrait(Pluggable(), tn))
plug_like(tn, plug, ::DelegateTo) = plug_like(delegator(Pluggable(), tn), plug)
plug_like(tn, plug, ::DontDelegate) = first(Iterators.filter(Base.Fix1(is_plug_equal, plug), all_plugs_iter(tn)))
plug_like(tn, plug, ::DelegateToInterface{Taggable}) = link_like(is_plug_equal, tn, plug)

## `ind_at_plug`
ind_at_plug(tn, plug) = ind_at_plug(tn, plug, DelegatorTrait(Pluggable(), tn))
ind_at_plug(tn, plug, ::DelegateTo) = ind_at_plug(delegator(Pluggable(), tn), plug)
ind_at_plug(tn, plug, ::DontDelegate) = throw(MethodError(ind_at_plug, (tn, plug)))
ind_at_plug(tn, plug, ::DelegateToInterface{Taggable}) = ind_at(tn, link_like(is_plug_equal, tn, plug))

### alias
ind(kwargs::NamedTuple{(:plug,)}, tn) = ind_at_plug(tn, kwargs.plug)

## `plug_at`
# plug_at(tn, plug) = first(Iterators.filter(Base.Fix1(is_plug_equal, plug), all_links_iter(tn)))

## `plugs_set`
@valsplit plugs_set(tn, Val(set::Symbol)) = throw(ArgumentError("invalid `set` values: $(set)"))

plugs_set(tn, ::Val{:all}) = plugs_set_all(tn)
plugs_set_all(tn) = all_plugs(tn)

plugs_set(tn, ::Val{:inputs}) = plugs_set_inputs(tn)
plugs_set_inputs(tn) = plugs_set_inputs(tn, DelegatorTrait(Pluggable(), tn))
plugs_set_inputs(tn, ::DelegateTo) = plugs_set_inputs(delegator(Pluggable(), tn))
plugs_set_inputs(tn, ::DontDelegate) = filter(t -> isdual(t), all_plugs(tn))
plugs_set_inputs(tn, ::DelegateToInterface) = plugs_set_inputs(tn, DontDelegate())

plugs_set(tn, ::Val{:outputs}) = plugs_set_outputs(tn)
plugs_set_outputs(tn) = plugs_set_outputs(tn, DelegatorTrait(Pluggable(), tn))
plugs_set_outputs(tn, ::DelegateTo) = plugs_set_outputs(delegator(Pluggable(), tn))
plugs_set_outputs(tn, ::DontDelegate) = filter(t -> !isdual(t), all_plugs(tn))
plugs_set_outputs(tn, ::DelegateToInterface) = plugs_set_outputs(tn, DontDelegate())

## `inds_set` extensions
inds_set(tn, ::Val{:physical}) = inds_set_physical(tn)
inds_set_physical(tn) = inds_set_physical(tn, DelegatorTrait(Pluggable(), tn))
inds_set_physical(tn, ::DelegateTo) = inds_set_physical(delegator(Pluggable(), tn))
inds_set_physical(tn, ::DontDelegate) = Index[ind_at_plug(tn, i) for i in all_plugs(tn)]
inds_set_physical(tn, ::DelegateToInterface{Taggable}) = inds_set_physical(tn, DontDelegate())

inds_set(tn, ::Val{:virtual}) = inds_set_virtual(tn)
inds_set_virtual(tn) = inds_set_virtual(tn, DelegatorTrait(Pluggable(), tn))
inds_set_virtual(tn, ::DelegateTo) = inds_set_virtual(delegator(Pluggable(), tn))
inds_set_virtual(tn, ::DontDelegate) = setdiff(all_inds(tn), inds_set_physical(tn))
inds_set_virtual(tn, ::DelegateToInterface{Taggable}) = inds_set_virtual(tn, DontDelegate())

inds_set(tn, ::Val{:inputs}) = inds_set_inputs(tn)
inds_set_inputs(tn) = inds_set_inputs(tn, DelegatorTrait(Pluggable(), tn))
inds_set_inputs(tn, ::DelegateTo) = inds_set_inputs(delegator(Pluggable(), tn))
inds_set_inputs(tn, ::DontDelegate) = Index[ind_at_plug(tn, i) for i in plugs_set_inputs(tn)]
inds_set_inputs(tn, ::DelegateToInterface{Taggable}) = inds_set_inputs(tn, DontDelegate())

inds_set(tn, ::Val{:outputs}) = inds_set_outputs(tn)
inds_set_outputs(tn) = inds_set_outputs(tn, DelegatorTrait(Pluggable(), tn))
inds_set_outputs(tn, ::DelegateTo) = inds_set_outputs(delegator(Pluggable(), tn))
inds_set_outputs(tn, ::DontDelegate) = Index[ind_at_plug(tn, i) for i in plugs_set_outputs(tn)]
inds_set_outputs(tn, ::DelegateToInterface{Taggable}) = inds_set_outputs(tn, DontDelegate())
