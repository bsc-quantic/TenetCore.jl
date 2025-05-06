using LinearAlgebra: LinearAlgebra
using QuantumTags: isdual, is_plug_equal
using ValSplit

# interface object
struct Pluggable <: Interface end

# keyword-dispatching methods
function plugs end
function plug end

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

function adjoint_plugs! end

# implementation
## `plugs`
plugs(tn; kwargs...) = plugs(sort_nt(values(kwargs)), tn)
plugs(::@NamedTuple{}, tn) = all_plugs(tn)
plugs(kwargs::NamedTuple{(:set,)}, tn) = plugs_set(tn, kwargs.set)

## `plug`
plug(tn::AbstractTensorNetwork; kwargs...) = plug(sort_nt(values(kwargs)), tn)
plug(::NamedTuple{(:at,)}, tn) = plug_at(tn, kwargs.at)
plug(::NamedTuple{(:like,)}, tn) = plug_like(tn, kwargs.like)

## `all_plugs`
all_plugs(tn) = all_plugs(tn, delegates(Pluggable(), tn))
all_plugs(tn, ::DelegateTo) = all_plugs(delegate(Pluggable(), tn))
all_plugs(tn, ::DontDelegate) = throw(MethodError(all_plugs, (tn,)))

## `all_plugs_iter`
all_plugs_iter(tn) = all_plugs_iter(tn, delegates(Pluggable(), tn))
all_plugs_iter(tn, ::DelegateTo) = all_plugs_iter(delegate(Pluggable(), tn))
function all_plugs_iter(tn, ::DontDelegate)
    @debug "Falling back to default implementation of `all_plugs_iter`"
    all_plugs(tn)
end

## `hasplug`
hasplug(tn, plug) = any(Base.Fix1(is_plug_equal, plug), links_iter(tn))

## `nplugs`
nplugs(tn; kwargs...) = length(plugs(tn; kwargs...))

## `plugs_like`
plugs_like(tn, plug) = links_like(is_plug_equal, tn, plug)

## `plug_like`
plug_like(tn, plug) = link_like(is_plug_equal, tn, plug)

## `ind_at_plug`
ind_at_plug(tn, plug) = ind_at(tn, tag_like(tn, plug))
ind(kwargs::NamedTuple{(:plug,)}, tn) = ind_at_plug(tn, kwargs.plug)

## `plug_at`
# plug_at(tn, plug) = first(Iterators.filter(Base.Fix1(is_plug_equal, plug), links_iter(tn)))

## `plugs_set`
@valsplit plugs_set(tn, Val(set::Symbol)) = throw(ArgumentError("invalid `set` values: $(set)"))

plugs_set(tn, ::Val{:all}) = plugs_set_all(tn)
plugs_set_all(tn) = all_plugs(tn)

plugs_set(tn, ::Val{:inputs}) = plugs_set_inputs(tn)
plugs_set_inputs(tn) = filter(t -> isplug(t) && isdual(t), all_links_iter(tn))

plugs_set(tn, ::Val{:outputs}) = plugs_set_outputs(tn)
plugs_set_outputs(tn) = filter(t -> isplug(t) && !isdual(t), all_links_iter(tn))

## `inds_set` extensions
inds_set(tn, ::Val{:physical}) = inds_set_physical(tn)
inds_set_physical(tn) = Index[ind_at(tn, i) for i in plugs(tn)]

inds_set(tn, ::Val{:virtual}) = inds_set_virtual(tn)
inds_set_virtual(tn) = setdiff(all_inds(tn), Index[ind_at(tn, i) for i in plugs(tn; set=:physical)])

inds_set(tn, ::Val{:inputs}) = inds_set_inputs(tn)
inds_set_inputs(tn) = Index[ind_at(tn, i) for i in plugs(tn; set=:inputs)]

inds_set(tn, ::Val{:outputs}) = inds_set_outputs(tn)
inds_set_outputs(tn) = Index[ind_at(tn, i) for i in plugs(tn; set=:outputs)]

## `adjoint_plugs!`
function adjoint_plugs!(tn)
    # update plug information and rename inner indices
    # generate mapping
    mapping = Dict(plug => ind(tn; at=plug) for plug in plugs(tn))

    # remove sites preemptively to avoid issues on renaming
    for plug_tag in plugs(tn)
        untag!(tn, plug_tag)
    end

    # set new site mapping
    for (site, index) in mapping
        tag!(tn, index, site')
    end

    # rename inner indices
    # replace!(tn, map(i -> i => Symbol(i, "'"), inds(tn; set=:virtual)))

    return tn
end
