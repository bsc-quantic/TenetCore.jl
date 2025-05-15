using Base: AbstractVecOrTuple
using ArgCheck
using ValSplit
using Networks
using QuantumTags
using EinExprs: EinExprs
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

function tensors_with_inds end
function tensors_contain_inds end
function tensors_intersect_inds end

function inds_set end
function inds_parallel_to end

function size_inds end
function size_ind end

# extra: optional methods that could be other interfaces...
## get vertex/edge from tensor/index
function tensor_vertex end
function index_edge end

## get tensor/index from vertex/edge
function vertex_tensor end
function edge_index end

# mutating methods
function addtensor! end
function addtensor_inner! end

function rmtensor! end
function rmtensor_inner! end

function replace_tensor! end
function replace_tensor_inner! end

function replace_ind! end
function replace_ind_inner! end

"""
    slice!(tn, index::Symbol, i)

In-place projection of `index` on dimension `i`.

See also: [`selectdim`](@ref), [`view`](@ref).
"""
function slice! end
function slice_inner! end

"""
    fuse!(tn, ind)

Group indices parallel to `ind` and reshape the tensors accordingly.
"""
function fuse! end
function fuse_inner! end

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

"""
    SliceEffect{I,D} <: Effect

Represents the effect of slicing an index on a dimension.
"""
struct SliceEffect{D} <: Effect
    ind::Index
    dim::D
end

"""
    FuseEffect <: Effect

Represents the effect of fusing indices in a Tensor Network.
"""
struct FuseEffect <: Effect
    old_inds::Vector{Index}
    new_ind::Index
end

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
tensor(kwargs::@NamedTuple{vertex::V}, tn) where {V} = vertex_tensor(tn, kwargs.vertex)

## `inds`
inds(tn; kwargs...) = inds(sort_nt(values(kwargs)), tn)
inds(::@NamedTuple{}, tn) = all_inds(tn) # inds((;), tn, DelegatorTrait(TensorNetwork(), tn))
inds(kwargs::@NamedTuple{set::Symbol}, tn) = inds_set(tn, kwargs.set)
inds(kwargs::NamedTuple{(:parallel_to,)}, tn) = inds_parallel_to(tn, kwargs.parallel_to)
inds(kwargs::NamedTuple{(:parallelto,)}, tn) = inds_parallel_to(tn, kwargs.parallelto)

### singular version of `inds`
ind(tn; kwargs...) = ind(sort_nt(values(kwargs)), tn)
ind(kwargs::NamedTuple, tn) = only(inds(kwargs, tn))
ind(kwargs::@NamedTuple{edge::E}, tn) where {E} = edge_index(tn, kwargs.edge)

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

## `tensor_vertex`
tensor_vertex(tn, tensor) = tensor_vertex(tn, tensor, DelegatorTrait(TensorNetwork(), tn))
tensor_vertex(tn, tensor, ::DelegateTo) = tensor_vertex(delegator(TensorNetwork(), tn), tensor)
function tensor_vertex(tn, tensor, ::DontDelegate)
    @debug "Falling back to default `tensor_vertex` method"
    @argcheck hastensor(tn, tensor) "tensor $tensor not found in the Tensor Network"
    if vertex_type(tn) >: Tensor
        return Vertex(tensor)
    else
        throw(MethodError(tensor_vertex, (tn, tensor)))
    end
end

### helper method
Networks.vertex(tn, tensor::Tensor) = tensor_vertex(tn, tensor)

## `index_edge`
index_edge(tn, index) = index_edge(tn, index, DelegatorTrait(TensorNetwork(), tn))
index_edge(tn, index, ::DelegateTo) = index_edge(delegator(TensorNetwork(), tn), index)
function index_edge(tn, index, ::DontDelegate)
    @debug "Falling back to default `index_edge` method"
    @argcheck hasind(tn, index) "index $index not found in the Tensor Network"
    if edge_type(tn) >: Index
        return Edge(index)
    else
        throw(MethodError(index_edge, (tn, index)))
    end
end

### helper method
Networks.edge(tn, index::Index) = index_edge(tn, index)

## `vertex_tensor`
vertex_tensor(tn, vertex) = vertex_tensor(tn, vertex, DelegatorTrait(TensorNetwork(), tn))
vertex_tensor(tn, vertex, ::DelegateTo) = vertex_tensor(delegator(TensorNetwork(), tn), vertex)
vertex_tensor(tn, vertex, ::DontDelegate) = throw(MethodError(vertex_tensor, (tn, vertex)))

## `edge_index`
edge_index(tn, edge) = edge_index(tn, edge, DelegatorTrait(TensorNetwork(), tn))
edge_index(tn, edge, ::DelegateTo) = edge_index(delegator(TensorNetwork(), tn), edge)
edge_index(tn, edge, ::DontDelegate) = throw(MethodError(edge_index, (tn, edge)))

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
handle!(_, @nospecialize(e::AddTensorEffect), ::DontDelegate) = nothing # throw(MethodError(handle!, (tn, e)))

# `addtensor_inner!`
addtensor_inner!(tn, tensor) = addtensor_inner!(tn, tensor, DelegatorTrait(TensorNetwork(), tn))
addtensor_inner!(tn, tensor, ::DelegateTo) = addtensor!(delegator(TensorNetwork(), tn), tensor)
addtensor_inner!(tn, tensor, ::DontDelegate) = throw(MethodError(addtensor_inner!, (tn, tensor)))

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
handle!(_, @nospecialize(e::RemoveTensorEffect), ::DontDelegate) = nothing # throw(MethodError(handle!, (tn, e)))

## `rmtensor_inner!`
rmtensor_inner!(tn, tensor) = rmtensor_inner!(tn, tensor, DelegatorTrait(TensorNetwork(), tn))
rmtensor_inner!(tn, tensor, ::DelegateTo) = rmtensor!(delegator(TensorNetwork(), tn), tensor)
rmtensor_inner!(tn, tensor, ::DontDelegate) = throw(MethodError(rmtensor_inner!, (tn, tensor)))

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

    # if !isscoped(tn)
    @argcheck issetequal(inds(e.new), inds(e.old)) "replacing tensor indices don't match"
    # end
end

handle!(tn, @nospecialize(e::ReplaceEffect{<:Tensor,<:Tensor})) = handle!(tn, e, DelegatorTrait(TensorNetwork(), tn))
function handle!(tn, @nospecialize(e::ReplaceEffect{<:Tensor,<:Tensor}), ::DelegateTo)
    handle!(delegator(TensorNetwork(), tn), e)
end
handle!(_, @nospecialize(e::ReplaceEffect{Told,Tnew}), ::DontDelegate) where {Told<:Tensor,Tnew<:Tensor} = nothing # throw(MethodError(handle!, (tn, e)))

## `replace_tensor_inner!`
replace_tensor_inner!(tn, old, new) = replace_tensor_inner!(tn, old, new, DelegatorTrait(TensorNetwork(), tn))
replace_tensor_inner!(tn, old, new, ::DelegateTo) = replace_tensor!(delegator(TensorNetwork(), tn), old, new)
replace_tensor_inner!(tn, old, new, ::DontDelegate) = throw(MethodError(replace_tensor_inner!, (tn, old, new)))

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
    hasind(tn, e.old) || throw(ArgumentError("old index ($(e.old)) not found"))
    hasind(tn, e.new) && throw(ArgumentError("new index ($(e.new)) already exists"))
end

handle!(tn, @nospecialize(e::ReplaceEffect{<:Index,<:Index})) = handle!(tn, e, DelegatorTrait(TensorNetwork(), tn))
handle!(tn, @nospecialize(e::ReplaceEffect{<:Index,<:Index}), ::DelegateTo) = handle!(delegator(TensorNetwork(), tn), e)
handle!(_, @nospecialize(e::ReplaceEffect{Iold,Inew}), ::DontDelegate) where {Iold<:Index,Inew<:Index} = nothing # throw(MethodError(handle!, (tn, e)))

## `replace_ind_inner!`
replace_ind_inner!(tn, old, new) = replace_ind_inner!(tn, old, new, DelegatorTrait(TensorNetwork(), tn))
replace_ind_inner!(tn, old, new, ::DelegateTo) = replace_ind!(delegator(TensorNetwork(), tn), old, new)
replace_ind_inner!(tn, old, new, ::DontDelegate) = throw(MethodError(replace_ind_inner!, (tn, old, new)))

## `slice!`
function slice!(tn, ind, i)
    checkeffect(tn, SliceEffect(ind, i))
    slice_inner!(tn, ind, i)
    handle!(tn, SliceEffect(ind, i))
    return tn
end

checkeffect(tn, @nospecialize(e::SliceEffect)) = checkeffect(tn, e, DelegatorTrait(TensorNetwork(), tn))
checkeffect(tn, @nospecialize(e::SliceEffect), ::DelegateTo) = checkeffect(delegator(TensorNetwork(), tn), e)
function checkeffect(tn, e::SliceEffect, ::DontDelegate)
    hasind(tn, e.ind) || throw(ArgumentError("index ($(e.ind)) not found"))
end

handle!(tn, @nospecialize(e::SliceEffect)) = handle!(tn, e, DelegatorTrait(TensorNetwork(), tn))
handle!(tn, @nospecialize(e::SliceEffect), ::DelegateTo) = handle!(delegator(TensorNetwork(), tn), e)
handle!(_, e::SliceEffect, ::DontDelegate) = nothing # throw(MethodError(handle!, (tn, e)))

## `slice_inner!`
slice_inner!(tn, ind, i) = slice_inner!(tn, ind, i, DelegatorTrait(TensorNetwork(), tn))
slice_inner!(tn, ind, i, ::DelegateTo) = slice_inner!(delegator(TensorNetwork(), tn), ind, i)
slice_inner!(tn, ind, i, ::DontDelegate) = throw(MethodError(slice_inner!, (tn, ind, i)))

## `fuse!`
function fuse!(tn, i)
    parinds = inds(tn; parallelto=i)
    length(parinds) == 0 && return tn

    parinds = (i,) ∪ parinds
    checkeffect(tn, FuseEffect(parinds, i))
    fuse_inner!(tn, parinds)
    handle!(tn, FuseEffect(parinds, i))

    return tn
end

## `fuse_inner!`
fuse_inner!(tn, i) = fuse_inner!(tn, i, DelegatorTrait(TensorNetwork(), tn))
fuse_inner!(tn, i, ::DelegateTo) = fuse!(DelegatorTrait(TensorNetwork(), tn), i)

# TODO replace ind for `Index(Fused(parinds))`?
function fuse_inner!(tn, parinds, ::DontDelegate)
    @debug "Fallback to default fuse_inner! for $(typeof(tn))"
    @unsafe_region tn for tensor in tensors(tn; intersect=parinds)
        # the way it is currently implemented, we must emit a `ReplaceEffect` because `Tensors` have changed
        # TODO maybe refactor this when we stop using `Tensors` as graph vertices?
        replace_tensor!(tn, tensor, Muscle.fuse(tensor, parinds))
    end
    return tn
end
