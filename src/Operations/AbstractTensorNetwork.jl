abstract type AbstractTensorNetwork end

# TensorNetwork interface
Base.summary(io::IO, tn::T) where {T<:AbstractTensorNetwork} = print(io, "$(ntensors(tn))-tensors $T")

function Base.show(io::IO, tn::T) where {T<:AbstractTensorNetwork}
    print(io, "$T (#tensors=$(ntensors(tn)), #inds=$(ninds(tn)))")
end

Base.in(i::Index, tn::AbstractTensorNetwork) = hasind(tn, i)
Base.in(tensor::Tensor, tn::AbstractTensorNetwork) = hastensor(tn, tensor)

Base.size(tn::AbstractTensorNetwork) = size_inds(tn)
Base.size(tn::AbstractTensorNetwork, i::Index) = size_ind(tn, i)

Base.eltype(tn::AbstractTensorNetwork) = promote_type(eltype.(all_tensors(tn))...)

"""
    Base.collect(tn::AbstractTensorNetwork)

Return a list of the [`Tensor`](@ref)s in the Tensor Network. It is equivalent to `tensors(tn)`.
"""
Base.collect(tn::AbstractTensorNetwork) = all_tensors(tn)

"""
    Base.similar(tn::AbstractTensorNetwork)

Return a copy of the `TensorNetwork` with all [`Tensor`](@ref)s replaced by their `similar` version.
"""
function Base.similar(tn::AbstractTensorNetwork)
    tn = copy(tn)
    replace!(tn, all_tensors(tn) .=> similar.(all_tensors(tn)))
    return tn
end

"""
    Base.zero(tn::AbstractTensorNetwork)

Return a copy of the `TensorNetwork` with all [`Tensor`](@ref)s replaced by their `zero` version.
"""
function Base.zero(tn::AbstractTensorNetwork)
    tn = copy(tn)
    replace!(tn, all_tensors(tn) .=> zero.(all_tensors(tn)))
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
    replace!(tn, all_tensors(tn) .=> conj.(all_tensors(tn)))
    return tn
end

"""
    conj!(tn::AbstractTensorNetwork)

Conjugate all tensors in the [`AbstractTensorNetwork`](@ref) in-place.

See also: [`conj`](@ref).
"""
function Base.conj!(tn::AbstractTensorNetwork)
    foreach(conj!, all_tensors(tn))
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
    push!(tn::AbstractTensorNetwork, tensor)

Add a [`Tensor`](@ref) to the Tensor Network.
"""
Base.push!(tn::AbstractTensorNetwork, tensor::Tensor; kwargs...) = addtensor!(tn, tensor; kwargs...)

"""
    append!(tn::AbstractTensorNetwork, tensors)

Add a tensors to a Tensor Network from a list of [`Tensor`](@ref)s or from another Tensor Network.

See also: [`push!`](@ref).
"""
Base.append!(tn::AbstractTensorNetwork, tensors) = (foreach(Base.Fix1(push!, tn), tensors); tn)

# TODO how do we deal with the tags from the other Tensor Network?
# function Base.append!(tn::AbstractTensorNetwork, other::AbstractTensorNetwork)
#     (foreach(Base.Fix1(push!, tn), tensors(other)); tn)
# end

# TODO `pop!`
# """
#     pop!(tn::AbstractTensorNetwork, tensor::Tensor)
#     pop!(tn::AbstractTensorNetwork, i::Union{Symbol,AbstractVecOrTuple{Symbol}})

# Remove and return the first tensor in `tn`` that satisfies _egality_ (i.e. `≡`or`===`) with `tensor`.

# See also: [`push!`](@ref), [`delete!`](@ref).
# """
# Base.pop!(tn::AbstractTensorNetwork, tensor::Tensor) = (delete!(tn, tensor); tensor)

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
            EinExprs.EinExpr.(Iterators.map(collect ∘ inds, tensors(tn)))
        ),
        Dict(ind => size(tn, ind) for ind in inds(tn))
    )
    #! format: on

    # don't use `sum(::Vector{EinExpr})`: it's broken and takes x10 more time
    return EinExprs.einexpr(optimizer, path; kwargs...)
end

# Taggable interface
# TODO Base.getindex
# TODO Base.setindex!
# WARN the decoupling of `Site`/`Link` in `Taggable` can affect this

# Pluggable interface
"""
    Base.adjoint(::AbstractTensorNetwork)

Return the adjoint of a Pluggable Tensor Network; i.e. the conjugate Tensor Network with the inputs and outputs swapped.
"""
Base.adjoint(tn::AbstractTensorNetwork) = adjoint_plugs!(conj(tn))

"""
    LinearAlgebra.adjoint!(::AbstractTensorNetwork)

Like [`adjoint`](@ref), but in-place.
"""
LinearAlgebra.adjoint!(tn::AbstractTensorNetwork) = adjoint_plugs!(conj!(tn))

# Attributeable interface
# TODO Base.get?

# Pluggable interface
# TODO couldn't be written more generically as `plug` belongs to `QuantumTags`. maybe rewrite?
## `plug`
plug(tn::AbstractTensorNetwork; kwargs...) = plug(sort_nt(values(kwargs)), tn)
plug(::NamedTuple{(:at,)}, tn) = plug_at(tn, kwargs.at)
plug(::NamedTuple{(:like,)}, tn) = plug_like(tn, kwargs.like)
