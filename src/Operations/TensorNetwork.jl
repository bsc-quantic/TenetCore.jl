# TODO Base.copy ==> copy_tn

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

"""
    arrays(tn; kwargs...)

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
    for leaf in EinExprs.leaves(path)
        selection = tensors(tn; withinds=EinExprs.head(leaf))
        if length(selection) > 1
            @warn "Found more than one tensor with index $(EinExprs.head(leaf))... Using first one"
        end
        selection = first(selection)
        cache[leaf] = selection
        delete!(tn, selection)
    end

    for intermediate in EinExprs.Branches(path)
        if EinExprs.nargs(intermediate) == 1
            a = only(EinExprs.args(intermediate))
            cache[intermediate] = Muscle.Operations.unary_einsum(cache[a]; dims=EinExprs.suminds(intermediate))
            delete!(cache, a)
        elseif EinExprs.nargs(intermediate) == 2
            a, b = EinExprs.args(intermediate)
            cache[intermediate] = Muscle.Operations.binary_einsum(
                cache[a], cache[b]; dims=EinExprs.suminds(intermediate)
            )
            delete!(cache, a)
            delete!(cache, b)
        else
            # TODO we should fix this in EinExprs, this is a temporal fix meanwhile
            @warn "Found a contraction with $(EinExprs.nargs(intermediate)) arguments... Using reduction which might be sub-optimal"
            target_tensors = map(EinExprs.args(intermediate)) do branch
                pop!(cache, branch)
            end
            cache[intermediate] = foldl(target_tensors) do a, b
                Muscle.Operations.binary_einsum(a, b; dims=EinExprs.suminds(intermediate))
            end
        end
    end
    return cache[path]
end

# TODO to add in the future
# """
#     gauge!(tn, ind, U[, Uinv])

# Perform a gauge transformation on index `ind`.
# """
# function gauge!(tn, ind::Symbol, U::AbstractMatrix, Uinv::AbstractMatrix=inv(U))
#     a, b = tensors(tn; contains=ind)
#     tmpind = gensym(ind)

#     tU = Tensor(U, [ind, tmpind])
#     tUinv = Tensor(Uinv, [tmpind, ind])

#     gauged_a = replace(contract(a, tU), tmpind => ind)
#     gauged_b = replace(contract(tUinv, b), tmpind => ind)

#     replace!(tn, [a => gauged_a, b => gauged_b])
# end

"""
    resetinds!(tn, method=:gensymnew; kwargs...)

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
    fuse!(tn, i::Symbol)

Group indices parallel to `i` and reshape the tensors accordingly.
"""
function fuse!(tn, i)
    parinds = filter!(!=(i), inds(tn; parallelto=i))
    length(parinds) == 0 && return tn

    # TODO replace ind for `Index(Fused(parinds))`?
    parinds = (i,) ∪ parinds
    newtensors = map(Base.Fix2(fuse, parinds), pop!(tn, parinds))

    append!(tn, newtensors)

    return tn
end
