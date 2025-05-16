function adjoint_plugs!(tn)
    # update plug information and rename inner indices
    # generate mapping
    mapping = Dict(plug => ind(tn; at=plug) for plug in all_plugs(tn))

    # remove sites preemptively to avoid issues on renaming
    for plug_tag in all_plugs(tn)
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

"""
    align!(a, ioa, b, iob)

Align the physical indices of `b` to match the physical indices of `a`. `ioa` and `iob` are either `:inputs` or `:outputs`.
"""
function align!(a, ioa, b, iob)
    @assert ioa === :inputs || ioa === :outputs
    @assert iob === :inputs || iob === :outputs

    # If `reset=true`, then all indices are renamed. If `reset=false`, then only the indices of the input/output sites are renamed.

    # if !isdisjoint(inds(a), inds(b))
    #     @warn "Overlapping indices"
    # end

    # if reset
    #     @debug "[align!] Renaming indices of b"
    #     resetinds!(b, :gensymclean)
    # end

    target_plugs_a = plugs(a; set=ioa)
    target_plugs_b = plugs(b; set=iob)

    replacements = map(zip(target_plugs_a, target_plugs_b)) do (plug_a, plug_b)
        ind(b; at=plug_b) => ind(a; at=plug_a)
    end

    if issetequal(first.(replacements), last.(replacements))
        return b
    end

    replace!(b, replacements)

    return a, b
end

align!((a, b)::P) where {P<:Pair} = align!(a, :outputs, b, :inputs)

"""
    @align! a => b reset=true

Rename in-place the indices of the input/output sites of two Pluggable Tensor Networks to be able to connect between them.
"""
macro align!(expr)
    @assert Meta.isexpr(expr, :call) && expr.args[1] == :(=>)
    Base.remove_linenums!(expr)
    a, b = expr.args[2:end]

    # @assert Meta.isexpr(reset, :(=)) && reset.args[1] == :reset

    @assert Meta.isexpr(a, :call)
    @assert Meta.isexpr(b, :call)
    ioa, ida = a.args
    iob, idb = b.args
    return quote
        align!($(esc(ida)), $(Meta.quot(ioa)), $(esc(idb)), $(Meta.quot(iob)))
        $(esc(idb))
    end
end

@deprecate reindex!(args...; kwargs...) align!(args...; kwargs...)

macro reindex!(args...)
    Base.depwarn("Macro @reindex! is deprecated, use @align! instead", :@align!)
    :(@reindex!($(args...)))
end

"""
    isconnectable(a, b)

Return `true` if two [Pluggable](@ref man-interface-pluggable) Tensor Networks can be connected. This means:

 1. The outputs of `a` are a superset of the inputs of `b`.
 2. The outputs of `a` and `b` are disjoint except for the sites that are connected.
"""
function isconnectable(a, b)
    plug.(plugs(a; set=:outputs)) ⊇ adjoint.(plug.(plugs(b; set=:inputs))) && isdisjoint(
        setdiff(plug.(plugs(a; set=:outputs)), adjoint.(plug.(plugs(b; set=:inputs)))),
        setdiff(plug.(plugs(b; set=:inputs)), adjoint.(plug.(plugs(b; set=:outputs)))),
    )
end
