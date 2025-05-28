module TenetCoreReactantExt

using TenetCore
using Reactant
using Reactant: Enzyme

# issue fix: strange behavior between `IdDict` and `Tensor` when using `TracedRArray`... can't figure out why or do a MWE
function TenetCore.hastensor(tn::SimpleTensorNetwork, tensor::Tensor{<:Reactant.TracedRNumber})
    any(t -> tensor === t, all_tensors_iter(tn))
end

function TenetCore.tensor_vertex(tn::SimpleTensorNetwork, tensor::Tensor{<:Reactant.TracedRNumber})
    for (vertex, mapped_tensor) in tn.tensormap
        if mapped_tensor === tensor
            return vertex
        end
    end

    throw(ErrorException("Tensor $tensor not found in tensor network but `checkeffect` should have stopped this!"))
end

# we specify `mode` and `track_numbers` types due to ambiguity
Base.@nospecializeinfer function Reactant.traced_type_inner(
    @nospecialize(T::Type{<:TenetCore.AbstractTensorNetwork}),
    seen,
    mode::Reactant.TraceMode,
    @nospecialize(track_numbers::Type),
    args...,
)
    return T
end

# TODO replace `tensor_id` for `Vertex{UUID}`?
function Reactant.Compiler.make_tracer(seen, prev::SimpleTensorNetwork, @nospecialize(path), mode; kwargs...)
    traced_tn = copy(prev)
    for (i, tensor) in enumerate(all_tensors(traced_tn))
        traced_tensor = Reactant.Compiler.make_tracer(
            seen, tensor, Reactant.append_path(path, (; tensor_id=i)), mode; kwargs...
        )

        # seems like in some tracing modes, the traced tensor is the same as the original tensor
        if tensor !== traced_tensor
            replace_tensor!(traced_tn, tensor, traced_tensor)
        end
    end
    return traced_tn
end

# requires a specialization due to default `create_result` getting confused with `CachedField`
function Reactant.Compiler.create_result(
    tocopy::SimpleTensorNetwork,
    @nospecialize(path),
    result_stores,
    path_to_shard_info,
    to_unreshard_results,
    unresharded_code::Vector{Expr},
    unresharded_arrays_cache,
    used_shardinfo,
    result_cache,
    var_idx,
    resultgen_code,
)
    expr_network = Reactant.Compiler.create_result(
        tocopy.network,
        Reactant.append_path(path, :network),
        result_stores,
        path_to_shard_info,
        to_unreshard_results,
        unresharded_code,
        unresharded_arrays_cache,
        used_shardinfo,
        result_cache,
        var_idx,
        resultgen_code,
    )

    expr_indmap = Reactant.Compiler.create_result(
        tocopy.indmap,
        Reactant.append_path(path, :indmap),
        result_stores,
        path_to_shard_info,
        to_unreshard_results,
        unresharded_code,
        unresharded_arrays_cache,
        used_shardinfo,
        result_cache,
        var_idx,
        resultgen_code,
    )

    # `tensormap` requires special treatment due to the `path` used to store the tensors
    # TODO refactor the way we mark the path of `tensors` in a `AbstractTensorNetwork`
    tensormap_results = map(enumerate(tocopy.tensormap)) do (i, (vertex, tensor))
        :(
            $vertex => $(Reactant.Compiler.create_result(
                tensor,
                Reactant.append_path(path, (; tensor_id=i)),
                result_stores,
                path_to_shard_info,
                to_unreshard_results,
                unresharded_code,
                unresharded_arrays_cache,
                used_shardinfo,
                result_cache,
                var_idx,
                resultgen_code,
            ))
        )
    end
    expr_tensormap = :($(typeof(tocopy.tensormap))([$(tensormap_results...)]))

    return quote
        network = $expr_network
        tensormap = $expr_tensormap
        indmap = $expr_indmap
        $SimpleTensorNetwork(network, tensormap, indmap)
    end
end

Reactant.traced_getfield(x::SimpleTensorNetwork, i::Int) = all_tensors(x)[i]
Reactant.traced_getfield(x::SimpleTensorNetwork, fld::@NamedTuple{tensor_id::Int}) = all_tensors(x)[fld.tensor_id]

function Reactant.TracedUtils.push_val!(ad_inputs, x::SimpleTensorNetwork, path)
    @assert length(path) == 2
    @assert path[2] === :data

    x = parent(tensors(x)[path[1].tensor_id]).mlir_data

    return push!(ad_inputs, x)
end

function Reactant.TracedUtils.set!(x::SimpleTensorNetwork, path, tostore; emptypath=false)
    @assert length(path) == 2
    @assert path[2] === :data

    x = parent(tensors(x)[path[1].tensor_id])
    x.mlir_data = tostore

    if emptypath
        x.paths = ()
    end
end

function Reactant.set_act!(inp::Enzyme.Annotation{SimpleTensorNetwork}, path, reverse, tostore; emptypath=false)
    @assert length(path) == 2
    @assert path[2] === :data

    x = if inp isa Enzyme.Active
        inp.val
    else
        inp.dval
    end

    x = parent(tensors(x)[path[1].tensor_id])
    x.mlir_data = tostore

    if emptypath
        x.paths = ()
    end
end

end
