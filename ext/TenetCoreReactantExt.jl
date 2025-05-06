module TenetCoreReactantExt

using TenetCore
using Reactant
using Reactant: Enzyme

# we specify `mode` and `track_numbers` types due to ambiguity
Base.@nospecializeinfer function Reactant.traced_type_inner(
    @nospecialize(T::Type{<:TenetCore.AbstractTensorNetwork}),
    seen,
    mode::Reactant.TraceMode,
    @nospecialize(track_numbers::Type)
)
    return T
end

function Reactant.Compiler.make_tracer(
    seen, prev::TenetCore.AbstractTensorNetwork, @nospecialize(path), mode; kwargs...
)
    traced_tn = copy(prev)
    for (i, tensor) in enumerate(all_tensors(prev))
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

# function Reactant.Compiler.create_result(tocopy::TenetCore.AbstractTensorNetwork, @nospecialize(path), result_stores)
#     elems = map(1:ntensors(tocopy)) do i
#         Reactant.create_result(tensors(tocopy)[i], Reactant.append_path(path, i), result_stores)
#     end
#     return :($TensorNetwork([$(elems...)]))
# end

Reactant.traced_getfield(x::TenetCore.AbstractTensorNetwork, i::Int) = all_tensors(x)[i]
function Reactant.traced_getfield(x::TenetCore.AbstractTensorNetwork, fld::@NamedTuple{tensor_id::Int})
    all_tensors(x)[fld.tensor_id]
end

function Reactant.TracedUtils.push_val!(ad_inputs, x::TenetCore.AbstractTensorNetwork, path)
    @assert length(path) == 2
    @assert path[2] === :data

    x = parent(tensors(x)[path[1].tensor_id]).mlir_data

    return push!(ad_inputs, x)
end

function Reactant.TracedUtils.set!(x::TenetCore.AbstractTensorNetwork, path, tostore; emptypath=false)
    @assert length(path) == 2
    @assert path[2] === :data

    x = parent(tensors(x)[path[1].tensor_id])
    x.mlir_data = tostore

    if emptypath
        x.paths = ()
    end
end

function Reactant.set_act!(
    inp::Enzyme.Annotation{TenetCore.AbstractTensorNetwork}, path, reverse, tostore; emptypath=false
)
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
