module TenetCoreGraphsExt

using TenetCore
using Graphs: Graphs
using ArgCheck

"""
    Graphs.neighbors(tn::AbstractTensorNetwork, tensor; open=true)

Return the neighboring [`Tensor`](@ref)s of `tensor` in the Tensor Network.
If `open=true`, the `tensor` itself is not included in the result.
"""
function Graphs.neighbors(tn::TenetCore.AbstractTensorNetwork, tensor::Tensor; open::Bool=true)
    @argcheck hastensor(tn, tensor) "Tensor not found in TensorNetwork"
    neigh_tensors = mapreduce(∪, inds(tensor)) do index
        tensors(tn; intersects=index)
    end
    open && filter!(x -> x !== tensor, neigh_tensors)
    return neigh_tensors
end

"""
    Graphs.neighbors(tn::AbstractTensorNetwork, ind; open=true)

Return the neighboring indices of `ind` in the Tensor Network.
If `open=true`, the `ind` itself is not included in the result.
"""
function Graphs.neighbors(tn::TenetCore.AbstractTensorNetwork, i::Index; open::Bool=true)
    @argcheck i ∈ tn "Index $i not found in TensorNetwork"
    neigh_inds = mapreduce(inds, ∪, tensors(tn; intersects=i))
    open && filter(x -> x !== i, neigh_inds)
    return neigh_inds
end

end
