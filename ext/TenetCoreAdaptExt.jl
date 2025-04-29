module TenetCoreAdaptExt

using TenetCore
using Adapt

# TODO move to Muscle
# Adapt.adapt_structure(to, x::Tensor) = Tensor(adapt(to, parent(x)), inds(x))

function Adapt.adapt_structure(to, x::AbstractTensorNetwork)
    y = copy(x)

    for tensor in all_tensors(y)
        replace_tensor!(y, tensor => adapt(to, tensor))
    end

    return y
end

end
