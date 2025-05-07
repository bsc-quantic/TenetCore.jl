module TenetCoreFiniteDifferencesExt

using TenetCore
using FiniteDifferences

# TODO try avoid the `copy(tn)` to not have the original `tn` dangling around
function FiniteDifferences.to_vec(tn::TenetCore.AbstractTensorNetwork)
    tn_vec, back = to_vec(tensors(tn))
    function TensorNetwork_from_vec(v)
        new_tn = copy(tn)
        for (old_tensor, new_tensor) in zip(tensors(new_tn), back(v))
            replace_tensor!(new_tn, old_tensor, new_tensor)
        end
        return new_tn
    end

    return tn_vec, TensorNetwork_from_vec
end

end
