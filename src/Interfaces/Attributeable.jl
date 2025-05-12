# interface object
"""
    Attributeable <: Interface
"""
struct Attributeable <: Interface end

# dispatching methods
function attrs end
function getattr end
function setattr! end
function hasattr end

# query methods
function attrs_global end
function attrs_tensor end
function attrs_ind end

# function attrkeys_global end
# function attrkeys_tensor end
# function attrkeys_ind end

function getattr_global end
function getattr_tensor end
function getattr_ind end

function hasattr_global end
function hasattr_tensor end
function hasattr_ind end

# mutating methods
function setattr_global_inner! end
function setattr_tensor_inner! end
function setattr_ind_inner! end

function setattr_global! end
function setattr_tensor! end
function setattr_ind! end

# implementation
## `attrs`
attrs(tn) = attrs_global(tn)
attrs(tn, tensor::Tensor) = attrs_tensor(tn, tensor)
attrs(tn, ind::Index) = attrs_ind(tn, ind)

attrs(tn, site::Site) = attrs(tn, tensor_at(tn, site))
attrs(tn, link::Link) = attrs(tn, ind_at(tn, link))

## `getattr`
getattr(tn, key) = getattr_global(tn, key)
getattr(tn, tensor::Tensor, key) = getattr_tensor(tn, tensor, key)
getattr(tn, ind::Index, key) = getattr_ind(tn, ind, key)

getattr(tn, site::Site, key) = getattr(tn, tensor_at(tn, site), key)
getattr(tn, link::Link, key) = getattr(tn, ind_at(tn, link), key)

getattr(tn, key, default) = hasattr(tn, key) ? getattr(tn, key) : default
getattr(tn, tensor::Tensor, key, default) = hasattr(tn, tensor, key) ? getattr(tn, tensor, key) : default
getattr(tn, ind::Index, key, default) = hasattr(tn, ind, key) ? getattr(tn, ind, key) : default

getattr(tn, site::Site, key, default) = getattr(tn, tensor_at(tn, site), key, default)
getattr(tn, link::Link, key, default) = getattr(tn, ind_at(tn, link), key, default)

## `setattr!`
setattr!(tn, key, value) = setattr_global!(tn, key, value)
setattr!(tn, tensor::Tensor, key, value) = setattr_tensor!(tn, tensor, key, value)
setattr!(tn, ind::Index, key, value) = setattr_ind!(tn, ind, key, value)

setattr!(tn, site::Site, key, value) = setattr!(tn, tensor_at(tn, site), key, value)
setattr!(tn, link::Link, key, value) = setattr!(tn, ind_at(tn, link), key, value)

## `hasattr`
hasattr(tn, key) = hasattr_global(tn, key)
hasattr(tn, tensor::Tensor, key) = hasattr_tensor(tn, tensor, key)
hasattr(tn, ind::Index, key) = hasattr_tensor(tn, ind, key)

hasattr(tn, site::Site, key) = hasattr(tn, tensor_at(tn, site), key)
hasattr(tn, link::Link, key) = hasattr(tn, ind_at(tn, link), key)

## `attrs_global`
attrs_global(tn) = attrs_global(tn, DelegatorTrait(Attributeable(), tn))
attrs_global(tn, ::DelegateTo) = attrs_global(delagate(Attributeable(), tn))
attrs_global(tn, ::DontDelegate) = throw(MethodError(attrs_global, (tn,)))

## `attrs_tensor`
attrs_tensor(tn, tensor) = attrs_tensor(tn, tensor, DelegatorTrait(Attributeable(), tn))
attrs_tensor(tn, tensor, ::DelegateTo) = attrs_tensor(delegator(Attributeable(), tn), tensor)
attrs_tensor(tn, tensor, ::DontDelegate) = throw(MethodError(attrs_tensor, (tn, tensor)))

## `attrs_ind`
attrs_ind(tn, ind) = attrs_ind(tn, ind, DelegatorTrait(Attributeable(), tn))
attrs_ind(tn, ind, ::DelegateTo) = attrs_ind(delegator(Attributeable(), tn), ind)
attrs_ind(tn, ind, ::DontDelegate) = throw(MethodError(attrs_ind, (tn, ind)))

## `getattr_global`
getattr_global(tn, key) = getattr_global(tn, key, DelegatorTrait(Attributeable(), tn))
getattr_global(tn, key, ::DelegateTo) = getattr_global(delegator(Attributeable(), tn), key)
function getattr_global(tn, key, ::DontDelegate)
    @debug "Falling back to default implementation of `getattr_global`"
    return getindex(attrs_global(tn), key)
end

## `getattr_tensor`
getattr_tensor(tn, tensor, key) = getattr_tensor(tn, tensor, key, DelegatorTrait(Attributeable(), tn))
getattr_tensor(tn, tensor, key, ::DelegateTo) = getattr_tensor(delegator(Attributeable(), tn), tensor, key)
function getattr_tensor(tn, tensor, key, ::DontDelegate)
    @debug "Falling back to default implementation of `getattr_tensor`"
    return getindex(attrs_tensor(tn, tensor), key)
end

## `getattr_ind`
getattr_ind(tn, ind, key) = getattr_ind(tn, ind, key, DelegatorTrait(Attributeable(), tn))
getattr_ind(tn, ind, key, ::DelegateTo) = getattr_ind(delegator(Attributeable(), tn), ind, key)
function getattr_ind(tn, ind, key, ::DontDelegate)
    @debug "Falling back to default implementation of `getattr_ind`"
    return getindex(attrs_ind(tn, ind), key)
end

## `hasattr_global`
hasattr_global(tn, key) = hasattr_global(tn, key, DelegatorTrait(Attributeable(), tn))
hasattr_global(tn, key, ::DelegateTo) = hasattr_global(delegator(Attributeable(), tn), key)
function hasattr_global(tn, key, ::DontDelegate)
    @debug "Falling back to default implementation of `hasattr_global`"
    return haskey(attrs_global(tn), key)
end

## `hasattr_tensor`
hasattr_tensor(tn, tensor, key) = hasattr_tensor(tn, tensor, key, DelegatorTrait(Attributeable(), tn))
hasattr_tensor(tn, tensor, key, ::DelegateTo) = hasattr_tensor(delegator(Attributeable(), tn), tensor, key)
function hasattr_tensor(tn, tensor, key, ::DontDelegate)
    @debug "Falling back to default implementation of `hasattr_tensor`"
    return haskey(attrs_tensor(tn, tensor), key)
end

## `hasattr_ind`
hasattr_ind(tn, ind, key) = hasattr_ind(tn, ind, key, DelegatorTrait(Attributeable(), tn))
hasattr_ind(tn, ind, key, ::DelegateTo) = hasattr_ind(delegator(Attributeable(), tn), ind, key)
function hasattr_ind(tn, ind, key, ::DontDelegate)
    @debug "Falling back to default implementation of `hasattr_ind`"
    return haskey(attrs_ind(tn, ind), key)
end

## `setattr_global_inner!`
setattr_global_inner!(tn, key, value) = setattr_global_inner!(tn, key, value, DelegatorTrait(Attributeable(), tn))
setattr_global_inner!(tn, key, value, ::DelegateTo) = setattr_global_inner!(delegator(Attributeable(), tn), key, value)
setattr_global_inner!(tn, key, value, ::DontDelegate) = throw(MethodError(setattr_global_inner!, (tn, key, value)))

## `setattr_tensor_inner!`
function setattr_tensor_inner!(tn, tensor, key, value)
    setattr_tensor_inner!(tn, tensor, key, value, DelegatorTrait(Attributeable(), tn))
end
function setattr_tensor_inner!(tn, tensor, key, value, ::DelegateTo)
    setattr_tensor_inner!(delegator(Attributeable(), tn), tensor, key, value)
end
function setattr_tensor_inner!(tn, tensor, key, value, ::DontDelegate)
    throw(MethodError(setattr_tensor_inner!, (tn, tensor, key, value)))
end

## `setattr_ind_inner!`
setattr_ind_inner!(tn, ind, key, value) = setattr_ind_inner!(tn, ind, key, value, DelegatorTrait(Attributeable(), tn))
function setattr_ind_inner!(tn, ind, key, value, ::DelegateTo)
    setattr_ind_inner!(delegator(Attributeable(), tn), ind, key, value)
end
setattr_ind_inner!(tn, ind, key, value, ::DontDelegate) = throw(MethodError(setattr_ind_inner!, (tn, ind, key, value)))

## `setattr_global!`
struct SetAttrGlobal{T} <: Effect
    key::Symbol
    value::T
end

function setattr_global!(tn, key, value)
    checkeffect(tn, SetAttrGlobal(key, value))
    setattr_global_inner!(tn, key, value)
    handle!(tn, SetAttrGlobal(key, value))
    return tn
end

checkeffect(tn, e::SetAttrGlobal) = checkeffect(tn, e, DelegatorTrait(Attributeable(), tn))
checkeffect(tn, e::SetAttrGlobal, ::DelegateTo) = checkeffect(delegator(Attributeable(), tn), e)
checkeffect(tn, e::SetAttrGlobal, ::DontDelegate) = nothing

handle!(tn, e::SetAttrGlobal) = handle!(tn, e, DelegatorTrait(Attributeable(), tn))
handle!(tn, e::SetAttrGlobal, ::DelegateTo) = handle!(delegator(Attributeable(), tn), e)
handle!(tn, e::SetAttrGlobal, ::DontDelegate) = nothing

## `setattr_tensor!`
struct SetAttrTensor{T} <: Effect
    tensor::Tensor
    key::Symbol
    value::T
end

function setattr_tensor!(tn, tensor, key, value)
    checkeffect(tn, SetAttrTensor(tensor, key, value))
    setattr_tensor_inner!(tn, tensor, key, value)
    handle!(tn, SetAttrTensor(tensor, key, value))
    return tn
end

checkeffect(tn, e::SetAttrTensor) = checkeffect(tn, e, DelegatorTrait(Attributeable(), tn))
checkeffect(tn, e::SetAttrTensor, ::DelegateTo) = checkeffect(delegator(Attributeable(), tn), e)
function checkeffect(tn, e::SetAttrTensor, ::DontDelegate)
    hastensor(tn, e.tensor) || throw(ArgumentError("Tensor $(e.tensor) not in tensor network"))
end

handle!(tn, e::SetAttrTensor) = handle!(tn, e, DelegatorTrait(Attributeable(), tn))
handle!(tn, e::SetAttrTensor, ::DelegateTo) = handle!(delegator(Attributeable(), tn), e)
handle!(tn, e::SetAttrTensor, ::DontDelegate) = nothing

## `setattr_ind!`
struct SetAttrInd{T} <: Effect
    ind::Index
    key::Symbol
    value::T
end

function setattr_ind!(tn, ind, key, value)
    checkeffect(tn, SetAttrInd(ind, key, value))
    setattr_ind_inner!(tn, ind, key, value)
    handle!(tn, SetAttrInd(ind, key, value))
    return tn
end

checkeffect(tn, e::SetAttrInd) = checkeffect(tn, e, DelegatorTrait(Attributeable(), tn))
checkeffect(tn, e::SetAttrInd, ::DelegateTo) = checkeffect(delegator(Attributeable(), tn), e)
function checkeffect(tn, e::SetAttrInd, ::DontDelegate)
    hasind(tn, e.ind) || throw(ArgumentError("Index $(e.ind) not in tensor network"))
end

handle!(tn, e::SetAttrInd) = handle!(tn, e, DelegatorTrait(Attributeable(), tn))
handle!(tn, e::SetAttrInd, ::DelegateTo) = handle!(delegator(Attributeable(), tn), e)
handle!(tn, e::SetAttrInd, ::DontDelegate) = nothing
