# TODO document that a `TensorNetwork` implementor must implement these methods
struct UnsafeScopeable <: Interface end

# interface
function is_scopeable end
function get_unsafe_scope end
function set_unsafe_scope! end
function checksizes end
function inscope end

function is_scopeable(tn::T) where {T}
    if delegates(UnsafeScopeable(), tn) isa DelegateTo
        true
        # elseif hasmethod(get_unsafe_scope, Tuple{T}) && hasmethod(set_unsafe_scope!, Tuple{T,UnsafeScope})
        #     true
    else
        false
    end
end

get_unsafe_scope(tn) = get_unsafe_scope(tn, delegates(UnsafeScopeable(), tn))
get_unsafe_scope(tn, ::DelegateTo) = get_unsafe_scope(tn, delegate(UnsafeScopeable(), tn))
get_unsafe_scope(_, ::DontDelegate) = nothing

set_unsafe_scope!(tn, uc) = set_unsafe_scope!(tn, uc, delegates(UnsafeScopeable(), tn))
set_unsafe_scope!(tn, uc, ::DelegateTo) = set_unsafe_scope!(tn, delegate(UnsafeScopeable(), tn), uc)
set_unsafe_scope!(tn, uc, ::DontDelegate) = throw(MethodError(set_unsafe_scope!, (tn, uc)))

checksizes(tn) = checksizes(tn, delegates(UnsafeScopeable(), tn))
checksizes(tn, ::DelegateTo) = checksizes(tn, delegate(UnsafeScopeable(), tn))
function checksizes(tn, ::DontDelegate)
    @debug "Falling back to default `checksizes` method"
    sizedict = size(tn)
    return all(tensors(tn)) do tensor
        return all(enumerate(inds(tensor))) do (i, ind)
            size(tensor, ind) == sizedict[ind] == size(tensor, i)
        end
    end
end

# UnsafeScope
struct UnsafeScope
    refs::Vector{WeakRef}

    UnsafeScope() = new(Vector{WeakRef}())
end

# aliases
Base.values(uc::UnsafeScope) = map(x -> x.value, uc.refs)
Base.push!(uc::UnsafeScope, ref::WeakRef) = push!(uc.refs, ref)
Base.push!(uc::UnsafeScope, tn) = push!(uc.refs, WeakRef(tn))

inscope(tn, uc::UnsafeScope) = tn ∈ uc.refs
inscope(tn, ::Nothing) = false

isscoped(tn) = !isnothing(get_unsafe_scope(tn))

macro unsafe_region(tn, block)
    return esc(
        quote
            local old = copy($tn)

            # Create a new UnsafeScope and set it to the current tn
            local _uc = $UnsafeScope()
            $set_unsafe_scope!($tn, _uc)

            # Register the tensor network in the UnsafeScope
            push!($get_unsafe_scope($tn).refs, WeakRef($tn))

            e = nothing
            try
                $block # Execute the user-provided block
            catch e
                $tn = old # Restore the original tensor network in case of an exception
                rethrow(e)
            finally
                if isnothing(e)
                    # Perform checks of registered tensor networks
                    for ref in values($get_unsafe_scope($tn))
                        if !isnothing(ref) && ref ∈ $get_unsafe_scope($tn).refs
                            if !$checksizes(ref)
                                $tn = old

                                # Set `unsafe` field to `nothing`
                                $set_unsafe_scope!($tn, nothing)

                                throw(DimensionMismatch("Inconsistent size of indices"))
                            end
                        end
                    end
                end
            end
        end,
    )
end
