# TODO document that a `TensorNetwork` implementor must implement these methods
struct UnsafeScopeable <: Interface end

# interface
function get_unsafe_scope end
function set_unsafe_scope! end
function checksizes end

get_unsafe_scope(tn) = get_unsafe_scope(delegates(UnsafeScopeable(), tn), tn)
get_unsafe_scope(::DontDelegate, tn) = throw(MethodError(get_unsafe_scope, (typeof(tn),)))
get_unsafe_scope(::DelegateTo, tn) = get_unsafe_scope(tn, delegate(UnsafeScopeable(), tn))

set_unsafe_scope!(tn, uc) = set_unsafe_scope!(delegates(UnsafeScopeable(), tn), tn, uc)
set_unsafe_scope!(::DontDelegate, _, _) = throw(MethodError(set_unsafe_scope!, (typeof(tn), typeof(uc))))
set_unsafe_scope!(::DelegateTo, tn, uc) = set_unsafe_scope!(tn, delegate(UnsafeScopeable(), tn), uc)

checksizes(tn) = checksizes(tn, delegates(UnsafeScopeable(), tn))
checksizes(::DelegateTo, tn) = checksizes(tn, delegate(UnsafeScopeable(), tn))
function checksizes(::DontDelegate, tn)
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

isscoped(tn) = inscope(tn, get_unsafe_scope(tn))

macro unsafe_region(tn, block)
    return esc(
        quote
            local old = copy($tn)

            # Create a new UnsafeScope and set it to the current tn
            local _uc = Tenet.UnsafeScope()
            Tenet.set_unsafe_scope!($tn, _uc)

            # Register the tensor network in the UnsafeScope
            push!(Tenet.get_unsafe_scope($tn).refs, WeakRef($tn))

            e = nothing
            try
                $block # Execute the user-provided block
            catch e
                $tn = old # Restore the original tensor network in case of an exception
                rethrow(e)
            finally
                if isnothing(e)
                    # Perform checks of registered tensor networks
                    for ref in values(Tenet.get_unsafe_scope($tn))
                        if !isnothing(ref) && ref ∈ Tenet.get_unsafe_scope($tn).refs
                            if !Tenet.checksizes(ref)
                                $tn = old

                                # Set `unsafe` field to `nothing`
                                Tenet.set_unsafe_scope!($tn, nothing)

                                throw(DimensionMismatch("Inconsistent size of indices"))
                            end
                        end
                    end
                end
            end
        end,
    )
end
