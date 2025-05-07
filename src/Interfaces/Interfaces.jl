abstract type Interface end

# It might not be "exactly" like the delegator pattern, because our "delegatee"
# doesn't know that it is being delegated to, but it is a good name for now.
abstract type DelegatorTrait end
struct DontDelegate <: DelegatorTrait end
struct DelegateTo{X} <: DelegatorTrait end

delegates(interface, x) = DontDelegate()

delegate(interface, x) = delegate(interface, x, delegates(interface, x))
delegate(interface, _, ::DontDelegate) = throw(ArgumentError("Cannot delegate to $interface"))
delegate(interface, x, ::DelegateTo{F}) where {F} = getproperty(x, F)

abstract type InterfaceTrait end
struct NotImplements <: InterfaceTrait end
struct Implements <: InterfaceTrait end

function implements(interface, x)
    if delegates(interface, x) isa DelegateTo
        return implements(interface, delegate(interface, x))
    else
        return NotImplements()
    end
end

# NOTE for developers
# try using functions owned by us (e.g. `mysize` instead of `Base.size`)
# aliases to `Base` come after and can be removed if problematic

include("Effects.jl")
include("UnsafeScope.jl")
include("TensorNetwork.jl")
include("Taggable.jl")
# include("Lattice.jl")
include("Pluggable.jl")
include("Attributeable.jl")
