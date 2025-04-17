using QuantumTags

"""
    CanonicalForm

Abstract type representing the canonical form trait of a Tensor Network.
"""
abstract type CanonicalForm end

Base.copy(x::CanonicalForm) = x

"""
    NonCanonical

[`CanonicalForm`](@ref) trait representing a Tensor Network in a non-canonical form.
"""
struct NonCanonical <: CanonicalForm end

"""
    MixedCanonical

[`CanonicalForm`](@ref) trait representing a Tensor Network in the mixed-canonical form.

  - The orthogonality center is a [`Site`](@ref) or a vector of [`Site`](@ref)s.
  - The tensors to the left and right of the orthogonality center are isommetries pointing towards the orthogonality center.
"""
struct MixedCanonical <: CanonicalForm
    orthog_center::Union{S,Vector{S}} where {S<:Site}
end

Base.copy(x::MixedCanonical) = MixedCanonical(copy(x.orthog_center))
Base.:(==)(a::MixedCanonical, b::MixedCanonical) = a.orthog_center == b.orthog_center

"""
    BondCanonical

[`CanonicalForm`](@ref) trait representing a Tensor Network in the bond-canonical form.

  - The orthogonality center is a [`Link`](@ref) or a vector of [`Link`](@ref)s.
  - The tensors to the left and right of the orthogonality center are isommetries pointing towards the orthogonality center.
"""
struct BondCanonical <: CanonicalForm
    orthog_center::Link
end

Base.copy(x::BondCanonical) = BondCanonical(copy(x.orthog_center))
Base.:(==)(a::BondCanonical, b::BondCanonical) = a.orthog_center == b.orthog_center

"""
    VidalGauge

[`CanonicalForm`](@ref) trait representing a Tensor Network in canonical form or Vidal gauge; i.e. the singular values matrix
``\\Lambda_i`` between each tensor ``\\Gamma_{i-1}`` and ``\\Gamma_i``.
"""
struct VidalGauge <: CanonicalForm end

struct VidalLambda{B} <: Site
    bond::B
end

"""
    form(tn)

Return the canonical form of the Tensor Network.
"""
function form end

form(::AbstractTensorNetwork) = NonCanonical()

"""
    canonize!(tn, form)

Transform an Tensor Network into a canonical [`CanonicalForm`](@ref).

See also: [`NonCanonical`](@ref), [`MixedCanonical`](@ref), [`Canonical`](@ref).
"""
function canonize! end

"""
    canonize(tn)

Like [`canonize!`](@ref), but returns a new Tensor Network instead of modifying the original one.
"""
canonize(tn::AbstractTensorNetwork, args...; kwargs...) = canonize!(deepcopy(tn), args...; kwargs...)

# canonize_site(tn::AbstractTensorNetwork, args...; kwargs...) = canonize_site!(deepcopy(tn), args...; kwargs...)

"""
    checkform(tn)

Check whether a Tensor Network fulfills the properties of the canonical form is in.
"""
function checkform end
