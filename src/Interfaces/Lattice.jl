using DelegatorTraits

"""
    Lattice

The `Lattice` interface defines the methods for working with an object with lattice-like structure.
"""
struct Lattice <: Interface end

# keyword-dispatching methods
function sites end
function bonds end

function site end
function bond end

# query methods
function all_sites end
function all_bonds end

function all_sites_iter end
function all_bonds_iter end

function hassite end
function hasbond end

function nsites end
function nbonds end

function site_at end
function bond_at end

# mutating methods
function setsite! end
function setbond! end
function unsetsite! end
function unsetbond! end

# implementation
# TODO doesn't this clash with `QuantumTags.sites`?
sites(tn; kwargs...) = sites(sort_nt(values(kwargs)), tn)
sites(::@NamedTuple{}, tn) = all_sites(tn)

# TODO maybe is good idea to have a function that returns the default comparer method
# e.g. `is_like_f(::Plug)` returns `is_plug_equal`... so `like` is a trait?
# TODO important: if we do that, `is_like_f` should be able to compose with parametric types of `Plug` and such
# sites(kwargs::NamedTuple{(:like)}, tn) = sites(tn; by=isequal, kwargs...)
# sites(kwargs::NamedTuple{(:by, :like)}, tn) = sites_like(kwargs.by, tn, kwargs.like)

# site(tn; kwargs...) = site(sort_nt(values(kwargs)), tn)
# site(kwargs::NamedTuple{(:at,)}, tn) = site_at(tn, kwargs.at)

# TODO maybe is good idea to have a function that returns the default comparer method
# e.g. `is_like_f(::Plug)` returns `is_plug_equal`... so `like` is a trait?
# TODO important: if we do that, `is_like_f` should be able to compose with parametric types of `Plug` and such
# site(kwargs::NamedTuple{(:like)}, tn) = site(tn; by=isequal, kwargs...)
# site(kwargs::NamedTuple{(:by, :like)}, tn) = site_like(kwargs.by, tn, kwargs.like)

## `all_sites`
all_sites(lattice) = all_sites(lattice, DelegatorTrait(Lattice(), lattice))
all_sites(lattice, ::DelegateToField) = all_sites(delegator(Lattice(), lattice))
all_sites(lattice, ::DontDelegate) = throw(MethodError(all_sites, (lattice,)))

## `all_bonds`
all_bonds(lattice) = all_bonds(lattice, DelegatorTrait(Lattice(), lattice))
all_bonds(lattice, ::DelegateToField) = all_bonds(delegator(Lattice(), lattice))
all_bonds(lattice, ::DontDelegate) = throw(MethodError(all_bonds, (lattice,)))

## `all_sites_iter`
all_sites_iter(lattice) = all_sites_iter(lattice, DelegatorTrait(Lattice(), lattice))
all_sites_iter(lattice, ::DelegateToField) = all_sites_iter(delegator(Lattice(), lattice))
all_sites_iter(lattice, ::DontDelegate) = throw(MethodError(all_sites_iter, (lattice,)))

## `all_bonds_iter`
all_bonds_iter(lattice) = all_bonds_iter(lattice, DelegatorTrait(Lattice(), lattice))
all_bonds_iter(lattice, ::DelegateToField) = all_bonds_iter(delegator(Lattice(), lattice))
all_bonds_iter(lattice, ::DontDelegate) = throw(MethodError(all_bonds_iter, (lattice,)))

## `hassite`
hassite(lattice, site) = hassite(lattice, site, DelegatorTrait(Lattice(), lattice))
hassite(lattice, site, ::DelegateToField) = hassite(delegator(Lattice(), lattice), site)
function hassite(lattice, site, ::DontDelegate)
    fallback(hassite)
    any(Base.Fix1(is_site_equal, site), all_sites_iter(lattice))
end

## `hasbond`
hasbond(lattice, bond) = hasbond(lattice, bond, DelegatorTrait(Lattice(), lattice))
hasbond(lattice, bond, ::DelegateToField) = hasbond(delegator(Lattice(), lattice), bond)
function hasbond(lattice, bond, ::DontDelegate)
    fallback(hasbond)
    any(Base.Fix1(is_bond_equal, bond), all_bonds_iter(lattice))
end

## `nsites`
nsites(lattice) = nsites(lattice, DelegatorTrait(Lattice(), lattice))
nsites(lattice, ::DelegateToField) = nsites(delegator(Lattice(), lattice))
function nsites(lattice, ::DontDelegate)
    fallback(nsites)
    all_sites(lattice) |> length
end

## `nbonds`
nbonds(lattice) = nbonds(lattice, DelegatorTrait(Lattice(), lattice))
nbonds(lattice, ::DelegateToField) = nbonds(delegator(Lattice(), lattice))
function nbonds(lattice, ::DontDelegate)
    fallback(nbonds)
    all_bond(lattice) |> length
end

## `site_at`
site_at(lattice, tag) = site_at(lattice, tag, DelegatorTrait(Lattice(), lattice))
site_at(lattice, tag, ::DelegateToField) = site_at(delegator(Lattice(), lattice), tag)
site_at(lattice, tag, ::DontDelegate) = throw(MethodError(site_at, (lattice, tag)))

## `bond_at`
bond_at(lattice, tag) = bond_at(lattice, tag, DelegatorTrait(Lattice(), lattice))
bond_at(lattice, tag, ::DelegateToField) = bond_at(delegator(Lattice(), lattice), tag)
bond_at(lattice, tag, ::DontDelegate) = throw(MethodError(bond_at, (lattice, tag)))

## `setsite!`
# TODO check that the site does not exist and that the tensor exists
#   hassite(tn, e.site) && throw(ArgumentError("Lattice already contains site $(e.site)"))
#   hastensor(tn, e.tensor) || throw(ArgumentError("Tensor $(e.tensor) does not exist in the lattice"))
setsite!(lattice, vertex, site) = setsite!(lattice, vertex, site, DelegatorTrait(Lattice(), lattice))
setsite!(lattice, vertex, site, ::DelegateToField) = setsite!(delegator(Lattice(), lattice), vertex, site)
setsite!(lattice, vertex, site, ::DontDelegate) = throw(MethodError(setsite!, (lattice, vertex, site)))

## `setbond!`
# TODO check that the bond does not exist and that the tensor exists
#   hasbond(tn, e.bond) && throw(ArgumentError("Lattice already contains bond $(e.bond)"))
#   hastensor(tn, e.tensor) || throw(ArgumentError("Tensor $(e.tensor) does not exist in the lattice"))
setbond!(lattice, edge, bond) = setbond!(lattice, edge, bond, DelegatorTrait(Lattice(), lattice))
setbond!(lattice, edge, bond, ::DelegateToField) = setbond!(delegator(Lattice(), lattice), edge, bond)
setbond!(lattice, edge, bond, ::DontDelegate) = thow(MethodError(setbond!, (lattice, edge, bond)))

## `unsetsite!`
# TODO check that the site exists
#   hassite(tn, e.site) || throw(ArgumentError("Lattice does not contain site $(e.site)"))
unsetsite!(lattice, site) = unsetsite!(lattice, site, DelegatorTrait(Lattice(), lattice))
unsetsite!(lattice, site, ::DelegateToField) = unsetsite!(delegator(Lattice(), lattice), site)
unsetsite!(lattice, site, ::DontDelegate) = throw(MethodError(unsetsite!, (lattice, site)))

## `unsetbond!`
# TODO check that the bond exists
#   hasbond(tn, e.bond) || throw(ArgumentError("Lattice does not contain bond $(e.bond)"))
unsetbond!(lattice, bond) = unsetbond!(lattice, bond, DelegatorTrait(Lattice(), lattice))
unsetbond!(lattice, bond, ::DelegateToField) = unsetbond!(delegator(Lattice(), lattice), bond)
unsetbond!(lattice, bond, ::DontDelegate) = throw(MethodError(unsetbond!, (lattice, bond)))
