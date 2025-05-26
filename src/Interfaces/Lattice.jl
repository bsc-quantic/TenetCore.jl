struct Lattice <: Interface end

# keyword-dispatching methods
function sites end
function bonds end

function site end
function bond end

# query methods
function all_sites end
function all_bonds end

function hassite end
function hasbond end

function nsites end
function nbonds end

function tensor_at_site end
function ind_at_bond end

function site_at_tensor end
function bond_at_ind end

# mutating methods
function setsite! end
function setsite_inner! end

function setbond! end
function setbond_inner! end

function unsetsite! end
function unsetsite_inner! end

function unsetbond! end
function unsetbond_inner! end

# effects
"""
    SetSiteEffect{Tag,Obj} <: Effect

Represents the effect of setting a mapping between a `Tag` and an `Obj`ect.
"""
struct SetSiteEffect{T,O} <: Effect
    site::T
    obj::O
end

SetSiteEffect(site::T, @nospecialize(obj::Tensor)) where {T} = SetSiteEffect{T,Tensor}(site, obj)
SetSiteEffect(site::T, @nospecialize(obj::Index)) where {T} = SetSiteEffect{T,Index}(site, obj)

"""
    SetBondEffect{Tag,Obj} <: Effect

Represents the effect of setting a mapping between a `Tag` and an `Obj`ect.
"""
struct SetBondEffect{T,O} <: Effect
    bond::T
    obj::O
end

SetBondEffect(bond::T, @nospecialize(obj::Tensor)) where {T} = SetBondEffect{T,Tensor}(bond, obj)
SetBondEffect(bond::T, @nospecialize(obj::Index)) where {T} = SetBondEffect{T,Index}(bond, obj)

"""
    UnsetSiteEffect{Tag} <: Effect

Represents the effect of unsetting a mapping of a `Site` `Tag`.
"""
struct UnsetSiteEffect{T} <: Effect
    site::T
end

"""
    UnsetBondEffect{Tag} <: Effect

Represents the effect of unsetting a mapping of a `Bond` `Tag`.
"""
struct UnsetBondEffect{T} <: Effect
    site::T
end

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
all_sites(lattice, ::DelegateTo) = all_sites(delegator(Lattice(), lattice))
all_sites(lattice, ::DontDelegate) = error("") # filter(issite, )

## `all_bonds`
all_bonds(lattice) = all_bonds(lattice, DelegatorTrait(Lattice(), lattice))
all_bonds(lattice, ::DelegateTo) = all_bonds(delegator(Lattice(), lattice))
all_bonds(lattice, ::DontDelegate) = error("") # filter(isbond, )

## `hassite`
hassite(lattice, site) = hassite(lattice, site, DelegatorTrait(Lattice(), lattice))
hassite(lattice, site, ::DelegateTo) = hassite(delegator(Lattice(), lattice), site)
function hassite(lattice, site, ::DontDelegate)
    fallback(hassite)
    all_sites(lattice) |> any(Base.Fix1(is_site_equal, site))
end

## `hasbond`
hasbond(lattice, bond) = hasbond(lattice, bond, DelegatorTrait(Lattice(), lattice))
hasbond(lattice, bond, ::DelegateTo) = hasbond(delegator(Lattice(), lattice), bond)
function hasbond(lattice, bond, ::DontDelegate)
    fallback(hasbond)
    all_bonds(lattice) |> any(Base.Fix1(is_bond_equal, bond))
end

## `nsites`
nsites(lattice) = nsites(lattice, DelegatorTrait(Lattice(), lattice))
nsites(lattice, ::DelegateTo) = nsites(delegator(Lattice(), lattice))
function nsites(lattice, ::DontDelegate)
    fallback(nsites)
    all_sites(lattice) |> length
end

## `nbonds`
nbonds(lattice) = nbonds(lattice, DelegatorTrait(Lattice(), lattice))
nbonds(lattice, ::DelegateTo) = nbonds(delegator(Lattice(), lattice))
function nbonds(lattice, ::DontDelegate)
    fallback(nbonds)
    all_bonds(lattice) |> length
end

## `tensor_at_site`
tensor_at_site(lattice, site) = tensor_at_site(lattice, site, DelegatorTrait(Lattice(), lattice))
tensor_at_site(lattice, site, ::DelegateTo) = tensor_at_site(delegator(Lattice(), lattice), site)
tensor_at_site(lattice, site, ::DontDelegate) = throw(MethodError(tensor_at_site, (lattice, site)))

## `ind_at_bond`
ind_at_bond(lattice, bond) = ind_at_bond(lattice, bond, DelegatorTrait(Lattice(), lattice))
ind_at_bond(lattice, bond, ::DelegateTo) = ind_at_bond(delegator(Lattice(), lattice), bond)
ind_at_bond(lattice, bond, ::DontDelegate) = throw(MethodError(ind_at_bond, (lattice, bond)))

## `site_at_tensor`
site_at_tensor(lattice, tensor) = site_at_tensor(lattice, tensor, DelegatorTrait(Lattice(), lattice))
site_at_tensor(lattice, tensor, ::DelegateTo) = site_at_tensor(delegator(Lattice(), lattice), tensor)
site_at_tensor(lattice, tensor, ::DontDelegate) = throw(MethodError(site_at_tensor, (lattice, tensor)))

## `bond_at_ind`
bond_at_ind(lattice, ind) = bond_at_ind(lattice, ind, DelegatorTrait(Lattice(), lattice))
bond_at_ind(lattice, ind, ::DelegateTo) = bond_at_ind(delegator(Lattice(), lattice), ind)
bond_at_ind(lattice, ind, ::DontDelegate) = throw(MethodError(bond_at_ind, (lattice, ind)))

## `setsite!`
function setsite!(tn, tensor, site)
    checkeffect(tn, SetEffect(tensor, site))
    setsite_inner!(tn, tensor, site)
    handle!(tn, SetEffect(tensor, site))
    return tn
end

checkeffect(tn, @nospecialize(e::SetSiteEffect)) = checkeffect(tn, e, DelegatorTrait(Lattice(), tn))
checkeffect(tn, e::SetSiteEffect, ::DelegateTo) = checkeffect(delegator(Lattice(), tn), e)
function checkeffect(tn, e::SetSiteEffect, ::DontDelegate)
    hassite(tn, e.site) && throw(ArgumentError("Lattice already contains site $(e.site)"))
    hastensor(tn, e.tensor) || throw(ArgumentError("Tensor $(e.tensor) does not exist in the lattice"))
end

## `setsite_inner!`
setsite_inner!(lattice, site) = setsite_inner!(lattice, site, DelegatorTrait(Lattice(), lattice))
setsite_inner!(lattice, site, ::DelegateTo) = setsite_inner!(delegator(Lattice(), lattice), site)
setsite_inner!(lattice, site, ::DontDelegate) = error("")

## `setbond!`
function setbond!(tn, tensor, bond)
    checkeffect(tn, SetEffect(tensor, bond))
    setbond_inner!(tn, tensor, bond)
    handle!(tn, SetEffect(tensor, bond))
    return tn
end

checkeffect(tn, @nospecialize(e::SetBondEffect)) = checkeffect(tn, e, DelegatorTrait(Lattice(), tn))
checkeffect(tn, e::SetBondEffect, ::DelegateTo) = checkeffect(delegator(Lattice(), tn), e)
function checkeffect(tn, e::SetBondEffect, ::DontDelegate)
    hasbond(tn, e.bond) && throw(ArgumentError("Lattice already contains bond $(e.bond)"))
    hastensor(tn, e.tensor) || throw(ArgumentError("Tensor $(e.tensor) does not exist in the lattice"))
end

## `setbond_inner!`
setbond_inner!(lattice, bond) = setbond_inner!(lattice, bond, DelegatorTrait(Lattice(), lattice))
setbond_inner!(lattice, bond, ::DelegateTo) = setbond_inner!(delegator(Lattice(), lattice), bond)
setbond_inner!(lattice, bond, ::DontDelegate) = error("")

## `unsetsite!`
function unsetsite!(tn, site)
    checkeffect(tn, UnsetEffect(site))
    unsetsite_inner!(tn, site)
    handle!(tn, UnsetEffect(site))
    return tn
end

checkeffect(tn, @nospecialize(e::UnsetSiteEffect)) = checkeffect(tn, e, DelegatorTrait(Lattice(), tn))
checkeffect(tn, e::UnsetSiteEffect, ::DelegateTo) = checkeffect(delegator(Lattice(), tn), e)
function checkeffect(tn, e::UnsetSiteEffect, ::DontDelegate)
    hassite(tn, e.site) || throw(ArgumentError("Lattice does not contain site $(e.site)"))
end

## `unsetsite_inner!`
unsetsite_inner!(lattice, site) = unsetsite_inner!(lattice, site, DelegatorTrait(Lattice(), lattice))
unsetsite_inner!(lattice, site, ::DelegateTo) = unsetsite_inner!(delegator(Lattice(), lattice), site)
unsetsite_inner!(lattice, site, ::DontDelegate) = error("")

## `unsetbond!`
function unsetbond!(tn, bond)
    checkeffect(tn, UnsetEffect(bond))
    unsetbond_inner!(tn, bond)
    handle!(tn, UnsetEffect(bond))
    return tn
end

checkeffect(tn, @nospecialize(e::UnsetBondEffect)) = checkeffect(tn, e, DelegatorTrait(Lattice(), tn))
checkeffect(tn, e::UnsetBondEffect, ::DelegateTo) = checkeffect(delegator(Lattice(), tn), e)
function checkeffect(tn, e::UnsetBondEffect, ::DontDelegate)
    hasbond(tn, e.bond) || throw(ArgumentError("Lattice does not contain bond $(e.bond)"))
end

## `unsetbond_inner!`
unsetbond_inner!(lattice, bond) = unsetbond_inner!(lattice, bond, DelegatorTrait(Lattice(), lattice))
unsetbond_inner!(lattice, bond, ::DelegateTo) = unsetbond_inner!(delegator(Lattice(), lattice), bond)
unsetbond_inner!(lattice, bond, ::DontDelegate) = error("")
