module TenetCore

using Reexport
using Compat

import EinExprs: inds

# reexports
@reexport import QuantumTags: Tag, Site, Link
@reexport import QuantumTags: CartesianSite, @site_str, issite, site, sites, is_site_equal
@reexport import QuantumTags: Bond, @bond_str, bond, isbond, hassite
@reexport import QuantumTags: Plug, @plug_str, isplug, plug, is_plug_equal, isdual

@reexport import Muscle: Tensor, Index

include("Utils.jl")

@reexport using Networks
import Networks: Interface
import Networks: DelegatorTrait, DontDelegate, DelegateTo, delegator
import Networks: ImplementorTrait, NotImplements, Implements
import Networks: Effect, checkeffect, handle!
using Networks: fallback

# NOTE for developers
# try using functions owned by us (e.g. `mysize` instead of `Base.size`)
include("Interfaces/UnsafeScope.jl")
@compat public UnsafeScopeable
export @unsafe_region

include("Interfaces/TensorNetwork.jl")
export TensorNetwork
export tensors, tensor, hastensor, ntensors, all_tensors, all_tensors_iter, addtensor!, rmtensor!, replace_tensor!
export inds, ind, hasind, ninds, all_inds, all_inds_iter, replace_ind!
export tensors_with_inds, tensors_contain_inds, tensors_intersect_inds
export inds_set, inds_parallel_to
export size_inds, size_ind

include("Interfaces/Taggable.jl")
# include("Interfaces/Lattice.jl")
@compat public Taggable
export sites, site, hassite, nsites, all_sites, sites_like, site_like
export links, link, haslink, nlinks, all_links, links_like, link_like
export tensor_at, ind_at, site_at, link_at, size_link
export tag!, untag!, replace_tag!

include("Interfaces/Pluggable.jl")
@compat public Pluggable
export plugs,
    plug,
    all_plugs,
    all_plugs_iter,
    nplugs,
    hasplug,
    plugs_like,
    plug_like,
    ind_at_plug,
    plugs_set_outputs,
    plugs_set_inputs,
    inds_set_physical,
    inds_set_virtual,
    inds_set_inputs,
    inds_set_outputs

# aliases to `Base` are in "src/Operations/AbstractTensorNetwork.jl"
include("Operations/TensorNetwork.jl")
export arrays, contract, resetinds!

include("Operations/Pluggable.jl")
export adjoint_plugs!, align!, @align!

include("Operations/AbstractTensorNetwork.jl")

# implementations
include("Components/SimpleTensorNetwork.jl")
export SimpleTensorNetwork

include("Components/GenericTensorNetwork.jl")
export GenericTensorNetwork

# extra
include("Operations/TensorExtra.jl")

end
