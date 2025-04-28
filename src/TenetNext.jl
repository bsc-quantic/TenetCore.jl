module TenetNext

import EinExprs: inds

# reexports
import QuantumTags: Tag, Site, Link
import QuantumTags: CartesianSite, @site_str, issite, site, sites, is_site_equal
import QuantumTags: Bond, @bond_str, bond, isbond, hassite
import QuantumTags: Plug, @plug_str, isplug, plug, is_plug_equal, isdual
export Site,
    Link, CartesianSite, Bond, Plug, @site_str, @plug_str, @bond_str, site, sites, plug, isplug, bond, isbond, hassite

import Muscle: Tensor, Index
export Tensor, Index

include("Utils.jl")

include("Interfaces/Interfaces.jl")

# from UnsafeScopeable
export @unsafe_region

# from TensorNetwork
export TensorNetwork
export tensors, tensor, hastensor, ntensors, all_tensors, addtensor!, rmtensor!, replace_tensor!
export inds, ind, hasind, ninds, all_inds, replace_ind!
export tensors_with_inds, tensors_contain_inds, tensors_intersect_inds
export inds_set, inds_parallel_to
export size_inds, size_ind
export arrays

# from Taggable
export sites, site, hassite, nsites, all_sites, sites_like, site_like
export links, link, haslink, nlinks, all_links, links_like, link_like
export tensor_at, ind_at, site_at, link_at, size_link
export tag!, untag!, replace_tag!

# from Pluggable
export plugs, plug, nplugs, hasplug, plugs_like, plug_like, ind_at_plug, plugs_like, plug_like, align!, @align!

# implementations
include("GenericTensorNetwork.jl")
export GenericTensorNetwork

include("TaggedTensorNetwork.jl")
export TaggedTensorNetwork

end
