module TenetNext

import EinExprs: inds

# reexports
import QuantumTags:
    Site,
    Bond,
    Plug,
    @site_str,
    @plug_str,
    @bond_str,
    issite,
    site,
    is_site_equal,
    isplug,
    plug,
    is_plug_equal,
    isdual
export Site, Bond, Plug, @site_str, @plug_str, @bond_str, site, plug

import Muscle: Tensor, Index
export Tensor, Index

include("Utils.jl")

include("Interfaces/Interfaces.jl")

# from UnsafeScopeable
export @unsafe_region

# from TensorNetwork
export TensorNetwork
export tensors, tensor, hastensor, ntensors, all_tensors
export inds, ind, hasind, ninds, all_inds

# from Taggable
export sites, site, hassite, nsites, all_sites
export links, link, haslink, nlinks, all_links
export tensor_at, ind_at, site_at, link_at
export tag!, untag!

# from Pluggable
export plugs, plug, nplugs, hasplug, plugs_like, plug_like, ind_at_plug, plugs_like, plug_like, align!, @align!

# implementations
include("GenericTensorNetwork.jl")
export GenericTensorNetwork

end
