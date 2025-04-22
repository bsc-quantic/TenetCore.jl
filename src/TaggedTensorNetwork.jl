using BijectiveDicts: BijectiveDict

const LinkBiDict = BijectiveDict{Link,Index,Dict{Link,Index},Dict{Index,Link}}
const SiteBiDict = BijectiveDict{Site,Tensor,Dict{Site,Tensor},IdDict{Tensor,Site}}

@kwdef struct TaggedTensorNetwork <: AbstractTensorNetwork
    tn::GenericTensorNetwork

    linkmap::LinkBiDict = LinkBiDict()
    sitemap::SiteBiDict = SiteBiDict()
end

TaggedTensorNetwork(tn::GenericTensorNetwork; kwargs...) = TaggedTensorNetwork(; tn, kwargs...)

function Base.copy(tn::TaggedTensorNetwork)
    TaggedTensorNetwork(copy(tn.tn); linkmap=copy(tn.linkmap), sitemap=copy(tn.sitemap))
end

# UnsafeScopeable implementation
delegates(::UnsafeScopeable, ::TaggedTensorNetwork) = DelegateTo{:tn}()

# TensorNetwork implementation
delegates(::TensorNetwork, ::TaggedTensorNetwork) = DelegateTo{:tn}()

## removing tensor breaks mapping
## TODO for index removal?
function handle!(tn::TaggedTensorNetwork, e::DeleteEffect{<:Tensor})
    untag_inner!(tn, site_at(e.f))
    handle!(tn.tn, e)
end

## replacing tensor/index breaks mapping
function handle!(tn::TaggedTensorNetwork, e::ReplaceEffect{<:Tensor,<:Tensor})
    # TODO
end

# Taggable implementation
all_sites(tn::TaggedTensorNetwork) = collect(keys(tn.sitemap))
all_links(tn::TaggedTensorNetwork) = collect(keys(tn.linkmap))

all_sites_iter(tn::TaggedTensorNetwork) = keys(tn.sitemap)
all_links_iter(tn::TaggedTensorNetwork) = keys(tn.linkmap)

hastag(tn::TaggedTensorNetwork, tag) = haskey(tn.linkmap, tag) || haskey(tn.sitemap, tag)
ntags(tn::TaggedTensorNetwork) = length(tn.linkmap) + length(tn.sitemap)

tensor_at(tn::TaggedTensorNetwork, tag) = tn.sitemap[tag]
ind_at(tn::TaggedTensorNetwork, tag) = tn.linkmap[tag]

site_at(tn::TaggedTensorNetwork, tensor::Tensor) = tn.sitemap'[tensor]
link_at(tn::TaggedTensorNetwork, index::Index) = tn.linkmap'[index]

function tag_inner!(tn::TaggedTensorNetwork, tensor::Tensor, site::Site)
    tn.sitemap[site] = tensor
end

handle!(::TaggedTensorNetwork, @nospecialize(e::TagEffect{S,T})) where {S<:Site,T<:Tensor} = nothing

function tag_inner!(tn::TaggedTensorNetwork, ind::Index, link::Link)
    tn.linkmap[link] = ind
end

handle!(::TaggedTensorNetwork, @nospecialize(e::TagEffect{L,I})) where {L<:Link,I<:Index} = nothing

function untag_inner!(tn::TaggedTensorNetwork, site::Site)
    delete!(tn.sitemap, site)
end

handle!(::TaggedTensorNetwork, @nospecialize(e::UntagEffect{S})) where {S<:Site} = nothing

function untag_inner!(tn::TaggedTensorNetwork, link::Link)
    delete!(tn.linkmap, link)
end

handle!(::TaggedTensorNetwork, @nospecialize(e::UntagEffect{L})) where {L<:Link} = nothing

handle!(tn::TaggedTensorNetwork, e::ReplaceEffect{<:Site,<:Site}) = nothing

handle!(tn::TaggedTensorNetwork, e::ReplaceEffect{<:Link,<:Link}) = nothing
