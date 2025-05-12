using BijectiveDicts: BijectiveDict

const LinkBiDict = BijectiveDict{Link,Index,Dict{Link,Index},Dict{Index,Link}}
const SiteBiDict = BijectiveDict{Site,Tensor,Dict{Site,Tensor},IdDict{Tensor,Site}}

@kwdef struct TagMixin
    linkmap::LinkBiDict = LinkBiDict()
    sitemap::SiteBiDict = SiteBiDict()
end

Base.copy(mixin::TagMixin) = TagMixin(copy(mixin.linkmap), copy(mixin.sitemap))

## removing tensor breaks mapping
## TODO for index removal?
function handle!(mixin::TagMixin, e::RemoveTensorEffect)
    if haskey(mixin.sitemap', e.f)
        site_tag = site_at(mixin, e.f)
        untag_inner!(mixin, site_tag)
    end
end

## replacing tensor/index breaks mapping
function handle!(mixin::TagMixin, e::ReplaceEffect{<:Tensor,<:Tensor})
    if haskey(mixin.sitemap', e.old)
        site_tag = site_at(mixin, e.old)
        untag_inner!(mixin, site_tag)
        tag_inner!(mixin, e.new, site_tag)
    end
end

function handle!(mixin::TagMixin, e::ReplaceEffect{<:Index,<:Index})
    if haskey(mixin.linkmap', e.old)
        link_tag = link_at(mixin, e.old)
        untag_inner!(mixin, link_tag)
        tag_inner!(mixin, e.new, link_tag)
    end
end

# Taggable implementation
implements(::Taggable, ::TagMixin) = Implements()

all_sites(mixin::TagMixin) = collect(all_sites_iter(mixin))
all_links(mixin::TagMixin) = collect(all_links_iter(mixin))

all_sites_iter(mixin::TagMixin) = keys(mixin.sitemap)
all_links_iter(mixin::TagMixin) = keys(mixin.linkmap)

hassite(mixin::TagMixin, site) = haskey(mixin.sitemap, site)
haslink(mixin::TagMixin, link) = haskey(mixin.linkmap, link)

nsites(::@NamedTuple{}, tn::TagMixin) = length(tn.sitemap)
nlinks(::@NamedTuple{}, tn::TagMixin) = length(tn.linkmap)

tensor_at(mixin::TagMixin, tag) = mixin.sitemap[tag]
ind_at(mixin::TagMixin, tag) = mixin.linkmap[tag]

site_at(mixin::TagMixin, tensor::Tensor) = mixin.sitemap'[tensor]
link_at(mixin::TagMixin, index::Index) = mixin.linkmap'[index]

tag_inner!(mixin::TagMixin, tensor::Tensor, site::Site) = mixin.sitemap[site] = tensor
handle!(::TagMixin, @nospecialize(e::TagEffect{S,T})) where {S<:Site,T<:Tensor} = nothing

tag_inner!(mixin::TagMixin, ind::Index, link::Link) = mixin.linkmap[link] = ind
handle!(::TagMixin, @nospecialize(e::TagEffect{L,I})) where {L<:Link,I<:Index} = nothing

untag_inner!(mixin::TagMixin, site::Site) = delete!(mixin.sitemap, site)
handle!(::TagMixin, @nospecialize(e::UntagEffect{S})) where {S<:Site} = nothing

untag_inner!(mixin::TagMixin, link::Link) = delete!(mixin.linkmap, link)
handle!(::TagMixin, @nospecialize(e::UntagEffect{L})) where {L<:Link} = nothing

handle!(::TagMixin, ::ReplaceEffect{<:Site,<:Site}) = nothing
handle!(::TagMixin, ::ReplaceEffect{<:Link,<:Link}) = nothing
