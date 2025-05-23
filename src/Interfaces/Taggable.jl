using QuantumTags: Tag
using Networks: Vertex, Edge

# TODO it has a lot of overlap with the `Lattice` interface
# TODO try decoupling `Site` and `Link`, and allow setting any `Tag`
# TODO try defining `Base.getindex` / `Base.setindex`

# interface object
"""
    Taggable <: Interface

A singleton type that reprents the interface of "taggable" Tensor Networks; i.e. tensors and indices can be indexed with
`Tag`s. An alternative name could be `Indexable`, but "index" is too used and can lead to confusion.
"""
struct Taggable <: Interface end

# keyword-dispatching methods
function sites end
function links end

function site end
function link end

# query methods
function all_sites end
function all_links end

function all_sites_iter end
function all_links_iter end

function hassite end
function haslink end

function nsites end
function nlinks end

function tensor_at end
function ind_at end
function site_at end
function link_at end

function size_link end

function sites_like end
function site_like end
function links_like end
function link_like end

# extra: optional methods that could be other interfaces...
## get vertex/edge from site/link
function site_vertex end
function link_edge end

## get site/link from vertex/edge
function vertex_site end
function edge_link end

# mutating methods
function tag! end
function tag_inner! end

function untag! end
function untag_inner! end

function replace_tag! end
function replace_tag_inner! end

# effects
"""
    TagEffect{Tag,Obj} <: Effect

Represents the effect of setting a link or mapping between a `Tag` and an `Obj`ect.
"""
struct TagEffect{T,O} <: Effect
    tag::T
    obj::O
end

TagEffect(tag::T, @nospecialize(obj::Tensor)) where {T} = TagEffect{T,Tensor}(tag, obj)
TagEffect(tag::T, @nospecialize(obj::Index)) where {T} = TagEffect{T,Index}(tag, obj)

"""
    UntagEffect{Tag,Obj} <: Effect

Represents the effect of setting a link or mapping between a `Tag` and an `Obj`ect.
"""
struct UntagEffect{T} <: Effect
    tag::T
end

# implementation
tensor(kwargs::NamedTuple{(:at,)}, tn) = tensor_at(tn, kwargs.at)
ind(kwargs::NamedTuple{(:at,)}, tn) = ind_at(tn, kwargs.at)

## `sites`
sites(tn; kwargs...) = sites(sort_nt(values(kwargs)), tn)
sites(::@NamedTuple{}, tn) = all_sites(tn)

# TODO maybe is good idea to have a function that returns the default comparer method
# e.g. `is_like_f(::Plug)` returns `is_plug_equal`... so `like` is a trait?
# TODO important: if we do that, `is_like_f` should be able to compose with parametric types of `Plug` and such
sites(kwargs::NamedTuple{(:like)}, tn) = sites(tn; by=isequal, kwargs...)
sites(kwargs::NamedTuple{(:by, :like)}, tn) = sites_like(kwargs.by, tn, kwargs.like)

site(tn; kwargs...) = site(sort_nt(values(kwargs)), tn)
site(kwargs::NamedTuple{(:at,)}, tn) = site_at(tn, kwargs.at)

# TODO maybe is good idea to have a function that returns the default comparer method
# e.g. `is_like_f(::Plug)` returns `is_plug_equal`... so `like` is a trait?
# TODO important: if we do that, `is_like_f` should be able to compose with parametric types of `Plug` and such
site(kwargs::NamedTuple{(:like)}, tn) = site(tn; by=isequal, kwargs...)
site(kwargs::NamedTuple{(:by, :like)}, tn) = site_like(kwargs.by, tn, kwargs.like)

## `links`
links(tn; kwargs...) = links(sort_nt(values(kwargs)), tn)
links(::@NamedTuple{}, tn) = all_links(tn)

# TODO maybe is good idea to have a function that returns the default comparer method
# e.g. `is_like_f(::Plug)` returns `is_plug_equal`... so `like` is a trait?
# TODO important: if we do that, `is_like_f` should be able to compose with parametric types of `Plug` and such
links(kwargs::NamedTuple{(:like)}, tn) = links(tn; by=isequal, kwargs...)
links(kwargs::NamedTuple{(:by, :like)}, tn) = links_like(kwargs.by, tn, kwargs.like)

link(tn; kwargs...) = link(sort_nt(values(kwargs)), tn)
link(kwargs::NamedTuple{(:at,)}, tn) = link_at(tn, kwargs.at)

# TODO maybe is good idea to have a function that returns the default comparer method
# e.g. `is_like_f(::Plug)` returns `is_plug_equal`... so `like` is a trait?
# TODO important: if we do that, `is_like_f` should be able to compose with parametric types of `Plug` and such
link(kwargs::NamedTuple{(:like)}, tn) = link(tn; by=isequal, kwargs...)
link(kwargs::NamedTuple{(:by, :like)}, tn) = link_like(kwargs.by, tn, kwargs.like)

## `all_sites`
all_sites(tn) = all_sites(tn, DelegatorTrait(Taggable(), tn))
all_sites(tn, ::DelegateTo) = all_sites(delegator(Taggable(), tn))
all_sites(tn, ::DontDelegate) = throw(MethodError(all_sites, (tn,)))

## `all_links`
all_links(tn) = all_links(tn, DelegatorTrait(Taggable(), tn))
all_links(tn, ::DelegateTo) = all_links(delegator(Taggable(), tn))
all_links(tn, ::DontDelegate) = throw(MethodError(all_links, (tn,)))

## `all_sites_iter`
### helper method to avoid allocations on interation
### WARN it may mutate stuff
all_sites_iter(tn) = all_sites_iter(tn, DelegatorTrait(Taggable(), tn))
all_sites_iter(tn, ::DelegateTo) = all_sites_iter(delegator(Taggable(), tn))
function all_sites_iter(tn, ::DontDelegate)
    fallback(all_sites_iter)
    sites(tn)
end

## `all_links_iter`
### helper method to avoid allocations on interation
### WARN it may mutate stuff
all_links_iter(tn) = all_links_iter(tn, DelegatorTrait(Taggable(), tn))
all_links_iter(tn, ::DelegateTo) = all_links_iter(delegator(Taggable(), tn))
all_links_iter(tn, ::DontDelegate) = links(tn)

## `hassite`
hassite(tn, site) = hassite(tn, site, DelegatorTrait(Taggable(), tn))
hassite(tn, site, ::DelegateTo) = hassite(delegator(Taggable(), tn), site)
function hassite(tn, site, ::DontDelegate)
    fallback(hassite)
    site ∈ all_sites(tn)
end

## `haslink`
haslink(tn, link) = haslink(tn, link, DelegatorTrait(Taggable(), tn))
haslink(tn, link, ::DelegateTo) = haslink(delegator(Taggable(), tn), link)
function haslink(tn, link, ::DontDelegate)
    fallback(haslink)
    link ∈ all_links(tn)
end

## `nsites`
nsites(tn; kwargs...) = nsites(sort_nt(values(kwargs)), tn)
nsites(::@NamedTuple{}, tn) = nsites((;), tn, DelegatorTrait(Taggable(), tn))
nsites(::@NamedTuple{}, tn, ::DelegateTo) = nsites(delegator(Taggable(), tn))

function nsites(::@NamedTuple{}, tn, ::DontDelegate)
    fallback(nsites)
    length(sites(tn))
end

function nsites(kwargs::NamedTuple, tn)
    fallback(nsites)
    length(sites(kwargs, tn))
end

## `nlinks`
nlinks(tn; kwargs...) = nlinks(sort_nt(values(kwargs)), tn)
nlinks(::@NamedTuple{}, tn) = nlinks((;), tn, DelegatorTrait(Taggable(), tn))
nlinks(::@NamedTuple{}, tn, ::DelegateTo) = nlinks(delegator(Taggable(), tn))

function nlinks(::@NamedTuple{}, tn, ::DontDelegate)
    fallback(nlinks)
    length(links(tn))
end

function nlinks(kwargs::NamedTuple, tn)
    fallback(nlinks)
    length(links(kwargs, tn))
end

## `tensor_at`
tensor_at(tn, tag) = tensor_at(tn, tag, DelegatorTrait(Taggable(), tn))
tensor_at(tn, tag, ::DelegateTo) = tensor_at(delegator(Taggable(), tn), tag)
tensor_at(tn, tag, ::DontDelegate) = tensor(tn; vertex=site_vertex(tn, tag))

## `ind_at`
ind_at(tn, tag) = ind_at(tn, tag, DelegatorTrait(Taggable(), tn))
ind_at(tn, tag, ::DelegateTo) = ind_at(delegator(Taggable(), tn), tag)
ind_at(tn, tag, ::DontDelegate) = ind(tn; edge=link_edge(tn, tag))

## `site_at`
site_at(tn, x) = site_at(tn, x, DelegatorTrait(Taggable(), tn))
site_at(tn, x, ::DelegateTo) = site_at(delegator(Taggable(), tn), x)
site_at(tn, x, ::DontDelegate) = throw(MethodError(site_at, (tn, x)))
site_at(tn, tensor::Tensor, ::DontDelegate) = vertex_site(tn, tensor_vertex(tn, tensor))
site_at(tn, v::Networks.AbstractVertex, ::DontDelegate) = vertex_site(tn, v)

## `link_at`
link_at(tn, x) = link_at(tn, x, DelegatorTrait(Taggable(), tn))
link_at(tn, x, ::DelegateTo) = link_at(delegator(Taggable(), tn), x)
link_at(tn, x, ::DontDelegate) = throw(MethodError(link_at, (tn, x)))
link_at(tn, ind::Index, ::DontDelegate) = edge_link(tn, index_edge(tn, ind))
link_at(tn, e::Networks.AbstractEdge, ::DontDelegate) = edge_link(tn, e)

## `size_link`
size_link(tn, link) = size_ind(tn, ind_at(tn, link))

## `sites_like`
### TODO might be interesting to dispatch for performance?
sites_like(isequal_f, tn, ref_site) = filter(Base.Fix1(isequal_f, ref_site), all_sites_iter(tn))

## `site_like`
### TODO might be interesting to dispatch for performance?
function site_like(isequal_f, tn, ref_site)
    # we use `first` for performance, but `only` would be more correct
    first(sites_like(isequal_f, tn, ref_site))
end

## `links_like`
### TODO might be interesting to dispatch for performance?
links_like(isequal_f, tn, ref_link) = filter(Base.Fix1(isequal_f, ref_link), all_links_iter(tn))

## `link_like`
### TODO might be interesting to dispatch for performance?
function link_like(isequal_f, tn, ref_link)
    # we use `first` for performance, but `only` would be more correct
    first(links_like(isequal_f, tn, ref_link))
end

## `site_vertex`
site_vertex(tn, site) = site_vertex(tn, site, DelegatorTrait(Taggable(), tn))
site_vertex(tn, site, ::DelegateTo) = site_vertex(delegator(Taggable(), tn), site)
site_vertex(tn, site, ::DontDelegate) = throw(MethodError(site_vertex, (tn, site)))

## `link_edge`
link_edge(tn, link) = link_edge(tn, link, DelegatorTrait(Taggable(), tn))
link_edge(tn, link, ::DelegateTo) = link_edge(delegator(Taggable(), tn), link)
link_edge(tn, link, ::DontDelegate) = throw(MethodError(link_edge, (tn, link)))

## `vertex_site`
vertex_site(tn, vertex) = vertex_site(tn, vertex, DelegatorTrait(Taggable(), tn))
vertex_site(tn, vertex, ::DelegateTo) = vertex_site(delegator(Taggable(), tn), vertex)
vertex_site(tn, vertex, ::DontDelegate) = throw(MethodError(vertex_site, (tn, vertex)))

## `edge_link`
edge_link(tn, edge) = edge_link(tn, edge, DelegatorTrait(Taggable(), tn))
edge_link(tn, edge, ::DelegateTo) = edge_link(delegator(Taggable(), tn), edge)
edge_link(tn, edge, ::DontDelegate) = throw(MethodError(edge_link, (tn, edge)))

## `tag!`
function tag!(tn, x, tag)
    checkeffect(tn, TagEffect(tag, x))
    tag_inner!(tn, x, tag)
    handle!(tn, TagEffect(tag, x))
    return tn
end

tag!(tn, tensor::Tensor, tag) = tag!(tn, tensor_vertex(tn, tensor), tag)
tag!(tn, ind::Index, tag) = tag!(tn, index_edge(tn, ind), tag)

checkeffect(tn, @nospecialize(e::TagEffect)) = checkeffect(tn, e, DelegatorTrait(Taggable(), tn))
checkeffect(tn, @nospecialize(e::TagEffect), ::DelegateTo) = checkeffect(delegator(Taggable(), tn), e)
checkeffect(tn, @nospecialize(e::TagEffect), ::DontDelegate) = throw(MethodError(checkeffect, (tn, e)))

function checkeffect(tn, @nospecialize(e::TagEffect{<:Site,<:Vertex}))
    hassite(tn, e.tag) && throw(ArgumentError("Tag $(e.tag) already exists in TensorNetwork"))
    hasvertex(tn, e.obj) || throw(ArgumentError("Vertex not found in TensorNetwork"))
end

function checkeffect(tn, @nospecialize(e::TagEffect{<:Site,<:Tensor}))
    hassite(tn, e.tag) && throw(ArgumentError("Tag $(e.tag) already exists in TensorNetwork"))
    hastensor(tn, e.obj) || throw(ArgumentError("Tensor not found in TensorNetwork"))
end

function checkeffect(tn, @nospecialize(e::TagEffect{<:Link,<:Edge}))
    haslink(tn, e.tag) && throw(ArgumentError("Tag $(e.tag) already exists in TensorNetwork"))
    hasedge(tn, e.obj) || throw(ArgumentError("Edge not found in TensorNetwork"))
end

function checkeffect(tn, @nospecialize(e::TagEffect{<:Link,<:Index}))
    haslink(tn, e.tag) && throw(ArgumentError("Tag $(e.tag) already exists in TensorNetwork"))
    hasind(tn, e.obj) || throw(ArgumentError("Index not found in TensorNetwork"))
end

handle!(tn, @nospecialize(e::E)) where {E<:TagEffect} = handle!(tn, e, DelegatorTrait(Taggable(), tn))
handle!(tn, @nospecialize(e::E), ::DelegateTo) where {E<:TagEffect} = handle!(delegator(Taggable(), tn), e)
handle!(tn, @nospecialize(e::E), ::DontDelegate) where {E<:TagEffect} = nothing

## `tag_inner!`
tag_inner!(tn, x, tag) = tag_inner!(tn, x, tag, DelegatorTrait(Taggable(), tn))
tag_inner!(tn, x, tag, ::DelegateTo) = tag_inner!(delegator(Taggable(), tn), x, tag)
tag_inner!(tn, x, tag, ::DontDelegate) = throw(MethodError(tag_inner!, (tn, x, tag)))

## `untag!`
function untag!(tn, tag)
    checkeffect(tn, UntagEffect(tag))
    untag_inner!(tn, tag)
    handle!(tn, UntagEffect(tag))
    return tn
end

checkeffect(tn, @nospecialize(e::UntagEffect)) = checkeffect(tn, e, DelegatorTrait(Taggable(), tn))
checkeffect(tn, @nospecialize(e::UntagEffect), ::DelegateTo) = checkeffect(delegator(Taggable(), tn), e)
checkeffect(tn, @nospecialize(e::UntagEffect), ::DontDelegate) = throw(MethodError(checkeffect, (tn, e)))

function checkeffect(tn, @nospecialize(e::UntagEffect{<:Site}))
    hassite(tn, e.tag) || throw(ArgumentError("Site $(e.tag) not found in TensorNetwork"))
end

function checkeffect(tn, @nospecialize(e::UntagEffect{<:Link}))
    haslink(tn, e.tag) || throw(ArgumentError("Link $(e.tag) not found in TensorNetwork"))
end

handle!(tn, @nospecialize(e::E)) where {E<:UntagEffect} = handle!(tn, e, DelegatorTrait(Taggable(), tn))
handle!(tn, @nospecialize(e::E), ::DelegateTo) where {E<:UntagEffect} = handle!(delegator(Taggable(), tn), e)
handle!(tn, @nospecialize(e::E), ::DontDelegate) where {E<:UntagEffect} = nothing

## `untag_inner!`
untag_inner!(tn, tag) = untag_inner!(tn, tag, DelegatorTrait(Taggable(), tn))
untag_inner!(tn, tag, ::DelegateTo) = untag_inner!(delegator(Taggable(), tn), tag)
untag_inner!(tn, tag, ::DontDelegate) = throw(MethodError(untag_inner!, (tn, tag)))

## `replace_tag!`
function replace_tag!(tn, old_tag, new_tag)
    checkeffect(tn, ReplaceEffect(old_tag, new_tag))
    replace_tag_inner!(tn, old_tag, new_tag)
    handle!(tn, ReplaceEffect(old_tag, new_tag))
    return tn
end
replace_tag!(tn, old_new::Pair) = replace_tag!(tn, old_new.first, old_new.second)

checkeffect(tn, @nospecialize(e::ReplaceEffect)) = checkeffect(tn, e, DelegatorTrait(Taggable(), tn))
checkeffect(tn, @nospecialize(e::ReplaceEffect), ::DelegateTo) = checkeffect(delegator(Taggable(), tn), e)

function checkeffect(tn, @nospecialize(e::ReplaceEffect{<:Site,<:Site}))
    old_tag = e.old
    new_tag = e.new

    old_tag == new_tag && return tn
    hassite(tn, old_tag) || throw(ArgumentError("Site $(old_tag) not found in TensorNetwork"))
    hassite(tn, new_tag) && throw(ArgumentError("Site $(new_tag) already exists in TensorNetwork"))
end

function checkeffect(tn, @nospecialize(e::ReplaceEffect{<:Link,<:Link}))
    old_tag = e.old
    new_tag = e.new

    old_tag == new_tag && return tn
    haslink(tn, old_tag) || throw(ArgumentError("Link $(old_tag) not found in TensorNetwork"))
    haslink(tn, new_tag) && throw(ArgumentError("Link $(new_tag) already exists in TensorNetwork"))
end

handle!(tn, @nospecialize(e::ReplaceEffect{<:Tag,<:Tag})) = handle!(tn, e, DelegatorTrait(Taggable(), tn))
handle!(tn, @nospecialize(e::ReplaceEffect{<:Tag,<:Tag}), ::DelegateTo) = handle!(delegator(Taggable(), tn), e)
handle!(tn, @nospecialize(e::ReplaceEffect{<:Tag,<:Tag}), ::DontDelegate) = nothing

## `replace_tag_inner!`
replace_tag_inner!(tn, old_tag, new_tag) = replace_tag_inner!(tn, old_tag, new_tag, DelegatorTrait(Taggable(), tn))
replace_tag_inner!(tn, old_tag, new_tag, ::DelegateTo) = replace_tag_inner!(delegator(Taggable(), tn), old_tag, new_tag)

function replace_tag_inner!(tn, old_tag::Site, new_tag::Site, ::DontDelegate)
    fallback(replace_tag_inner!)

    old_tag == new_tag && return tn
    hassite(tn, old_tag) || throw(ArgumentError("old tag not found"))
    hassite(tn, new_tag) && throw(ArgumentError("new tag already exists"))

    _vertex = site_vertex(tn, old_tag)
    untag_inner!(tn, old_tag)
    tag_inner!(tn, _vertex, new_tag)
end

function replace_tag_inner!(tn, old_tag::Link, new_tag::Link, ::DontDelegate)
    fallback(replace_tag_inner!)

    old_tag == new_tag && return tn
    haslink(tn, old_tag) || throw(ArgumentError("old tag not found"))
    haslink(tn, new_tag) && throw(ArgumentError("new tag already exists"))

    _edge = link_edge(tn, old_tag)
    untag_inner!(tn, old_tag)
    tag_inner!(tn, _edge, new_tag)
end
