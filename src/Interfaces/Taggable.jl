# interface object
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

# mutating methods
function tag! end
function untag! end
function replace_tag! end

function tag_inner! end
function untag_inner! end
function replace_tag_inner! end

# implementation
tensor(kwargs::NamedTuple{(:at,)}, tn) = tensor_at(tn, kwargs.at)
ind(kwargs::NamedTuple{(:at,)}, tn) = ind_at(tn, kwargs.at)

## `sites`
sites(tn; kwargs...) = sites(sort_nt(values(kwargs)), tn)
sites(::@NamedTuple{}, tn) = all_sites(tn)

site(tn; kwargs...) = site(sort_nt(values(kwargs)), tn)
site(kwargs::NamedTuple{(:at,)}, tn) = site_at(tn, kwargs.at)

## `links`
links(tn; kwargs...) = links(sort_nt(values(kwargs)), tn)
links(::@NamedTuple{}, tn) = all_links(tn)

link(tn; kwargs...) = link(sort_nt(values(kwargs)), tn)
link(kwargs::NamedTuple{(:at,)}, tn) = link_at(tn, kwargs.at)

## `all_sites`
all_sites(tn) = all_sites(tn, delegates(TensorNetwork(), tn))
all_sites(tn, ::DelegateTo) = all_sites(delegate(TensorNetwork(), tn))
all_sites(tn, ::DontDelegate) = throw(MethodError(all_sites, (tn,)))

## `all_links`
all_links(tn) = all_links(tn, delegates(TensorNetwork(), tn))
all_links(tn, ::DelegateTo) = all_links(delegate(TensorNetwork(), tn))
all_links(tn, ::DontDelegate) = throw(MethodError(all_links, (tn,)))

## `all_sites_iter`
### helper method to avoid allocations on interation
### WARN it may mutate stuff
all_sites_iter(tn) = all_sites_iter(tn, delegates(TensorNetwork(), tn))
all_sites_iter(tn, ::DelegateTo) = all_sites_iter(delegate(TensorNetwork(), tn))
function all_sites_iter(tn, ::DontDelegate)
    @debug "Falling back to default `all_sites_iter` method"
    sites(tn)
end

## `all_links_iter`
### helper method to avoid allocations on interation
### WARN it may mutate stuff
links_iter(tn) = links_iter(tn, delegates(TensorNetwork(), tn))
links_iter(tn, ::DelegateTo) = links_iter(delegate(TensorNetwork(), tn))
links_iter(tn, ::DontDelegate) = links(tn)

## `hassite`
hassite(tn, site) = hassite(tn, site, delegates(TensorNetwork(), tn))
hassite(tn, site, ::DelegateTo) = hassite(delegate(TensorNetwork(), tn), site)
function hassite(tn, site, ::DontDelegate)
    @debug "Falling back to default `hassite` method"
    site ∈ all_sites(tn)
end

## `haslink`
haslink(tn, link) = haslink(tn, link, delegates(TensorNetwork(), tn))
haslink(tn, link, ::DelegateTo) = haslink(delegate(TensorNetwork(), tn), link)
function haslink(tn, link, ::DontDelegate)
    @debug "Falling back to default `haslink` method"
    link ∈ all_links(tn)
end

## `nsites`
nsites(tn; kwargs...) = nsites(sort_nt(values(kwargs)), tn)
nsites(::@NamedTuple{}, tn) = nsites((;), tn, delegates(TensorNetwork(), tn))
nsites(::@NamedTuple{}, tn, ::DelegateTo) = nsites(delegate(TensorNetwork(), tn))

function nsites(::@NamedTuple{}, tn, ::DontDelegate)
    @debug "Falling back to default `nsites` method"
    length(sites(kwargs, tn))
end

function nsites(kwargs::NamedTuple, tn)
    @debug "Falling back to default `nsites` method"
    length(sites(kwargs, tn))
end

## `nlinks`
nlinks(tn; kwargs...) = nlinks(sort_nt(values(kwargs)), tn)
nlinks(::@NamedTuple{}, tn) = nlinks((;), tn, delegates(TensorNetwork(), tn))
nlinks(::@NamedTuple{}, tn, ::DelegateTo) = nlinks(delegate(TensorNetwork(), tn))

function nlinks(::@NamedTuple{}, tn, ::DontDelegate)
    @debug "Falling back to default `nlinks` method"
    length(links(tn))
end

function nlinks(kwargs::NamedTuple, tn)
    @debug "Falling back to default `nlinks` method"
    length(links(kwargs, tn))
end

## `tensor_at`
tensor_at(tn, tag) = tensor_at(tn, tag, delegates(TensorNetwork(), tn))
tensor_at(tn, tag, ::DelegateTo) = tensor_at(delegate(TensorNetwork(), tn), tag)
tensor_at(tn, tag, ::DontDelegate) = throw(MethodError(tensor_at, (tn, tag)))

## `ind_at`
ind_at(tn, tag) = ind_at(tn, tag, delegates(TensorNetwork(), tn))
ind_at(tn, tag, ::DelegateTo) = ind_at(delegate(TensorNetwork(), tn), tag)
ind_at(tn, tag, ::DontDelegate) = throw(MethodError(ind_at, (tn, tag)))

## `site_at`
site_at(tn, x) = site_at(tn, x, delegates(TensorNetwork(), tn))
site_at(tn, x, ::DelegateTo) = site_at(delegate(TensorNetwork(), tn), x)
site_at(tn, x, ::DontDelegate) = throw(MethodError(site_at, (tn, x)))

## `link_at`
link_at(tn, x) = link_at(tn, x, delegates(TensorNetwork(), tn))
link_at(tn, x, ::DelegateTo) = link_at(delegate(TensorNetwork(), tn), x)
link_at(tn, x, ::DontDelegate) = throw(MethodError(link_at, (tn, x)))

## `size_link`
size_link(tn, link) = size_ind(tn, ind_at(tn, link))

## `sites_like`
### TODO might be interesting to dispatch for performance?
sites_like(isequal_f, tn, ref_site) = filter(Base.Fix1(isequal_f, ref_site), sites_iter(tn))

## `site_like`
### TODO might be interesting to dispatch for performance?
site_like(isequal_f, tn, ref_site) = only(sites_like(isequal_f, tn, ref_site))

## `links_like`
### TODO might be interesting to dispatch for performance?
links_like(isequal_f, tn, ref_link) = filter(Base.Fix1(isequal_f, ref_link), links_iter(tn))

## `link_like`
### TODO might be interesting to dispatch for performance?
link_like(isequal_f, tn, ref_link) = only(links_like(isequal_f, tn, ref_link))

## `tag!`
function tag!(tn, x, tag)
    checkhandle(tn, TagEffect(tag, x))
    hastag(tn, tag) && throw(ArgumentError("Tag $(tag) already exists in TensorNetwork"))
    x ∈ tn || throw(ArgumentError("Object not found in TensorNetwork"))
    tag_inner!(tn, x, tag)
    handle!(tn, TagEffect(tag, x))
    return tn
end

## `untag!`
function untag!(tn, tag)
    checkhandle(tn, UntagEffect(tag))
    hastag(tn, tag) || throw(ArgumentError("Tag $(tag) not found in TensorNetwork"))
    untag_inner!(tn, tag)
    handle!(tn, UntagEffect(tag))
    return tn
end

## `replace_tag!`
function replace_tag!(tn, old_tag, new_tag)
    checkhandle(tn, ReplaceTagEffect(old_tag, new_tag))
    hastag(tn, old_tag) || throw(ArgumentError("Tag $(old_tag) not found in TensorNetwork"))
    hastag(tn, new_tag) && throw(ArgumentError("Tag $(new_tag) already exists in TensorNetwork"))
    replace_tag_inner!(tn, old_tag, new_tag)
    handle!(tn, ReplaceTagEffect(old_tag, new_tag))
    return tn
end

## `tag_inner!`
tag_inner!(tn, tag; kwargs...) = tag_inner!(tn, tag, delegates(TensorNetwork(), tn), kwargs...)
tag_inner!(tn, tag, ::DelegateTo) = tag_inner!(delegate(TensorNetwork(), tn), tag)
tag_inner!(tn, tag, ::DontDelegate) = throw(MethodError(tag_inner!, (tn, tag)))

## `untag_inner!`
untag_inner!(tn, tag; kwargs...) = untag_inner!(tn, tag, delegates(TensorNetwork(), tn), kwargs...)
untag_inner!(tn, tag, ::DelegateTo) = untag_inner!(delegate(TensorNetwork(), tn), tag)
untag_inner!(tn, tag, ::DontDelegate) = throw(MethodError(untag_inner!, (tn, tag)))

## `replace_tag_inner!`
function replace_tag_inner!(tn, old_tag::Site, new_tag::Site)
    tensor = tensor_at(tn, old_tag)
    untag_inner!(tn, old_tag)
    tag_inner!(tn, tensor, new_tag)
end

function replace_tag_inner!(tn, old_tag::Link, new_tag::Link)
    ind = ind_at(tn, old_tag)
    untag_inner!(tn, old_tag)
    tag_inner!(tn, ind, new_tag)
end
