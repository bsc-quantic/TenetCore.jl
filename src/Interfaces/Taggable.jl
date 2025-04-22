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
function tag_inner! end
function untag_inner! end
function replace_tag_inner! end

function tag! end
function untag! end
function replace_tag! end

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
all_sites(tn) = all_sites(tn, delegates(Taggable(), tn))
all_sites(tn, ::DelegateTo) = all_sites(delegate(Taggable(), tn))
all_sites(tn, ::DontDelegate) = throw(MethodError(all_sites, (tn,)))

## `all_links`
all_links(tn) = all_links(tn, delegates(Taggable(), tn))
all_links(tn, ::DelegateTo) = all_links(delegate(Taggable(), tn))
all_links(tn, ::DontDelegate) = throw(MethodError(all_links, (tn,)))

## `all_sites_iter`
### helper method to avoid allocations on interation
### WARN it may mutate stuff
all_sites_iter(tn) = all_sites_iter(tn, delegates(Taggable(), tn))
all_sites_iter(tn, ::DelegateTo) = all_sites_iter(delegate(Taggable(), tn))
function all_sites_iter(tn, ::DontDelegate)
    @debug "Falling back to default `all_sites_iter` method"
    sites(tn)
end

## `all_links_iter`
### helper method to avoid allocations on interation
### WARN it may mutate stuff
all_links_iter(tn) = all_links_iter(tn, delegates(Taggable(), tn))
all_links_iter(tn, ::DelegateTo) = all_links_iter(delegate(Taggable(), tn))
all_links_iter(tn, ::DontDelegate) = links(tn)

## `hassite`
hassite(tn, site) = hassite(tn, site, delegates(Taggable(), tn))
hassite(tn, site, ::DelegateTo) = hassite(delegate(Taggable(), tn), site)
function hassite(tn, site, ::DontDelegate)
    @debug "Falling back to default `hassite` method"
    site ∈ all_sites(tn)
end

## `haslink`
haslink(tn, link) = haslink(tn, link, delegates(Taggable(), tn))
haslink(tn, link, ::DelegateTo) = haslink(delegate(Taggable(), tn), link)
function haslink(tn, link, ::DontDelegate)
    @debug "Falling back to default `haslink` method"
    link ∈ all_links(tn)
end

## `nsites`
nsites(tn; kwargs...) = nsites(sort_nt(values(kwargs)), tn)
nsites(::@NamedTuple{}, tn) = nsites((;), tn, delegates(Taggable(), tn))
nsites(::@NamedTuple{}, tn, ::DelegateTo) = nsites(delegate(Taggable(), tn))

function nsites(::@NamedTuple{}, tn, ::DontDelegate)
    @debug "Falling back to default `nsites` method"
    length(sites(tn))
end

function nsites(kwargs::NamedTuple, tn)
    @debug "Falling back to default `nsites` method"
    length(sites(kwargs, tn))
end

## `nlinks`
nlinks(tn; kwargs...) = nlinks(sort_nt(values(kwargs)), tn)
nlinks(::@NamedTuple{}, tn) = nlinks((;), tn, delegates(Taggable(), tn))
nlinks(::@NamedTuple{}, tn, ::DelegateTo) = nlinks(delegate(Taggable(), tn))

function nlinks(::@NamedTuple{}, tn, ::DontDelegate)
    @debug "Falling back to default `nlinks` method"
    length(links(tn))
end

function nlinks(kwargs::NamedTuple, tn)
    @debug "Falling back to default `nlinks` method"
    length(links(kwargs, tn))
end

## `tensor_at`
tensor_at(tn, tag) = tensor_at(tn, tag, delegates(Taggable(), tn))
tensor_at(tn, tag, ::DelegateTo) = tensor_at(delegate(Taggable(), tn), tag)
tensor_at(tn, tag, ::DontDelegate) = throw(MethodError(tensor_at, (tn, tag)))

## `ind_at`
ind_at(tn, tag) = ind_at(tn, tag, delegates(Taggable(), tn))
ind_at(tn, tag, ::DelegateTo) = ind_at(delegate(Taggable(), tn), tag)
ind_at(tn, tag, ::DontDelegate) = throw(MethodError(ind_at, (tn, tag)))

## `site_at`
site_at(tn, x) = site_at(tn, x, delegates(Taggable(), tn))
site_at(tn, x, ::DelegateTo) = site_at(delegate(Taggable(), tn), x)
site_at(tn, x, ::DontDelegate) = throw(MethodError(site_at, (tn, x)))

## `link_at`
link_at(tn, x) = link_at(tn, x, delegates(Taggable(), tn))
link_at(tn, x, ::DelegateTo) = link_at(delegate(Taggable(), tn), x)
link_at(tn, x, ::DontDelegate) = throw(MethodError(link_at, (tn, x)))

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

## `tag_inner!`
tag_inner!(tn, x, tag) = tag_inner!(tn, x, tag, delegates(Taggable(), tn))
tag_inner!(tn, x, tag, ::DelegateTo) = tag!(delegate(Taggable(), tn), x, tag)
tag_inner!(tn, x, tag, ::DontDelegate) = throw(MethodError(tag_inner!, (tn, x, tag)))

## `untag_inner!`
untag_inner!(tn, tag) = untag_inner!(tn, tag, delegates(Taggable(), tn))
untag_inner!(tn, tag, ::DelegateTo) = untag!(delegate(Taggable(), tn), tag)
untag_inner!(tn, tag, ::DontDelegate) = throw(MethodError(untag_inner!, (tn, tag)))

## `replace_tag_inner!`
replace_tag_inner!(tn, old_tag, new_tag) = replace_tag_inner!(tn, old_tag, new_tag, delegates(Taggable(), tn))
replace_tag_inner!(tn, old_tag, new_tag, ::DelegateTo) = replace_tag!(tn, old_tag, new_tag)
function replace_tag_inner!(tn, old_tag::Site, new_tag::Site, ::DontDelegate)
    @debug "Falling back to the default `replace_tag_inner!` method"

    old_tag == new_tag && return tn
    hastag(tn, old_tag) || throw(ArgumentError("old tag not found"))
    hastag(tn, new_tag) && throw(ArgumentError("new tag already exists"))

    tensor = tensor_at(tn, old_tag)
    untag_inner!(tn, old_tag)
    tag_inner!(tn, tensor, new_tag)
end

function replace_tag_inner!(tn, old_tag::Link, new_tag::Link, ::DontDelegate)
    @debug "Falling back to the default `replace_tag_inner!` method"

    old_tag == new_tag && return tn
    hastag(tn, old_tag) || throw(ArgumentError("old tag not found"))
    hastag(tn, new_tag) && throw(ArgumentError("new tag already exists"))

    ind = ind_at(tn, old_tag)
    untag_inner!(tn, old_tag)
    tag_inner!(tn, ind, new_tag)
end

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
