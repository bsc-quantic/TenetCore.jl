using Test
using TenetCore
using TenetCore: LinkBijection, SiteBijection
using DelegatorTraits
using Networks
using QuantumTags
using Networks: vertex, edge

struct MockSite{S} <: Site
    tag::S
end

QuantumTags.issite(x::MockSite) = issite(x.tag)
QuantumTags.site(x::MockSite) = site(x.tag)

struct MockLink{L} <: Link
    tag::L
end

QuantumTags.isplug(x::MockLink) = isplug(x.tag)
QuantumTags.plug(x::MockLink) = plug(x.tag)

struct WrapperTaggableTensorNetwork{T} <: TenetCore.AbstractTensorNetwork
    tn::T
end

Base.copy(tn::WrapperTaggableTensorNetwork) = WrapperTaggableTensorNetwork(copy(tn.tn))
function DelegatorTraits.ImplementorTrait(interface, tn::WrapperTaggableTensorNetwork)
    DelegatorTraits.ImplementorTrait(interface, tn.tn)
end

function DelegatorTraits.DelegatorTrait(interface, tn::WrapperTaggableTensorNetwork)
    if DelegatorTraits.ImplementorTrait(interface, tn.tn) == DelegatorTraits.Implements()
        return DelegateToField{:tn}()
    else
        return DontDelegate()
    end
end

test_tensors = [
    Tensor(zeros(2, 2), [Index(:i), Index(:j)]),
    Tensor(zeros(2, 3), [Index(:j), Index(:k)]),
    Tensor(zeros(2), [Index(:j)]),
]

test_tn = SimpleTensorNetwork(test_tensors)

test_tagged_tn = GenericTensorNetwork(
    test_tn,
    SiteBijection(
        site"1" => vertex_at(test_tn, test_tensors[1]), MockSite(site"2") => vertex_at(test_tn, test_tensors[2])
    ),
    LinkBijection(
        plug"1" => edge_at(test_tn, Index(:i)),
        MockLink(plug"2") => edge_at(test_tn, Index(:k)),
        bond"1-2" => edge_at(test_tn, Index(:j)),
    ),
)

@testset "$(typeof(test_tn))" for test_tn in [test_tagged_tn, WrapperTaggableTensorNetwork(test_tagged_tn)]
    # TODO move to `Lattice` tests
    # @testset "all_sites" begin
    #     @test issetequal(all_sites(test_tn), [site"1", MockSite(site"2")])
    # end
    @testset "vertex_tags" begin
        @test issetequal(vertex_tags(test_tn), [site"1", MockSite(site"2")])
    end

    # TODO move to `Lattice` tests
    # @testset "all_links" begin
    #     @test issetequal(all_links(test_tn), [plug"1", MockLink(plug"2"), bond"1-2"])
    # end
    @testset "edge_tags" begin
        @test issetequal(edge_tags(test_tn), [plug"1", MockLink(plug"2"), bond"1-2"])
    end

    # TODO move to `Lattice` tests
    # @testset "hassite" begin
    #     @test hassite(test_tn, site"1")
    #     @test hassite(test_tn, MockSite(site"2"))
    #     @test !hassite(test_tn, site"-1")
    # end
    @testset "has_vertex_tag" begin
        @test has_vertex_tag(test_tn, site"1")
        @test has_vertex_tag(test_tn, MockSite(site"2"))
        @test !has_vertex_tag(test_tn, site"-1")
    end

    # TODO move to `Lattice` tests
    # @testset "haslink" begin
    #     @test haslink(test_tn, plug"1")
    #     @test haslink(test_tn, MockLink(plug"2"))
    #     @test haslink(test_tn, bond"1-2")
    #     @test !haslink(test_tn, bond"1-3")
    # end
    @testset "has_edge_tag" begin
        @test has_edge_tag(test_tn, plug"1")
        @test has_edge_tag(test_tn, MockLink(plug"2"))
        @test has_edge_tag(test_tn, bond"1-2")
        @test !has_edge_tag(test_tn, bond"1-3")
    end

    # TODO move to `Lattice` tests
    # @testset "nsites" begin
    #     @test nsites(test_tn) == 2
    # end

    # TODO move to `Lattice` tests
    # @testset "nlinks" begin
    #     @test nlinks(test_tn) == 3
    # end

    @testset "tensor_at(::Site)" begin
        @test tensor_at(test_tn, site"1") === test_tensors[1]
        @test tensor_at(test_tn, MockSite(site"2")) === test_tensors[2]
    end

    @testset "ind_at" begin
        @test ind_at(test_tn, plug"1") == Index(:i)
        @test ind_at(test_tn, MockLink(plug"2")) == Index(:k)
        @test ind_at(test_tn, bond"1-2") == Index(:j)
    end

    # TODO move to `Lattice` tests
    # @testset "site_at" begin
    #     @test site_at(test_tn, test_tensors[1]) == site"1"
    #     @test site_at(test_tn, test_tensors[2]) == MockSite(site"2")
    # end

    # TODO move to `Lattice` tests
    # @testset "link_at" begin
    #     @test link_at(test_tn, Index(:i)) == plug"1"
    #     @test link_at(test_tn, Index(:k)) == MockLink(plug"2")
    #     @test link_at(test_tn, Index(:j)) == bond"1-2"
    # end

    # TODO move to `Lattice` tests
    # @testset "sites_like" begin
    #     @test issetequal(sites_like(is_site_equal, test_tn, site"1"), [site"1"])
    #     @test issetequal(sites_like(is_site_equal, test_tn, site"2"), [MockSite(site"2")])
    # end

    # TODO move to `Lattice` tests
    # @testset "site_like" begin
    #     @test site_like(is_site_equal, test_tn, site"1") == site"1"
    #     @test site_like(is_site_equal, test_tn, site"2") == MockSite(site"2")
    # end

    # TODO move to `Lattice` tests
    # @testset "links_like" begin
    #     @test issetequal(links_like(is_plug_equal, test_tn, plug"1"), [plug"1"])
    #     @test issetequal(links_like(is_plug_equal, test_tn, plug"2"), [MockLink(plug"2")])
    # end

    # TODO move to `Lattice` tests
    # @testset "link_like" begin
    #     @test link_like(is_plug_equal, test_tn, plug"1") == plug"1"
    #     @test link_like(is_plug_equal, test_tn, plug"2") == MockLink(plug"2")
    # end

    @testset "tag_vertex!" begin
        @testset let test_tn = copy(test_tn)
            tag_vertex!(test_tn, vertex_at(test_tn, test_tensors[3]), site"3")
            @test tensor_at(test_tn, site"3") === test_tensors[3]
        end

        @testset let test_tn = copy(test_tn)
            tag_vertex!(test_tn, test_tensors[3], site"3")
            @test tensor_at(test_tn, site"3") === test_tensors[3]
        end
    end

    @testset "untag_vertex!" begin
        @testset let test_tn = copy(test_tn)
            untag_vertex!(test_tn, site"1")
            @test !has_vertex_tag(test_tn, site"1")
        end
    end

    @testset "tag_edge!" begin end

    @testset "untag_edge!" begin end

    @testset "replace_tag!" begin
        # replace site
        @testset let test_tn = copy(test_tn)
            replace_vertex_tag!(test_tn, site"1", site"3")
            @test has_vertex_tag(test_tn, site"3")
            @test !has_vertex_tag(test_tn, site"1")
            @test tensor_at(test_tn, site"3") === test_tensors[1]
        end

        # replace link
        @testset let test_tn = copy(test_tn)
            replace_edge_tag!(test_tn, plug"1", plug"3")
            @test has_edge_tag(test_tn, plug"3")
            @test !has_edge_tag(test_tn, plug"1")
            @test ind_at(test_tn, plug"3") == Index(:i)
        end
    end
end
