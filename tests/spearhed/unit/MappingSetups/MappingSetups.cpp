/* Copyright 2026 Tapish Narwal
 *
 * This file is part of SPEARHED.
 *
 * SPEARHED is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License or
 * (at your option) any later version.
 */

#include "spearhed/MappingSetups/AdaptiveSpatialSplit.hpp"
#include "spearhed/MappingSetups/AdaptiveVelocitySplit.hpp"
#include "spearhed/MappingSetups/FixedCartesian.hpp"
#include "spearhed/MappingSetups/MaterialAabb.hpp"
#include "spearhed/ParticleDefinition.hpp"
#include "spearhed/particles/attributes/Acceleration.hpp"
#include "spearhed/particles/attributes/Density.hpp"
#include "spearhed/particles/attributes/DuDt.hpp"
#include "spearhed/particles/attributes/InternalEnergy.hpp"
#include "spearhed/particles/attributes/Mass.hpp"
#include "spearhed/particles/attributes/SmoothingLength.hpp"
#include "spearhed/particles/attributes/Velocity.hpp"
#include "spearhed/particles/density/DensitySummation.hpp"
#include "spearhed/particles/initialization/InitParticles.hpp"
#include "spearhed/particles/initialization/InitRegions.hpp"
#include "spearhed/sph/CubicSplineKernel.hpp"
#include "spearhed/sph/HydroForces.hpp"
#include "spearhed/test/SpearhedParticleFixture.hpp"
#include "spmacc/particles/algorithms/FrameIndex.hpp"
#include "spmacc/particles/attributes/MultiMask.hpp"
#include "spmacc/particles/spatial/InteractionPlan.hpp"

#include <pmacc/eventSystem/waitForAllTasks.hpp>

#include <catch2/catch_test_macros.hpp>

namespace
{
    struct SmallTracerGrid
    {
        auto operator()(auto const& setup) const
        {
            using Grid = pmacc::spearhed::FixedCartesianGrid<spearhed::CS>;
            return Grid{
                {setup.domain.origin + setup.domain.min, setup.domain.origin + setup.domain.max},
                {2u, 1u, 1u}};
        }
    };

    struct MixedMappingSetup : spearhed::DefaultSetup
    {
        using DecompositionGroups = std::tuple<
            pmacc::spearhed::AdaptiveSplitDecompositionGroup<
                spearhed::mapping_setups::VelocitySplitTrigger,
                spearhed::mapping_setups::VelocitySplitPartition,
                pmacc::spearhed::species::Default>,
            pmacc::spearhed::MaterialAabbDecompositionGroup<pmacc::spearhed::species::Boundary>,
            pmacc::spearhed::FixedCartesianDecompositionGroup<SmallTracerGrid, pmacc::spearhed::species::Tracer>>;
    };

    struct SeedSplitVelocities
    {
        HDINLINE void operator()(auto&, auto& particle) const
        {
            bool const slow = particle[spearhed::tags::particleId] == 0u;
            particle[spearhed::tags::vel][pmacc::spearhed::tags::x] = slow ? 0.0f : 0.2f;
            particle[spearhed::tags::vel][pmacc::spearhed::tags::y] = 0.0f;
            particle[spearhed::tags::vel][pmacc::spearhed::tags::z] = 0.0f;
            particle[spearhed::tags::mass] = 1.0f;
            particle[spearhed::tags::smoothingLength] = 0.5f;
            particle[spearhed::tags::internalEnergy] = 1.0f;
            particle[spearhed::tags::density] = 0.0f;
            particle[spearhed::tags::dudt] = 0.0f;
            particle[spearhed::tags::dvdt][pmacc::spearhed::tags::x] = 0.0f;
            particle[spearhed::tags::dvdt][pmacc::spearhed::tags::y] = 0.0f;
            particle[spearhed::tags::dvdt][pmacc::spearhed::tags::z] = 0.0f;
        }
    };
} // namespace

TEST_CASE("mapping setup examples compile", "[spatial][mapping-setups]")
{
    using namespace spearhed::mapping_setups;

    MaterialAabbSetup material;
    FixedCartesianSetup fixed;
    AdaptiveSpatialSplitSetup spatial;
    AdaptiveVelocitySplitSetup velocity;
    STATIC_REQUIRE(
        pmacc::spearhed::
            DecompositionGroupAssignmentFor<MaterialAabbSetup::DecompositionGroups, spearhed::AllSpecies>);
    STATIC_REQUIRE(
        pmacc::spearhed::
            DecompositionGroupAssignmentFor<FixedCartesianSetup::DecompositionGroups, spearhed::AllSpecies>);
    STATIC_REQUIRE(
        pmacc::spearhed::
            DecompositionGroupAssignmentFor<AdaptiveSpatialSplitSetup::DecompositionGroups, spearhed::AllSpecies>);
    STATIC_REQUIRE(
        pmacc::spearhed::
            DecompositionGroupAssignmentFor<AdaptiveVelocitySplitSetup::DecompositionGroups, spearhed::AllSpecies>);

    REQUIRE(FluidGrid{}(fixed).bucketCount() == 512u);
    REQUIRE(SpatialSplitTrigger{}(spatial).threshold == 128u);
    REQUIRE(SpatialSplitPartition{}(spatial).partitionCount == 2u);
    REQUIRE(VelocitySplitTrigger{}(velocity).threshold == spearhed::Real{0.1f});
    REQUIRE(VelocitySplitPartition{}(velocity).partitionCount == 2u);
    static_cast<void>(material);
}

TEST_CASE_METHOD(
    spearhed::test::SpearhedParticleFixture<spearhed::simDim>,
    "mixed mapping group construction is safe",
    "[spatial][mapping-setups][mixed]")
{
    MixedMappingSetup setup;
    setup.totalParticles = 0u;
    spearhed::InitRegions{}(*deviceHeap, setup);
    using Allocator = decltype(deviceHeap->getAllocatorHandle());
    using Groups = pmacc::spearhed::DecompositionGroupSet<
        spearhed::AllSpecies,
        MixedMappingSetup::DecompositionGroups,
        MixedMappingSetup,
        Allocator>;
    Groups groups{setup, deviceHeap->getAllocatorHandle()};
    REQUIRE(groups.template storeFor<pmacc::spearhed::species::Tracer>().size == 2);
    pmacc::eventSystem::waitForAllTasks();
}

TEST_CASE_METHOD(
    spearhed::test::SpearhedParticleFixture<spearhed::simDim>,
    "mixed mapping first preparation is safe",
    "[spatial][mapping-setups][mixed]")
{
    MixedMappingSetup setup;
    setup.totalParticles = 2u;
    spearhed::InitRegions{}(*deviceHeap, setup);
    spearhed::InitParticles{}(setup);
    using Allocator = decltype(deviceHeap->getAllocatorHandle());
    using Groups = pmacc::spearhed::DecompositionGroupSet<
        spearhed::AllSpecies,
        MixedMappingSetup::DecompositionGroups,
        MixedMappingSetup,
        Allocator>;
    Groups groups{setup, deviceHeap->getAllocatorHandle()};
    auto& store = groups.template storeFor<pmacc::spearhed::species::Default>();
    pmacc::spearhed::launchForEach(pmacc::spearhed::levels::particle, store, SeedSplitVelocities{});
    groups.prepareAfterMotion();
    REQUIRE(store.size == 2);
    pmacc::eventSystem::waitForAllTasks();
}

TEST_CASE_METHOD(
    spearhed::test::SpearhedParticleFixture<spearhed::simDim>,
    "mixed mapping plan construction is safe",
    "[spatial][mapping-setups][mixed]")
{
    MixedMappingSetup setup;
    setup.totalParticles = 2u;
    spearhed::InitRegions{}(*deviceHeap, setup);
    spearhed::InitParticles{}(setup);
    using Allocator = decltype(deviceHeap->getAllocatorHandle());
    using Groups = pmacc::spearhed::DecompositionGroupSet<
        spearhed::AllSpecies,
        MixedMappingSetup::DecompositionGroups,
        MixedMappingSetup,
        Allocator>;
    Groups groups{setup, deviceHeap->getAllocatorHandle()};
    auto& store = groups.template storeFor<pmacc::spearhed::species::Default>();
    pmacc::spearhed::launchForEach(pmacc::spearhed::levels::particle, store, SeedSplitVelocities{});
    groups.prepareAfterMotion();
    auto target = groups.preparedFor(store);
    auto const query = pmacc::spearhed::InteractionQuery{1.0f};
    auto adaptiveProvider = pmacc::spearhed::makeCandidateProvider(target, query, groups.preparedFor(store));
    adaptiveProvider.regionOffsets.deviceToHost();
    pmacc::eventSystem::waitForAllTasks();
    REQUIRE(adaptiveProvider.regionOffsets.getHostBuffer().getDataBox()[2] == 4u);

    auto fixedProvider = pmacc::spearhed::makeCandidateProvider(
        target,
        query,
        groups.preparedFor(groups.template storeFor<pmacc::spearhed::species::Tracer>()));
    fixedProvider.regionOffsets.deviceToHost();
    pmacc::eventSystem::waitForAllTasks();
    REQUIRE(fixedProvider.regionOffsets.getHostBuffer().getDataBox()[2] == 4u);

    pmacc::eventSystem::waitForAllTasks();
}

TEST_CASE_METHOD(
    spearhed::test::SpearhedParticleFixture<spearhed::simDim>,
    "velocity split setup constructs and prepares through the normal lifecycle",
    "[spatial][mapping-setups]")
{
    using Setup = spearhed::mapping_setups::AdaptiveVelocitySplitSetup;
    Setup setup;
    setup.totalParticles = 2u;
    spearhed::InitRegions{}(*deviceHeap, setup);
    spearhed::InitParticles{}(setup);

    using Allocator = decltype(deviceHeap->getAllocatorHandle());
    using Groups
        = pmacc::spearhed::DecompositionGroupSet<spearhed::AllSpecies, Setup::DecompositionGroups, Setup, Allocator>;
    Groups groups{setup, deviceHeap->getAllocatorHandle()};
    auto& store = groups.template storeFor<pmacc::spearhed::species::Default>();
    pmacc::spearhed::launchForEach(pmacc::spearhed::levels::particle, store, SeedSplitVelocities{});

    pmacc::spearhed::FrameIndexBuffer<spearhed::PRType> index{store};
    auto const topologyBefore = store.topologyVersion;
    groups.prepareAfterMotion();
    REQUIRE(store.size == 2);
    REQUIRE(store.topologyVersion == topologyBefore + 1u);
    REQUIRE(index.builtVersion == topologyBefore);
    index.refreshIfStale(store);
    REQUIRE(index.builtVersion == store.topologyVersion);

    auto prepared = groups.preparedFor(store);
    auto plan = pmacc::spearhed::makeInteractionPlan(prepared, pmacc::spearhed::InteractionQuery{1.0f}, prepared);
    spearhed::UpdateDensity<spearhed::CubicSplineKernel>{}(plan, store, index, 0.5f).waitForFinished();
    spearhed::UpdateHydroForces<spearhed::CubicSplineKernel>{spearhed::gamma_eos}(plan, store, index, 0.5f)
        .waitForFinished();
    REQUIRE(groups.preparedFor(store).generation() == 1u);

    groups.prepareAfterMotion();
    REQUIRE(store.topologyVersion == topologyBefore + 1u);
    REQUIRE(index.builtVersion == store.topologyVersion);
    REQUIRE(groups.preparedFor(store).generation() == 2u);
}
