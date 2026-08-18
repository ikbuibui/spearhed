/* Copyright 2026 Tapish Narwal
 *
 * This file is part of SPEARHED.
 *
 * SPEARHED is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License or
 * the GNU Lesser General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include "spmacc/particles/regions/mapping/DecompositionGroup.hpp"

#include "spearhed/ParticleDefinition.hpp"

#include <tuple>

#include <catch2/catch_test_macros.hpp>

namespace
{
    namespace species = pmacc::spearhed::species;

    using SeparateMaterialDecompositionGroups = std::tuple<
        pmacc::spearhed::StaticMappingDecompositionGroup<species::Default>,
        pmacc::spearhed::StaticMappingDecompositionGroup<species::Boundary>,
        pmacc::spearhed::StaticMappingDecompositionGroup<species::Tracer>>;

    using SeparateStaticBoundaryDecompositionGroups = std::tuple<
        pmacc::spearhed::StaticMappingDecompositionGroup<species::Default>,
        pmacc::spearhed::ExplicitInvalidationDecompositionGroup<species::Boundary>,
        pmacc::spearhed::StaticMappingDecompositionGroup<species::Tracer>>;

    using DuplicateDefaultDecompositionGroups = std::tuple<
        pmacc::spearhed::StaticMappingDecompositionGroup<species::Default, species::Boundary>,
        pmacc::spearhed::StaticMappingDecompositionGroup<species::Default, species::Tracer>>;
} // namespace

TEST_CASE("Decomposition-group declarations assign each registered species exactly once", "[spatial][decomposition]")
{
    STATIC_REQUIRE(
        pmacc::spearhed::DecompositionGroupAssignmentFor<SeparateMaterialDecompositionGroups, spearhed::AllSpecies>);
    STATIC_REQUIRE(
        pmacc::spearhed::
            DecompositionGroupAssignmentFor<SeparateStaticBoundaryDecompositionGroups, spearhed::AllSpecies>);
    STATIC_REQUIRE_FALSE(
        pmacc::spearhed::DecompositionGroupAssignmentFor<DuplicateDefaultDecompositionGroups, spearhed::AllSpecies>);
}
