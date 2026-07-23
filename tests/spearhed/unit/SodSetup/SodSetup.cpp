/* Copyright 2026 Tapish Narwal
 *
 * This file is part of SPEARHED.
 *
 * SPEARHED is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SPEARHED is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SPEARHED.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#include "share/spearhed/ValidationSetups/SodShockTube.hpp"
#include "spearhed/particles/initialization/SetupInterface.hpp"

#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <catch2/catch_test_macros.hpp>

namespace
{
    struct Worker
    {
    };

    struct ParticleRegion
    {
        pmacc::spearhed::AABB<spearhed::CS> volume;
    };
} // namespace

TEST_CASE("Sod fluid allocation and placement consume the same checked plan", "[particles][sc][sod][plan]")
{
    auto const setup = spearhed::SodShockTube{};
    static_assert(spearhed::SetupInterface<spearhed::SodShockTube>);

    REQUIRE(
        setup.fluidPlan.actualParticles
        == setup.fluidPlan.regions[0].numParticles + setup.fluidPlan.regions[1].numParticles);

    std::vector<pmacc::spearhed::AABB<spearhed::CS>> volumes;
    setup.interior.template addRegions<pmacc::spearhed::species::Default>(volumes);
    REQUIRE(volumes.size() == setup.fluidPlan.regions.size());

    auto const countArgs = setup.interior.numParticlesToCreateArgs();
    auto const placeArgs = setup.interior.placeParticleArgs();
    auto const& countLattices = std::get<0>(countArgs);
    auto const& placementLattices = std::get<0>(placeArgs);

    for(std::size_t region = 0u; region < setup.fluidPlan.regions.size(); ++region)
    {
        auto const& regionPlan = setup.fluidPlan.regions[region];
        REQUIRE(countLattices[region].shape.cells == regionPlan.shape.cells);
        REQUIRE(placementLattices[region].shape.cells == regionPlan.shape.cells);
        REQUIRE(countLattices[region].numParticles == regionPlan.numParticles);
        REQUIRE(placementLattices[region].numParticles == regionPlan.numParticles);
        REQUIRE(pmacc::spearhed::checkedParticleCount(countLattices[region].shape) == regionPlan.numParticles);

        auto const selectedCount = std::apply(
            [&](auto const&... args)
            { return spearhed::sod::PlannedNumParticles{}(Worker{}, ParticleRegion{volumes[region]}, args...); },
            countArgs);
        REQUIRE(selectedCount == regionPlan.numParticles);
    }

    REQUIRE(std::get<2>(placeArgs) == setup.fluidPlan.particleMass);

    auto inconsistentPlan = setup.fluidPlan;
    ++inconsistentPlan.regions[0].numParticles;
    REQUIRE_THROWS_AS(spearhed::sod::makeFluidLattices(inconsistentPlan), std::logic_error);
}

TEST_CASE("Sod startup diagnostics report the complete fluid plan", "[particles][sc][sod][diagnostics]")
{
    auto const setup = spearhed::SodShockTube{};
    std::ostringstream output;
    setup.printStartupDiagnostics(output);
    auto const text = output.str();

    REQUIRE(text.find("target=") != std::string::npos);
    REQUIRE(text.find("actual=") != std::string::npos);
    REQUIRE(text.find("particle mass=") != std::string::npos);
    REQUIRE(text.find("left: shape=[") != std::string::npos);
    REQUIRE(text.find("right: shape=[") != std::string::npos);
    REQUIRE(text.find("effective spacing=[") != std::string::npos);
    REQUIRE(text.find("relative density error=") != std::string::npos);
}
