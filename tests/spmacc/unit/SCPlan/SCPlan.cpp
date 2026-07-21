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

#include "spmacc/particles/initialization/SCPlan.hpp"

#include "spmacc/topology/Cartesian.hpp"

#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

namespace
{
    template<typename CS>
    using Box = pmacc::spearhed::AABB<CS>;

    template<typename CS>
    using Input = pmacc::spearhed::SCRegionInput<CS>;
} // namespace

TEST_CASE("Equal-mass SC planning records weighted targets and common mass", "[particles][sc][plan]")
{
    using CS = pmacc::spearhed::Cartesian<double, 1>;
    std::array<Input<CS>, 2> const inputs{{
        {Box<CS>{{0.0}, {0.0}, {1.0}}, 3.0},
        {Box<CS>{{1.0}, {0.0}, {1.0}}, 1.0},
    }};

    auto const plan = pmacc::spearhed::makeEqualMassSCPlan(inputs, 400u);

    static_assert(std::is_trivially_copyable_v<decltype(plan)>);
    static_assert(std::is_same_v<decltype(plan.targetParticles), uint32_t>);
    static_assert(std::is_same_v<decltype(plan.actualParticles), uint32_t>);
    static_assert(std::is_same_v<decltype(plan.regions[0].numParticles), uint32_t>);
    REQUIRE(plan.targetParticles == 400u);
    REQUIRE(plan.actualParticles == 400u);
    REQUIRE(plan.regions[0].targetParticles == Catch::Approx(300.0));
    REQUIRE(plan.regions[1].targetParticles == Catch::Approx(100.0));
    REQUIRE(plan.regions[0].shape.cells == std::array<uint32_t, 1>{300u});
    REQUIRE(plan.regions[1].shape.cells == std::array<uint32_t, 1>{100u});
    REQUIRE(plan.regions[0].numParticles == 300u);
    REQUIRE(plan.regions[1].numParticles == 100u);
    REQUIRE(plan.particleMassDouble == Catch::Approx(plan.physicalMass / static_cast<double>(plan.actualParticles)));
    REQUIRE(plan.particleMass == Catch::Approx(0.01));
    REQUIRE(plan.particleMass == plan.particleMassDouble);
    REQUIRE(plan.regions[0].discreteDensity == Catch::Approx(3.0));
    REQUIRE(plan.regions[1].discreteDensity == Catch::Approx(1.0));
    REQUIRE(plan.maxRelativeDensityError == Catch::Approx(0.0).margin(1.0e-14));
}

TEST_CASE("SC planning follows region aspect ratio in each dimension", "[particles][sc][plan][shape]")
{
    SECTION("two dimensions")
    {
        using CS = pmacc::spearhed::Cartesian<double, 2>;
        std::array<Input<CS>, 1> const inputs{{
            {Box<CS>{{0.0, 0.0}, {0.0, 0.0}, {2.0, 1.0}}, 2.5},
        }};

        auto const plan = pmacc::spearhed::makeEqualMassSCPlan(inputs, 200u);
        REQUIRE(plan.regions[0].shape.cells == std::array<uint32_t, 2>{20u, 10u});
        REQUIRE(plan.regions[0].effectiveSpacing == std::array<double, 2>{0.1, 0.1});
        REQUIRE(plan.maxSpacingAnisotropy == Catch::Approx(0.0));
    }

    SECTION("three dimensions")
    {
        using CS = pmacc::spearhed::Cartesian<double, 3>;
        std::array<Input<CS>, 1> const inputs{{
            {Box<CS>{{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {3.0, 2.0, 1.0}}, 1.0},
        }};

        auto const plan = pmacc::spearhed::makeEqualMassSCPlan(inputs, 48u);
        REQUIRE(plan.regions[0].shape.cells == std::array<uint32_t, 3>{6u, 4u, 2u});
        REQUIRE(plan.regions[0].effectiveSpacing == std::array<double, 3>{0.5, 0.5, 0.5});
        REQUIRE(plan.actualParticles == 48u);
    }
}

TEST_CASE("SC planning uses double arithmetic with float coordinates", "[particles][sc][plan][precision]")
{
    using CS = pmacc::spearhed::Cartesian<float, 2>;
    std::array<Input<CS>, 2> const inputs{{
        {Box<CS>{{-1.0f, 0.0f}, {0.0f, 0.0f}, {1.0f, 0.07f}}, 1.0},
        {Box<CS>{{0.0f, 0.0f}, {0.0f, 0.0f}, {1.0f, 0.07f}}, 0.125},
    }};

    auto const plan = pmacc::spearhed::makeEqualMassSCPlan(inputs, 3150u);

    REQUIRE(plan.regions[0].shape.cells == std::array<uint32_t, 2>{200u, 14u});
    REQUIRE(plan.regions[1].shape.cells == std::array<uint32_t, 2>{70u, 5u});
    REQUIRE(plan.regions[0].numParticles == 2800u);
    REQUIRE(plan.regions[1].numParticles == 350u);
    REQUIRE(plan.actualParticles == 3150u);
    REQUIRE(plan.particleMassDouble == Catch::Approx(plan.physicalMass / static_cast<double>(plan.actualParticles)));
    REQUIRE(plan.particleMass == Catch::Approx(2.5e-5f));
    REQUIRE(plan.particleMass == static_cast<float>(plan.particleMassDouble));
    REQUIRE(plan.maxRelativeDensityError == Catch::Approx(0.0).margin(1.0e-14));
}

TEST_CASE("SC planning is equivariant under axis permutation", "[particles][sc][plan][shape]")
{
    using CS = pmacc::spearhed::Cartesian<double, 2>;
    std::array<Input<CS>, 1> const xy{{
        {Box<CS>{{0.0, 0.0}, {0.0, 0.0}, {4.0, 1.0}}, 1.0},
    }};
    std::array<Input<CS>, 1> const yx{{
        {Box<CS>{{0.0, 0.0}, {0.0, 0.0}, {1.0, 4.0}}, 1.0},
    }};

    auto const xyPlan = pmacc::spearhed::makeEqualMassSCPlan(xy, 64u);
    auto const yxPlan = pmacc::spearhed::makeEqualMassSCPlan(yx, 64u);

    REQUIRE(xyPlan.regions[0].shape.cells == std::array<uint32_t, 2>{16u, 4u});
    REQUIRE(yxPlan.regions[0].shape.cells == std::array<uint32_t, 2>{4u, 16u});
}

TEST_CASE("SC planning treats the requested count as approximate", "[particles][sc][plan][count]")
{
    using CS = pmacc::spearhed::Cartesian<double, 3>;
    std::array<Input<CS>, 1> const inputs{{
        {Box<CS>{{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}, 1.0},
    }};

    auto const plan = pmacc::spearhed::makeEqualMassSCPlan(inputs, 10u);

    REQUIRE(plan.regions[0].shape.cells == std::array<uint32_t, 3>{2u, 2u, 2u});
    REQUIRE(plan.actualParticles == 8u);
    REQUIRE(plan.relativeParticleCountError == Catch::Approx(0.2));
    REQUIRE(plan.particleMass == Catch::Approx(0.125));
    REQUIRE(plan.regions[0].discreteDensity == Catch::Approx(1.0));
}

TEST_CASE("SC planning expands candidates to satisfy equal-mass density", "[particles][sc][plan][density]")
{
    using CS = pmacc::spearhed::Cartesian<double, 1>;
    std::array<Input<CS>, 2> const inputs{{
        {Box<CS>{{0.0}, {0.0}, {1.0}}, 8.0},
        {Box<CS>{{1.0}, {0.0}, {1.0}}, 1.0},
    }};

    auto const plan = pmacc::spearhed::makeEqualMassSCPlan(inputs, 4u);

    REQUIRE(plan.regions[0].shape.cells == std::array<uint32_t, 1>{8u});
    REQUIRE(plan.regions[1].shape.cells == std::array<uint32_t, 1>{1u});
    REQUIRE(plan.actualParticles == 9u);
    REQUIRE(plan.particleMass == Catch::Approx(1.0));
    REQUIRE(plan.maxRelativeDensityError == Catch::Approx(0.0));
}

TEST_CASE("SC planning responds to density rather than a fixed spacing ratio", "[particles][sc][plan][density]")
{
    using CS = pmacc::spearhed::Cartesian<double, 1>;
    auto const left = Box<CS>{{0.0}, {0.0}, {1.0}};
    auto const right = Box<CS>{{1.0}, {0.0}, {1.0}};

    std::array<Input<CS>, 2> const uniform{{{left, 1.0}, {right, 1.0}}};
    std::array<Input<CS>, 2> const nonUniform{{{left, 3.0}, {right, 1.0}}};

    auto const uniformPlan = pmacc::spearhed::makeEqualMassSCPlan(uniform, 400u);
    auto const nonUniformPlan = pmacc::spearhed::makeEqualMassSCPlan(nonUniform, 400u);

    REQUIRE(uniformPlan.regions[0].numParticles == 200u);
    REQUIRE(uniformPlan.regions[1].numParticles == 200u);
    REQUIRE(nonUniformPlan.regions[0].numParticles == 300u);
    REQUIRE(nonUniformPlan.regions[1].numParticles == 100u);
    REQUIRE(
        nonUniformPlan.regions[0].discreteDensity / nonUniformPlan.regions[1].discreteDensity == Catch::Approx(3.0));
}

TEST_CASE("SC planning rejects invalid inputs and overflow", "[particles][sc][plan][error]")
{
    using CS = pmacc::spearhed::Cartesian<double, 1>;
    auto const validBox = Box<CS>{{0.0}, {0.0}, {1.0}};

    REQUIRE_THROWS_AS(
        pmacc::spearhed::makeEqualMassSCPlan(std::array<Input<CS>, 1>{{{validBox, 1.0}}}, 0u),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        pmacc::spearhed::makeEqualMassSCPlan(std::array<Input<CS>, 1>{{{validBox, 0.0}}}, 10u),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        pmacc::spearhed::makeEqualMassSCPlan(
            std::array<Input<CS>, 1>{{{validBox, std::numeric_limits<double>::infinity()}}},
            10u),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        pmacc::spearhed::makeEqualMassSCPlan(std::array<Input<CS>, 1>{{{Box<CS>{{0.0}, {1.0}, {1.0}}, 1.0}}}, 10u),
        std::invalid_argument);
    auto const maximumPlan = pmacc::spearhed::makeEqualMassSCPlan(
        std::array<Input<CS>, 1>{{{validBox, 1.0}}},
        std::numeric_limits<uint32_t>::max());
    REQUIRE(maximumPlan.targetParticles == std::numeric_limits<uint32_t>::max());
    REQUIRE(maximumPlan.actualParticles == std::numeric_limits<uint32_t>::max());

    using FloatCS = pmacc::spearhed::Cartesian<float, 1>;
    auto const hugeBox = Box<FloatCS>{{0.0f}, {0.0f}, {std::numeric_limits<float>::max()}};
    REQUIRE_THROWS_AS(
        pmacc::spearhed::makeEqualMassSCPlan(std::array<Input<FloatCS>, 1>{{{hugeBox, 2.0}}}, 1u),
        std::overflow_error);
}
