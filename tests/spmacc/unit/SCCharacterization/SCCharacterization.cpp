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

#include "spmacc/particles/initialization/SC.hpp"
#include "spmacc/topology/Cartesian.hpp"

#include <array>
#include <cstdint>
#include <limits>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

namespace
{
    constexpr uint32_t legacyTruncateNearInteger(float value)
    {
        auto const truncated = static_cast<uint32_t>(value);
        auto const scale = (value > 1.0f) ? value : 1.0f;
        auto const tolerance = 4.0f * std::numeric_limits<float>::epsilon() * scale;
        auto const distanceToNext = static_cast<float>(truncated + 1u) - value;
        return (distanceToNext >= 0.0f && distanceToNext <= tolerance) ? truncated + 1u : truncated;
    }

    template<pmacc::spearhed::T_Dim Dim>
    constexpr auto legacySpacingCellCounts(std::array<float, Dim> const& extents, float spacing)
    {
        std::array<uint32_t, Dim> cells{};
        cells[0] = legacyTruncateNearInteger(extents[0] / spacing);
        cells[0] = (cells[0] == 0u) ? 1u : cells[0];
        for(std::size_t i = 1; i < Dim; ++i)
        {
            auto const rounded = static_cast<uint32_t>(extents[i] / spacing + 0.5f);
            cells[i] = (rounded == 0u) ? 1u : rounded;
        }
        return cells;
    }

    template<pmacc::spearhed::T_Dim Dim>
    constexpr uint32_t legacyNumSites(std::array<uint32_t, Dim> const& cells)
    {
        uint32_t sites = 1u;
        for(auto const count : cells)
            sites *= count;
        return sites;
    }

    template<pmacc::spearhed::T_Dim Dim>
    struct FluidLayout
    {
        std::array<uint32_t, Dim> leftCells;
        std::array<uint32_t, Dim> rightCells;
        uint32_t leftParticles;
        uint32_t rightParticles;
        float particleMass;
        std::array<float, Dim> leftEffectiveSpacing;
        std::array<float, Dim> rightEffectiveSpacing;
    };

    template<pmacc::spearhed::T_Dim Dim>
    constexpr FluidLayout<Dim> characterize(
        std::array<float, Dim> const& regionExtents,
        float leftSpacing,
        float spacingRatio)
    {
        auto const leftCells = legacySpacingCellCounts<Dim>(regionExtents, leftSpacing);
        auto const rightCells = legacySpacingCellCounts<Dim>(regionExtents, leftSpacing * spacingRatio);

        float particleMass = 1.0f;
        for(std::size_t i = 0; i < Dim; ++i)
            particleMass *= leftSpacing;

        std::array<float, Dim> leftEffectiveSpacing{};
        std::array<float, Dim> rightEffectiveSpacing{};
        for(std::size_t i = 0; i < Dim; ++i)
        {
            leftEffectiveSpacing[i] = regionExtents[i] / static_cast<float>(leftCells[i]);
            rightEffectiveSpacing[i] = regionExtents[i] / static_cast<float>(rightCells[i]);
        }

        return {
            leftCells,
            rightCells,
            legacyNumSites<Dim>(leftCells),
            legacyNumSites<Dim>(rightCells),
            particleMass,
            leftEffectiveSpacing,
            rightEffectiveSpacing,
        };
    }

    void requireApprox(float actual, float expected)
    {
        REQUIRE(actual == Catch::Approx(expected).epsilon(1.0e-6));
    }
} // namespace

TEST_CASE("Legacy Sod SC fluid layouts remain characterized", "[particles][sc][sod][baseline]")
{
    SECTION("1D")
    {
        auto const layout = characterize<1>({1.0f}, 1.125f / 32000.0f, 8.0f);

        REQUIRE(layout.leftCells == std::array<uint32_t, 1>{28444u});
        REQUIRE(layout.rightCells == std::array<uint32_t, 1>{3555u});
        REQUIRE(layout.leftParticles == 28444u);
        REQUIRE(layout.rightParticles == 3555u);
        REQUIRE(layout.leftParticles + layout.rightParticles == 31999u);
        requireApprox(layout.particleMass, 3.515625e-5f);
        requireApprox(layout.leftEffectiveSpacing[0], 3.5156798e-5f);
        requireApprox(layout.rightEffectiveSpacing[0], 2.8129396e-4f);
    }

    SECTION("2D")
    {
        auto const layout = characterize<2>({1.0f, 0.07f}, 0.005f, 2.8284271f);

        REQUIRE(layout.leftCells == std::array<uint32_t, 2>{200u, 14u});
        REQUIRE(layout.rightCells == std::array<uint32_t, 2>{70u, 5u});
        REQUIRE(layout.leftParticles == 2800u);
        REQUIRE(layout.rightParticles == 350u);
        REQUIRE(layout.leftParticles + layout.rightParticles == 3150u);
        requireApprox(layout.particleMass, 2.5e-5f);
        requireApprox(layout.leftEffectiveSpacing[0], 0.005f);
        requireApprox(layout.leftEffectiveSpacing[1], 0.005f);
        requireApprox(layout.rightEffectiveSpacing[0], 0.014285714f);
        requireApprox(layout.rightEffectiveSpacing[1], 0.014f);
    }

    SECTION("3D")
    {
        auto const layout = characterize<3>({1.0f, 1.0f / 3.0f, 1.0f / 3.0f}, 1.0f / 30.0f, 2.0f);

        REQUIRE(layout.leftCells == std::array<uint32_t, 3>{30u, 10u, 10u});
        REQUIRE(layout.rightCells == std::array<uint32_t, 3>{15u, 5u, 5u});
        REQUIRE(layout.leftParticles == 3000u);
        REQUIRE(layout.rightParticles == 375u);
        REQUIRE(layout.leftParticles + layout.rightParticles == 3375u);
        requireApprox(layout.particleMass, 3.7037044e-5f);
        requireApprox(layout.leftEffectiveSpacing[0], 0.033333335f);
        requireApprox(layout.leftEffectiveSpacing[1], 0.033333335f);
        requireApprox(layout.leftEffectiveSpacing[2], 0.033333335f);
        requireApprox(layout.rightEffectiveSpacing[0], 0.06666667f);
        requireApprox(layout.rightEffectiveSpacing[1], 0.06666667f);
        requireApprox(layout.rightEffectiveSpacing[2], 0.06666667f);
    }
}
