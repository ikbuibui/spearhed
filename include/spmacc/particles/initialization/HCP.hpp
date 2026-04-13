/* Copyright 2025-2026 Tapish Narwal
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "spmacc/particles/attributes/Cartesian.hpp"
#include "spmacc/particles/attributes/RelativePosition.hpp"
#include "spmacc/particles/regions/AABB.hpp"
#include "spmacc/topology/CoordinateSystem.hpp"

#include <pmacc/attribute/FunctionSpecifier.hpp>

#include <cmath>
#include <cstdint>

namespace pmacc::spearhed
{
    /**
     * Compute the number of columns for a 2D hexagonal close-packed layout
     * that best matches the domain aspect ratio.
     *
     * The column count is derived by requiring:
     *   - nx * ny = numParticles
     *   - nx / ny ~ Lx * sqrt(3) / (2 * Ly)  (to match domain aspect ratio)
     *
     * @param numParticles total number of particles to place
     * @param aabb bounding box; first two CS axes define the HCP plane
     * @return optimal number of columns (particles per row), at least 1
     */
    template<CoordinateSystem CS>
    requires(CS::dimension == 2)
    constexpr uint32_t computeHCPNumColumns(uint32_t numParticles, AABB<CS> const& aabb)
    {
        using Scalar = typename CS::T_Axis;

        Scalar const Lx = aabb.max[tags::x] - aabb.min[tags::x];
        Scalar const Ly = aabb.max[tags::y] - aabb.min[tags::y];

        auto const nx = static_cast<uint32_t>(
            std::sqrt(static_cast<Scalar>(numParticles) * Lx * std::sqrt(Scalar{3}) / (Scalar{2} * Ly)));

        return (nx < 1u) ? 1u : nx;
    }

    /**
     * Place particles in a 2D hexagonal close-packed (HCP) lattice.
     *
     * Particles are arranged in rows along the first CS axis (tag_of<CS,0>)
     * with alternating rows offset by half the column spacing in x.
     * Any axes beyond the first two are set to the midpoint of the AABB.
     *
     * The globalParticleIdx is the flat index within the region:
     *   row = globalParticleIdx / numColumns
     *   col = globalParticleIdx % numColumns
     *
     * Typical usage in a setup:
     * @code
     *   using PlaceParticle = pmacc::spearhed::HCP<CS>;
     *
     *   auto placeParticleArgs() const {
     *       return std::make_tuple(
     *           pmacc::spearhed::computeHCP2DNumColumns(numParticles, region.volume));
     *   }
     * @endcode
     */
    template<CoordinateSystem CS>
    requires(CS::dimension == 2)
    struct HCP
    {
        DINLINE constexpr void operator()(
            [[maybe_unused]] auto const& worker,
            auto& particle,
            auto const& particleRegion,
            uint32_t globalParticleIdx,
            uint32_t numColumns) const
        {
            // TODO this only works for Cartesian systems
            // For others either figure out how to do it in those systems, or convert to and from cartesian
            using Scalar = typename CS::T_Axis;
            auto const& aabb = particleRegion.volume;
            Scalar const Lx = aabb.max[tags::x] - aabb.min[tags::x];

            Scalar const dx = Lx / static_cast<Scalar>(numColumns);

            // sqrt(3)/2 (since sqrt isnt constexpr), the row-to-row distance in units of dx
            Scalar const dy = dx * static_cast<Scalar>(0.8660254037844387);

            uint32_t const row = globalParticleIdx / numColumns;
            uint32_t const col = globalParticleIdx % numColumns;

            // Odd rows are shifted by half a column spacing
            Scalar const xOffset = (row % 2u == 1u) ? dx * Scalar{0.5f} : Scalar{0};

            *particle[tags::relativePos][tags::x]
                = aabb.min[tags::x] + (static_cast<Scalar>(col) + Scalar{0.5f}) * dx + xOffset;
            *particle[tags::relativePos][tags::y] = aabb.min[tags::y] + (static_cast<Scalar>(row) + Scalar{0.5f}) * dy;
        }
    };

} // namespace pmacc::spearhed
