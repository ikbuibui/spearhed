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

#include <pmacc/assert.hpp>
#include <pmacc/attribute/FunctionSpecifier.hpp>

#include <array>
#include <cassert>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace pmacc::spearhed
{
    /**
     * Complete integer shape of a simple-cubic lattice.
     *
     * The descriptor is trivially copyable and is the only geometry argument
     * accepted by device placement. A valid shape has at least one cell on
     * every axis.
     */
    template<CoordinateSystem CS>
    struct SCShape
    {
        std::array<uint32_t, CS::dimension> cells{};

        /**
         * Return the bounds unchecked product of the per-axis cell counts.
         *
         * This is device-callable for placement paths. The caller must ensure
         * every axis is nonzero and the product fits in uint32_t.
         */
        [[nodiscard]] HDINLINE constexpr uint32_t numSites() const
        {
            uint32_t sites = 1u;
            for(auto const axisCells : cells)
                sites *= axisCells;
            return sites;
        }
    };

    /**
     * Return the checked number of particles in an SC shape.
     *
     * Each lattice site contains one particle, and the frame list accepts a
     * uint32_t particle count.
     *
     * @throws std::invalid_argument if any axis has zero cells
     * @throws std::overflow_error if the product does not fit in uint32_t
     */
    template<CoordinateSystem CS>
    [[nodiscard]] constexpr uint32_t checkedParticleCount(SCShape<CS> const& shape)
    {
        uint32_t count = 1u;
        for(auto const axisCells : shape.cells)
        {
            if(axisCells == 0u)
                throw std::invalid_argument("SCShape cell counts must be nonzero");
            if(count > std::numeric_limits<uint32_t>::max() / axisCells)
                throw std::overflow_error("SCShape site count exceeds uint32_t particle capacity");
            count *= axisCells;
        }
        return count;
    }

    /**
     * Compute effective per-axis spacing for host-side diagnostics.
     *
     * @throws std::invalid_argument if any axis has zero cells
     */
    template<CoordinateSystem CS>
    [[nodiscard]] constexpr std::array<double, CS::dimension> effectiveSCSpacing(
        AABB<CS> const& aabb,
        SCShape<CS> const& shape)
    {
        static_cast<void>(checkedParticleCount(shape));

        std::array<double, CS::dimension> spacing{};
        for_each_enum_tag<CS>(
            [&](auto i, auto tag)
            {
                spacing[i.value] = (static_cast<double>(aabb.max[tag]) - static_cast<double>(aabb.min[tag]))
                                   / static_cast<double>(shape.cells[i.value]);
            });
        return spacing;
    }

    /**
     * Build an SC shape whose per-axis spacing is nearest to a requested
     * uniform target spacing.
     *
     * Every axis uses the same round-to-nearest policy and is clamped to one
     * cell. This compatibility helper does not perform approximate-count
     * planning; new count-driven initialization should construct an SCShape on
     * the host. The AABB must have positive extents and targetSpacing must be
     * positive.
     */
    template<CoordinateSystem CS>
    HDINLINE constexpr SCShape<CS> makeSCShapeForTargetSpacing(AABB<CS> const& aabb, typename CS::T_Axis targetSpacing)
    {
        using Scalar = typename CS::T_Axis;
        assert(targetSpacing > Scalar{0});

        SCShape<CS> shape{};
        for_each_enum_tag<CS>(
            [&](auto i, auto tag)
            {
                auto const extent = aabb.max[tag] - aabb.min[tag];
                assert(extent > Scalar{0});
                auto const rounded = static_cast<uint32_t>(extent / targetSpacing + Scalar{0.5});
                shape.cells[i.value] = (rounded == 0u) ? 1u : rounded;
            });
        return shape;
    }

    /**
     * Place particles at the centers of a simple-cubic lattice.
     *
     * Planning is deliberately separate from placement. The caller must pass
     * the same complete shape used to allocate particles and guarantee:
     *
     * @code
     * globalParticleIdx < checkedParticleCount(shape)
     * @endcode
     */
    template<CoordinateSystem CS>
    struct SC
    {
        DINLINE constexpr void operator()(
            [[maybe_unused]] auto const& worker,
            auto& particle,
            auto const& particleRegion,
            uint32_t globalParticleIdx,
            SCShape<CS> const& shape) const
        {
            using Scalar = typename CS::T_Axis;
            auto const& aabb = particleRegion.volume;
            auto const extents = aabb.max - aabb.min;

            PMACC_DEVICE_ASSERT(static_cast<uint64_t>(globalParticleIdx) < shape.numSites());

            uint32_t divisor = 1u;
            for_each_enum_tag<CS>(
                [&](auto i, auto tag)
                {
                    auto const cells = shape.cells[i.value];
                    uint32_t const cellIdx = (globalParticleIdx / divisor) % cells;
                    divisor *= cells;
                    Scalar const cellWidth = extents[tag] / static_cast<Scalar>(cells);
                    particle[tags::relativePos][tag]
                        = aabb.min[tag] + (static_cast<Scalar>(cellIdx) + Scalar{0.5}) * cellWidth;
                });
        }
    };

} // namespace pmacc::spearhed
