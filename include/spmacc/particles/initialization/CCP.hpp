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
     * Compute the number of unit cells along the x-axis for a 3D face-centered cubic (FCC)
     * layout that best matches the domain aspect ratio.
     *
     * FCC has 4 atoms per conventional unit cell. The cell counts per axis are derived by
     * requiring:
     *   - nx * ny * nz = numParticles / 4
     *   - nx / ny ~ Lx / Ly  and  nx / nz ~ Lx / Lz  (to match domain aspect ratio)
     *
     * @param numParticles total number of particles to place
     * @param aabb bounding box; all three CS axes define the FCC volume
     * @return optimal number of unit cells along the x-axis, at least 1
     */
    template<CoordinateSystem CS>
    requires(CS::dimension == 3)
    constexpr uint32_t computeCCPNumCells(uint32_t numParticles, AABB<CS> const& aabb)
    {
        using Scalar = typename CS::T_Axis;

        Scalar const Lx = aabb.max[tags::x] - aabb.min[tags::x];
        Scalar const Ly = aabb.max[tags::y] - aabb.min[tags::y];
        Scalar const Lz = aabb.max[tags::z] - aabb.min[tags::z];

        // nx^3 = (N/4) * Lx^2 / (Ly * Lz)
        auto const nx
            = static_cast<uint32_t>(std::cbrt(static_cast<Scalar>(numParticles) / Scalar{4} * Lx * Lx / (Ly * Lz)));

        return (nx < 1u) ? 1u : nx;
    }

    /**
     * Place particles in a 3D cubic close-packed (CCP / FCC) lattice.
     *
     * Particles are arranged in a face-centered cubic (FCC) structure using the conventional
     * unit cell with 4 atoms at fractional positions:
     *   (0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)
     *
     * The number of cells along y and z are derived from the provided nx to preserve the
     * domain aspect ratio. The globalParticleIdx maps as:
     *   basisIdx   = globalParticleIdx % 4         (atom within unit cell)
     *   cellIdx    = globalParticleIdx / 4         (unit cell flat index)
     *   ix, iy, iz = decode cellIdx in row-major order over (nx, ny, nz)
     *
     * Typical usage in a setup:
     * @code
     *   using PlaceParticle = pmacc::spearhed::CCP<CS>;
     *
     *   auto placeParticleArgs() const {
     *       return std::make_tuple(
     *           pmacc::spearhed::computeCCPNumCells(numParticles, region.volume));
     *   }
     * @endcode
     */
    template<CoordinateSystem CS>
    requires(CS::dimension == 3)
    struct CCP
    {
        DINLINE constexpr void operator()(
            [[maybe_unused]] auto const& worker,
            auto& particle,
            auto const& particleRegion,
            uint32_t globalParticleIdx,
            uint32_t numCells) const
        {
            // TODO this only works for Cartesian systems
            // For others either figure out how to do it in those systems, or convert to and from cartesian
            using Scalar = typename CS::T_Axis;
            auto const& aabb = particleRegion.volume;

            Scalar const Lx = aabb.max[tags::x] - aabb.min[tags::x];
            Scalar const Ly = aabb.max[tags::y] - aabb.min[tags::y];
            Scalar const Lz = aabb.max[tags::z] - aabb.min[tags::z];

            uint32_t const nx = numCells;
            // Round to nearest integer while preserving aspect ratio
            auto const ny = static_cast<uint32_t>(static_cast<Scalar>(nx) * Ly / Lx + Scalar{0.5});
            auto const nz = static_cast<uint32_t>(static_cast<Scalar>(nx) * Lz / Lx + Scalar{0.5});

            uint32_t const basisIdx = globalParticleIdx % 4u;
            uint32_t const cellIdx = globalParticleIdx / 4u;

            uint32_t const iz = cellIdx / (nx * ny);
            uint32_t const iy = (cellIdx % (nx * ny)) / nx;
            uint32_t const ix = cellIdx % nx;

            Scalar const dx = Lx / static_cast<Scalar>(nx);
            Scalar const dy = Ly / static_cast<Scalar>(ny > 0u ? ny : 1u);
            Scalar const dz = Lz / static_cast<Scalar>(nz > 0u ? nz : 1u);

            // FCC basis in fractional unit-cell coordinates
            Scalar bx, by, bz;
            if(basisIdx == 0u)
            {
                bx = Scalar{0};
                by = Scalar{0};
                bz = Scalar{0};
            }
            else if(basisIdx == 1u)
            {
                bx = Scalar{0.5};
                by = Scalar{0.5};
                bz = Scalar{0};
            }
            else if(basisIdx == 2u)
            {
                bx = Scalar{0.5};
                by = Scalar{0};
                bz = Scalar{0.5};
            }
            else
            {
                bx = Scalar{0};
                by = Scalar{0.5};
                bz = Scalar{0.5};
            }

            *particle[tags::relativePos][tags::x] = aabb.min[tags::x] + (static_cast<Scalar>(ix) + bx) * dx;
            *particle[tags::relativePos][tags::y] = aabb.min[tags::y] + (static_cast<Scalar>(iy) + by) * dy;
            *particle[tags::relativePos][tags::z] = aabb.min[tags::z] + (static_cast<Scalar>(iz) + bz) * dz;
        }
    };

} // namespace pmacc::spearhed
