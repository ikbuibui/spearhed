/* Copyright 2025-2026 Tapish Narwal
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

#pragma once

#include "spearhed/ParticleDefinition.hpp"
#include "spearhed/param.hpp"
#include "spearhed/plugins/openPMD/Position.hpp"
#include "spearhed/sph/CubicSplineKernel.hpp"
#include "spmacc/particles/initialization/SC.hpp"
#include "spmacc/particles/initialization/SCPlan.hpp"
#include "spmacc/particles/regions/AABB.hpp"
#include "spmacc/particles/regions/ParticleRegionBuffer.hpp"
#include "spmacc/particles/regions/RegionRole.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include <llamaLite/Record.hpp>

namespace spearhed
{
    namespace sod
    {
        struct InitialConditions
        {
            Real densityLeft = Real{1.0f};
            Real pressureLeft = Real{1.0f};
            Real densityRight = Real{0.125f};
            Real pressureRight = Real{0.1f};
        };

        using FluidPlan = pmacc::spearhed::EqualMassSCPlan<CS, 2u>;
        using FluidShapes = std::array<pmacc::spearhed::SCShape<CS>, 2u>;

        constexpr Real wallThickness()
        {
            return Real{2} * h0;
        }

        constexpr uint32_t targetFluidParticles()
        {
#if SIM_DIM == 1
            return 32000u;
#elif SIM_DIM == 2
            return 3150u;
#else
            return 3375u;
#endif
        }

        /** The primary physical fluid domain for the selected dimension. */
        inline auto fluidDomain()
        {
#if SIM_DIM == 1
            return pmacc::spearhed::AABB<CS>{{Real{0}}, {Real{-1}}, {Real{1}}};
#elif SIM_DIM == 2
            return pmacc::spearhed::AABB<CS>{{Real{0}, Real{0}}, {Real{-1}, Real{0}}, {Real{1}, Real{0.07f}}};
#else
            return pmacc::spearhed::AABB<CS>{
                {Real{0}, Real{0}, Real{0}},
                {Real{-1}, Real{0}, Real{0}},
                {Real{1}, Real{1} / Real{3}, Real{1} / Real{3}}};
#endif
        }

        /**
         * Split a domain at a global x coordinate while preserving its origin.
         *
         * @throws std::invalid_argument if the split is not strictly inside the
         *         domain's x extent
         */
        inline std::array<pmacc::spearhed::AABB<CS>, 2u> splitAABB(
            pmacc::spearhed::AABB<CS> const& domain,
            Real splitPosition)
        {
            using x_t = std::tuple_element_t<0, typename CS::tags>;
            auto const localSplit = splitPosition - domain.origin[x_t{}];
            if(!(domain.min[x_t{}] < localSplit && localSplit < domain.max[x_t{}]))
                throw std::invalid_argument("Sod discontinuity must lie strictly inside the fluid domain");

            auto left = domain;
            auto right = domain;
            left.max[x_t{}] = localSplit;
            right.min[x_t{}] = localSplit;
            return {left, right};
        }

        inline FluidPlan makeFluidPlan(
            pmacc::spearhed::AABB<CS> const& fluidDomain,
            Real splitPosition,
            InitialConditions const& initialConditions,
            uint32_t targetFluidParticles)
        {
            auto const regions = splitAABB(fluidDomain, splitPosition);
            std::array<pmacc::spearhed::SCRegionInput<CS>, 2u> const inputs{{
                {regions[0], static_cast<double>(initialConditions.densityLeft)},
                {regions[1], static_cast<double>(initialConditions.densityRight)},
            }};
            return pmacc::spearhed::makeEqualMassSCPlan(inputs, targetFluidParticles);
        }

        inline auto fullDomain(pmacc::spearhed::AABB<CS> const& fluidDomain)
        {
            return fluidDomain.expand(wallThickness());
        }

        /**
         * Add the legacy fixed-particle wall decomposition around a fluid AABB.
         *
         * Geometry is derived exclusively from the fluid domain. Faces are
         * assigned in axis order so edges and corners belong to the first face
         * that reaches them. Non-x faces are split at the discontinuity to
         * retain the adjacent left/right state assignment.
         */
        inline void addWalls(
            std::vector<pmacc::spearhed::AABB<CS>>& out,
            pmacc::spearhed::AABB<CS> const& fluidDomain,
            Real splitPosition)
        {
            auto const outerDomain = fullDomain(fluidDomain);
            pmacc::spearhed::for_each_enum_tag<CS>(
                [&](auto faceAxis, auto faceTag)
                {
                    auto lowerFace = fluidDomain;
                    auto upperFace = fluidDomain;
                    pmacc::spearhed::for_each_enum_tag<CS>(
                        [&](auto otherAxis, auto otherTag)
                        {
                            if constexpr(otherAxis.value > faceAxis.value)
                            {
                                lowerFace.min[otherTag] = outerDomain.min[otherTag];
                                lowerFace.max[otherTag] = outerDomain.max[otherTag];
                                upperFace.min[otherTag] = outerDomain.min[otherTag];
                                upperFace.max[otherTag] = outerDomain.max[otherTag];
                            }
                        });

                    lowerFace.min[faceTag] = outerDomain.min[faceTag];
                    lowerFace.max[faceTag] = fluidDomain.min[faceTag];
                    upperFace.min[faceTag] = fluidDomain.max[faceTag];
                    upperFace.max[faceTag] = outerDomain.max[faceTag];

                    if constexpr(faceAxis.value == 0u)
                    {
                        out.push_back(lowerFace);
                        out.push_back(upperFace);
                    }
                    else
                    {
                        auto const lowerParts = splitAABB(lowerFace, splitPosition);
                        auto const upperParts = splitAABB(upperFace, splitPosition);
                        out.insert(out.end(), lowerParts.begin(), lowerParts.end());
                        out.insert(out.end(), upperParts.begin(), upperParts.end());
                    }
                });
        }

        DINLINE constexpr std::size_t fluidRegionIndex(auto const& particleRegion, Real splitPosition)
        {
            using x_t = std::tuple_element_t<0, typename CS::tags>;
            auto const& volume = particleRegion.volume;
            auto const globalMaximum = volume.origin[x_t{}] + volume.max[x_t{}];
            return (globalMaximum <= splitPosition) ? 0u : 1u;
        }

        struct PlannedNumParticles
        {
            DINLINE constexpr uint32_t operator()(
                [[maybe_unused]] auto const& worker,
                auto const& particleRegion,
                FluidShapes const& shapes,
                Real splitPosition) const
            {
                return shapes[fluidRegionIndex(particleRegion, splitPosition)].numSites();
            }
        };

        struct PlannedPlaceParticle
        {
            DINLINE constexpr void operator()(
                auto const& worker,
                auto& particle,
                auto const& particleRegion,
                uint32_t globalParticleIdx,
                FluidShapes const& shapes,
                Real splitPosition,
                Real particleMass,
                InitialConditions const& initialConditions) const
            {
                auto const region = fluidRegionIndex(particleRegion, splitPosition);
                auto const rho = (region == 0u) ? initialConditions.densityLeft : initialConditions.densityRight;
                auto const pressure
                    = (region == 0u) ? initialConditions.pressureLeft : initialConditions.pressureRight;

                pmacc::spearhed::SC<CS>{}(worker, particle, particleRegion, globalParticleIdx, shapes[region]);
                pmacc::spearhed::for_each_tag<CS>([&](auto axisTag) { particle[vel][axisTag] = Real{0}; });

                particle[mass] = particleMass;
                particle[smoothingLength] = h0;
                particle[density] = rho;
                particle[internalEnergy] = pressure / ((gamma_eos - Real{1}) * rho);
            }
        };

        /** Spacing-based realization for the legacy boundary block. */
        struct LegacyBoundaryNumParticles
        {
            DINLINE constexpr uint32_t operator()(
                [[maybe_unused]] auto const& worker,
                auto const& particleRegion,
                std::array<Real, 2u> const& effectiveSpacing,
                Real splitPosition) const
            {
                auto const spacing = effectiveSpacing[fluidRegionIndex(particleRegion, splitPosition)];
                return pmacc::spearhed::makeSCShapeForTargetSpacing(particleRegion.volume, spacing).numSites();
            }
        };

        struct LegacyBoundaryPlaceParticle
        {
            DINLINE constexpr void operator()(
                auto const& worker,
                auto& particle,
                auto const& particleRegion,
                uint32_t globalParticleIdx,
                std::array<Real, 2u> const& effectiveSpacing,
                Real splitPosition,
                Real particleMass,
                InitialConditions const& initialConditions) const
            {
                auto const region = fluidRegionIndex(particleRegion, splitPosition);
                auto const rho = (region == 0u) ? initialConditions.densityLeft : initialConditions.densityRight;
                auto const pressure
                    = (region == 0u) ? initialConditions.pressureLeft : initialConditions.pressureRight;
                auto const shape
                    = pmacc::spearhed::makeSCShapeForTargetSpacing(particleRegion.volume, effectiveSpacing[region]);

                pmacc::spearhed::SC<CS>{}(worker, particle, particleRegion, globalParticleIdx, shape);
                pmacc::spearhed::for_each_tag<CS>([&](auto axisTag) { particle[vel][axisTag] = Real{0}; });

                particle[mass] = particleMass;
                particle[smoothingLength] = h0;
                particle[density] = rho;
                particle[internalEnergy] = pressure / ((gamma_eos - Real{1}) * rho);
            }
        };

        struct InteriorBlock
        {
            using Species = pmacc::spearhed::species::Default;
            using NumParticlesToCreate = PlannedNumParticles;
            using PlaceParticle = PlannedPlaceParticle;

            InteriorBlock(FluidPlan fluidPlan, Real splitPosition, InitialConditions initialConditions)
                : m_fluidPlan(std::move(fluidPlan))
                , m_splitPosition(splitPosition)
                , m_initialConditions(initialConditions)
            {
            }

            auto numParticlesToCreateArgs() const
            {
                return std::make_tuple(fluidShapes(), m_splitPosition);
            }

            auto placeParticleArgs() const
            {
                return std::make_tuple(fluidShapes(), m_splitPosition, m_fluidPlan.particleMass, m_initialConditions);
            }

            template<typename>
            void addRegions(std::vector<pmacc::spearhed::AABB<CS>>& out) const
            {
                out.push_back(m_fluidPlan.regions[0].volume);
                out.push_back(m_fluidPlan.regions[1].volume);
            }

        private:
            [[nodiscard]] FluidShapes fluidShapes() const
            {
                return {m_fluidPlan.regions[0].shape, m_fluidPlan.regions[1].shape};
            }

            FluidPlan m_fluidPlan;
            Real m_splitPosition;
            InitialConditions m_initialConditions;
        };

        struct BoundaryBlock
        {
            using Species = pmacc::spearhed::species::Boundary;
            using NumParticlesToCreate = LegacyBoundaryNumParticles;
            using PlaceParticle = LegacyBoundaryPlaceParticle;

            BoundaryBlock(
                pmacc::spearhed::AABB<CS> fluidDomain,
                FluidPlan const& fluidPlan,
                Real splitPosition,
                InitialConditions initialConditions)
                : m_fluidDomain(std::move(fluidDomain))
                , m_effectiveSpacing{
                      static_cast<Real>(fluidPlan.regions[0].effectiveSpacing[0]),
                      static_cast<Real>(fluidPlan.regions[1].effectiveSpacing[0])}
                , m_particleMass(fluidPlan.particleMass)
                , m_splitPosition(splitPosition)
                , m_initialConditions(initialConditions)
            {
            }

            auto numParticlesToCreateArgs() const
            {
                return std::make_tuple(m_effectiveSpacing, m_splitPosition);
            }

            auto placeParticleArgs() const
            {
                return std::make_tuple(m_effectiveSpacing, m_splitPosition, m_particleMass, m_initialConditions);
            }

            template<typename>
            void addRegions(std::vector<pmacc::spearhed::AABB<CS>>& out) const
            {
                addWalls(out, m_fluidDomain, m_splitPosition);
            }

            [[nodiscard]] auto domain() const
            {
                return fullDomain(m_fluidDomain);
            }

        private:
            pmacc::spearhed::AABB<CS> m_fluidDomain;
            std::array<Real, 2u> m_effectiveSpacing;
            Real m_particleMass;
            Real m_splitPosition;
            InitialConditions m_initialConditions;
        };
    } // namespace sod

    struct SodShockTube
    {
        explicit SodShockTube(
            pmacc::spearhed::AABB<CS> fluidDomain = sod::fluidDomain(),
            Real splitPosition = Real{0},
            sod::InitialConditions initialConditions = {},
            uint32_t targetFluidParticles = sod::targetFluidParticles())
            : fluidDomain(std::move(fluidDomain))
            , splitPosition(splitPosition)
            , initialConditions(initialConditions)
            , targetFluidParticles(targetFluidParticles)
            , fluidPlan(
                  sod::makeFluidPlan(
                      this->fluidDomain,
                      this->splitPosition,
                      this->initialConditions,
                      this->targetFluidParticles))
            , interior(fluidPlan, this->splitPosition, this->initialConditions)
            , boundary(this->fluidDomain, fluidPlan, this->splitPosition, this->initialConditions)
            , domain(boundary.domain())
        {
        }

        void printStartupDiagnostics(std::ostream& out = std::cout) const
        {
            out << "Sod SC fluid plan: target=" << fluidPlan.targetParticles
                << ", actual=" << fluidPlan.actualParticles << ", particle mass=" << fluidPlan.particleMass << '\n';
            for(std::size_t region = 0u; region < fluidPlan.regions.size(); ++region)
            {
                auto const& regionPlan = fluidPlan.regions[region];
                out << "  " << ((region == 0u) ? "left" : "right") << ": shape=[";
                for(std::size_t axis = 0u; axis < CS::dimension; ++axis)
                {
                    if(axis != 0u)
                        out << ", ";
                    out << regionPlan.shape.cells[axis];
                }
                out << "], count=" << regionPlan.numParticles << ", effective spacing=[";
                for(std::size_t axis = 0u; axis < CS::dimension; ++axis)
                {
                    if(axis != 0u)
                        out << ", ";
                    out << regionPlan.effectiveSpacing[axis];
                }
                out << "], relative density error=" << regionPlan.relativeDensityError << '\n';
            }
        }

        auto blocks() const
        {
            return std::tie(interior, boundary);
        }

        pmacc::spearhed::AABB<CS> const fluidDomain;
        Real const splitPosition;
        sod::InitialConditions const initialConditions;
        uint32_t const targetFluidParticles;
        sod::FluidPlan const fluidPlan;

        sod::InteriorBlock const interior;
        sod::BoundaryBlock const boundary;
        pmacc::spearhed::AABB<CS> const domain;

        KernelVariant kernelVariant = makeKernel(KernelType::CubicSpline);

        using OutputParticleRecord = ll::Record<
            spearhed::tags::idField,
            spearhed::output::positionField<CS>,
            spearhed::tags::massField<Real>,
            spearhed::tags::velField<CS>,
            spearhed::tags::densityField<Real>,
            spearhed::tags::internalEnergyField<Real>>;
    };

    using Setup = SodShockTube;
} // namespace spearhed
