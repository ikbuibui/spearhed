/* Copyright 2026 Tapish Narwal
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
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "spmacc/particles/initialization/SC.hpp"
#include "spmacc/particles/regions/AABB.hpp"
#include "spmacc/topology/CoordinateSystem.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace pmacc::spearhed
{
    /** Physical input for one region of an equal-mass SC lattice plan. */
    template<CoordinateSystem CS>
    struct SCRegionInput
    {
        AABB<CS> volume;
        double density{};
    };

    /** Planned integer lattice and host-side diagnostics for one region. */
    template<CoordinateSystem CS>
    struct SCRegionPlan
    {
        AABB<CS> volume;
        SCShape<CS> shape;
        uint32_t numParticles{};
        double targetParticles{};
        double inputDensity{};
        std::array<double, CS::dimension> effectiveSpacing{};
        double discreteDensity{};
        double relativeDensityError{};
    };

    /**
     * Equal-particle-mass SC plan for a fixed number of regions.
     *
     * This record contains no owning storage and can be copied into smaller
     * kernel arguments containing only the required region shapes and mass.
     */
    template<CoordinateSystem CS, std::size_t NumRegions>
    struct EqualMassSCPlan
    {
        std::array<SCRegionPlan<CS>, NumRegions> regions{};
        uint32_t targetParticles{};
        uint32_t actualParticles{};
        double particleMassDouble{};
        typename CS::T_Axis particleMass{};
        double physicalMass{};
        double relativeParticleCountError{};
        double maxRelativeDensityError{};
        double maxSpacingAnisotropy{};
    };

    /** Host-side limits and acceptance criterion for SC shape selection. */
    struct SCPlanOptions
    {
        double maxRelativeDensityError = 0.01;
        uint32_t maxCandidateRadius = 4u;
        uint64_t maxCandidateCombinations = 2'000'000u;
    };

    namespace detail
    {
        template<CoordinateSystem CS>
        struct SCCandidate
        {
            SCShape<CS> shape;
            uint32_t numParticles{};
            std::array<double, CS::dimension> effectiveSpacing{};
            double spacingAnisotropy{};
        };

        template<CoordinateSystem CS, std::size_t NumRegions>
        struct SCSelection
        {
            std::array<SCCandidate<CS>, NumRegions> regions{};
            uint32_t actualParticles{};
            double particleMass{};
            double maxRelativeDensityError{};
            double maxSpacingAnisotropy{};
            double relativeParticleCountError{};
            bool valid = false;
        };

        template<CoordinateSystem CS, std::size_t NumRegions>
        [[nodiscard]] bool selectionIsBetter(
            SCSelection<CS, NumRegions> const& candidate,
            SCSelection<CS, NumRegions> const& current)
        {
            if(!current.valid)
                return true;
            if(candidate.maxRelativeDensityError != current.maxRelativeDensityError)
                return candidate.maxRelativeDensityError < current.maxRelativeDensityError;
            if(candidate.maxSpacingAnisotropy != current.maxSpacingAnisotropy)
                return candidate.maxSpacingAnisotropy < current.maxSpacingAnisotropy;
            if(candidate.relativeParticleCountError != current.relativeParticleCountError)
                return candidate.relativeParticleCountError < current.relativeParticleCountError;

            for(std::size_t region = 0u; region < NumRegions; ++region)
            {
                auto const& lhs = candidate.regions[region].shape;
                auto const& rhs = current.regions[region].shape;
                if(lhs.cells == rhs.cells)
                    continue;
                return std::lexicographical_compare(
                    lhs.cells.begin(),
                    lhs.cells.end(),
                    rhs.cells.begin(),
                    rhs.cells.end());
                ;
            }
            return false;
        }

        template<CoordinateSystem CS>
        [[nodiscard]] std::vector<SCCandidate<CS>> makeRegionCandidates(
            std::array<double, CS::dimension> const& extents,
            std::array<double, CS::dimension> const& idealCells,
            uint32_t radius,
            uint64_t maxCandidates)
        {
            std::array<uint32_t, CS::dimension> lowerBounds{};
            std::array<uint32_t, CS::dimension> upperBounds{};
            uint64_t candidateCount = 1u;
            for(std::size_t axis = 0u; axis < CS::dimension; ++axis)
            {
                auto const floored = static_cast<uint64_t>(std::floor(idealCells[axis]));
                auto const ceiled = static_cast<uint64_t>(std::ceil(idealCells[axis]));
                auto const lower = (floored > radius) ? floored - radius : uint64_t{1u};
                auto const upper
                    = std::min<uint64_t>(ceiled + static_cast<uint64_t>(radius), std::numeric_limits<uint32_t>::max());
                auto const axisCandidateCount = upper - lower + 1u;
                if(candidateCount > maxCandidates / axisCandidateCount)
                    throw std::length_error("SC plan region candidate search exceeds the configured limit");

                candidateCount *= axisCandidateCount;
                lowerBounds[axis] = static_cast<uint32_t>(lower);
                upperBounds[axis] = static_cast<uint32_t>(upper);
            }
            if(candidateCount > static_cast<uint64_t>(std::numeric_limits<std::size_t>::max()))
                throw std::length_error("SC plan region candidate count exceeds host address space");

            std::vector<SCCandidate<CS>> result;
            result.reserve(static_cast<std::size_t>(candidateCount));
            SCShape<CS> shape{};
            auto enumerateAxes = [&](auto&& self, std::size_t axis) -> void
            {
                if(axis != CS::dimension)
                {
                    for(uint64_t value = lowerBounds[axis]; value <= upperBounds[axis]; ++value)
                    {
                        shape.cells[axis] = static_cast<uint32_t>(value);
                        self(self, axis + 1u);
                    }
                    return;
                }

                uint32_t numParticles = 0u;
                try
                {
                    numParticles = checkedParticleCount(shape);
                }
                catch(std::overflow_error const&)
                {
                    // The current frame list cannot allocate this shape.
                    return;
                }

                std::array<double, CS::dimension> spacing{};
                double minimumSpacing = std::numeric_limits<double>::max();
                double maximumSpacing = 0.0;
                for(std::size_t i = 0u; i < CS::dimension; ++i)
                {
                    spacing[i] = extents[i] / static_cast<double>(shape.cells[i]);
                    minimumSpacing = std::min(minimumSpacing, spacing[i]);
                    maximumSpacing = std::max(maximumSpacing, spacing[i]);
                }

                result.push_back({shape, numParticles, spacing, maximumSpacing / minimumSpacing - 1.0});
            };
            enumerateAxes(enumerateAxes, 0u);

            if(result.empty())
                throw std::overflow_error("SC plan has no shape within the uint32_t particle capacity");
            return result;
        }

        template<CoordinateSystem CS, std::size_t NumRegions>
        [[nodiscard]] SCSelection<CS, NumRegions> selectShapes(
            std::array<std::vector<SCCandidate<CS>>, NumRegions> const& candidates,
            std::array<double, NumRegions> const& volumes,
            std::array<double, NumRegions> const& densities,
            double physicalMass,
            uint32_t targetParticles,
            uint64_t maxCandidateCombinations)
        {
            uint64_t combinations = 1u;
            for(auto const& regionCandidates : candidates)
            {
                auto const count = static_cast<uint64_t>(regionCandidates.size());
                if(count == 0u || combinations > maxCandidateCombinations / count)
                    throw std::length_error("SC plan candidate search exceeds the configured combination limit");
                combinations *= count;
            }

            SCSelection<CS, NumRegions> best{};
            std::array<SCCandidate<CS>, NumRegions> selected{};
            auto enumerateRegions = [&](auto&& self, std::size_t region) -> void
            {
                if(region != NumRegions)
                {
                    for(auto const& candidate : candidates[region])
                    {
                        selected[region] = candidate;
                        self(self, region + 1u);
                    }
                    return;
                }

                uint32_t actualParticles = 0u;
                for(auto const& candidate : selected)
                {
                    if(actualParticles > std::numeric_limits<uint32_t>::max() - candidate.numParticles)
                        return;
                    actualParticles += candidate.numParticles;
                }

                auto const particleMass = physicalMass / static_cast<double>(actualParticles);
                double maxDensityError = 0.0;
                double maxAnisotropy = 0.0;
                for(std::size_t i = 0u; i < NumRegions; ++i)
                {
                    auto const discreteDensity
                        = particleMass * static_cast<double>(selected[i].numParticles) / volumes[i];
                    auto const densityError = std::abs(discreteDensity - densities[i]) / densities[i];
                    maxDensityError = std::max(maxDensityError, densityError);
                    maxAnisotropy = std::max(maxAnisotropy, selected[i].spacingAnisotropy);
                }
                auto const countError
                    = std::abs(static_cast<double>(actualParticles) - static_cast<double>(targetParticles))
                      / static_cast<double>(targetParticles);

                SCSelection<CS, NumRegions> const
                    score{selected, actualParticles, particleMass, maxDensityError, maxAnisotropy, countError, true};
                if(selectionIsBetter(score, best))
                    best = score;
            };
            enumerateRegions(enumerateRegions, 0u);

            if(!best.valid)
                throw std::overflow_error("SC plan particle count exceeds uint32_t");
            return best;
        }
    } // namespace detail

    /**
     * Derive equal-mass simple-cubic lattices from physical regions and an
     * approximate total particle count.
     *
     * All planning arithmetic is performed in double precision. Candidate
     * shape tuples are scored lexicographically by maximum relative density
     * error, maximum per-region spacing anisotropy, total-count error, and
     * finally the region-major cell arrays. Spacing anisotropy is
     * `max(spacing) / min(spacing) - 1`.
     *
     * @throws std::invalid_argument for invalid input geometry, density,
     *         target count, or options
     * @throws std::overflow_error when a value cannot be represented by the
     *         plan records
     * @throws std::length_error when the configured candidate-search limit is
     *         exceeded
     * @throws std::runtime_error when the configured candidate search cannot
     *         meet the requested density-error tolerance
     */
    template<CoordinateSystem CS, std::size_t NumRegions>
    requires std::floating_point<typename CS::T_Axis> && (NumRegions > 0u) && (CS::dimension > 0u)
    [[nodiscard]] EqualMassSCPlan<CS, NumRegions> makeEqualMassSCPlan(
        std::array<SCRegionInput<CS>, NumRegions> const& inputs,
        uint32_t targetParticles,
        SCPlanOptions const& options = {})
    {
        if(targetParticles == 0u)
            throw std::invalid_argument("SC plan target particle count must be nonzero");
        if(!std::isfinite(options.maxRelativeDensityError) || options.maxRelativeDensityError < 0.0)
            throw std::invalid_argument("SC plan density-error tolerance must be finite and non-negative");
        if(options.maxCandidateRadius == 0u)
            throw std::invalid_argument("SC plan maximum candidate radius must be nonzero");
        if(options.maxCandidateCombinations == 0u)
            throw std::invalid_argument("SC plan candidate combination limit must be nonzero");

        std::array<std::array<double, CS::dimension>, NumRegions> extents{};
        std::array<double, NumRegions> volumes{};
        std::array<double, NumRegions> physicalMasses{};
        double physicalMass = 0.0;

        for(std::size_t region = 0u; region < NumRegions; ++region)
        {
            auto const& input = inputs[region];
            if(!std::isfinite(input.density) || input.density <= 0.0)
                throw std::invalid_argument("SC plan region density must be finite and positive");

            double regionVolume = 1.0;
            bool invalidAABB = false;
            for_each_enum_tag<CS>(
                [&](auto axis, auto tag)
                {
                    auto const origin = static_cast<double>(input.volume.origin[tag]);
                    auto const minimum = static_cast<double>(input.volume.min[tag]);
                    auto const maximum = static_cast<double>(input.volume.max[tag]);
                    if(!std::isfinite(origin) || !std::isfinite(minimum) || !std::isfinite(maximum)
                       || maximum <= minimum)
                    {
                        invalidAABB = true;
                        return;
                    }

                    auto const extent = maximum - minimum;
                    extents[region][axis.value] = extent;
                    regionVolume *= extent;
                });
            if(invalidAABB)
                throw std::invalid_argument("SC plan AABB coordinates must be finite with positive extents");
            if(!std::isfinite(regionVolume) || regionVolume <= 0.0)
                throw std::overflow_error("SC plan region volume is not representable in double precision");
            auto const regionMass = input.density * regionVolume;
            if(!std::isfinite(regionMass) || regionMass <= 0.0)
                throw std::overflow_error("SC plan region physical mass is not representable in double precision");
            if(physicalMass > std::numeric_limits<double>::max() - regionMass)
                throw std::overflow_error("SC plan total physical mass exceeds double precision");

            volumes[region] = regionVolume;
            physicalMasses[region] = regionMass;
            physicalMass += regionMass;
        }

        std::array<std::array<double, CS::dimension>, NumRegions> idealCells{};
        for(std::size_t region = 0u; region < NumRegions; ++region)
        {
            auto const regionalTarget = static_cast<double>(targetParticles) * (physicalMasses[region] / physicalMass);
            auto const sitesPerVolume = regionalTarget / volumes[region];
            auto const inverseDimension = 1.0 / static_cast<double>(CS::dimension);
            auto const inverseSpacing = std::pow(sitesPerVolume, inverseDimension);
            if(!std::isfinite(regionalTarget) || regionalTarget <= 0.0 || !std::isfinite(inverseSpacing)
               || inverseSpacing <= 0.0)
                throw std::overflow_error("SC plan ideal lattice shape is not representable in double precision");

            for(std::size_t axis = 0u; axis < CS::dimension; ++axis)
            {
                auto const ideal = extents[region][axis] * inverseSpacing;
                if(!std::isfinite(ideal) || ideal <= 0.0
                   || ideal > static_cast<double>(std::numeric_limits<uint32_t>::max()))
                    throw std::overflow_error("SC plan axis cell count exceeds uint32_t");
                idealCells[region][axis] = ideal;
            }
        }

        std::array<double, NumRegions> densities{};
        for(std::size_t region = 0u; region < NumRegions; ++region)
            densities[region] = inputs[region].density;

        detail::SCSelection<CS, NumRegions> selection{};
        for(uint32_t radius = 1u;; ++radius)
        {
            std::array<std::vector<detail::SCCandidate<CS>>, NumRegions> candidates;
            for(std::size_t region = 0u; region < NumRegions; ++region)
            {
                candidates[region] = detail::makeRegionCandidates<CS>(
                    extents[region],
                    idealCells[region],
                    radius,
                    options.maxCandidateCombinations);
            }

            selection = detail::selectShapes<CS>(
                candidates,
                volumes,
                densities,
                physicalMass,
                targetParticles,
                options.maxCandidateCombinations);
            if(selection.maxRelativeDensityError <= options.maxRelativeDensityError
               || radius == options.maxCandidateRadius)
                break;
        }

        if(!selection.valid || selection.maxRelativeDensityError > options.maxRelativeDensityError)
            throw std::runtime_error("SC plan could not meet the configured density-error tolerance");

        using Scalar = typename CS::T_Axis;
        if(!std::isfinite(selection.particleMass) || selection.particleMass <= 0.0
           || selection.particleMass > static_cast<double>(std::numeric_limits<Scalar>::max()))
            throw std::overflow_error("SC plan particle mass exceeds the coordinate scalar range");
        auto const scalarMass = static_cast<Scalar>(selection.particleMass);
        if(scalarMass <= Scalar{0})
            throw std::overflow_error("SC plan particle mass underflows the coordinate scalar range");

        EqualMassSCPlan<CS, NumRegions> plan{};
        plan.targetParticles = targetParticles;
        plan.actualParticles = selection.actualParticles;
        plan.particleMassDouble = selection.particleMass;
        plan.particleMass = scalarMass;
        plan.physicalMass = physicalMass;
        plan.relativeParticleCountError = selection.relativeParticleCountError;
        plan.maxRelativeDensityError = selection.maxRelativeDensityError;
        plan.maxSpacingAnisotropy = selection.maxSpacingAnisotropy;

        for(std::size_t region = 0u; region < NumRegions; ++region)
        {
            auto const& selected = selection.regions[region];
            auto& regionPlan = plan.regions[region];
            regionPlan.volume = inputs[region].volume;
            regionPlan.shape = selected.shape;
            regionPlan.numParticles = selected.numParticles;
            regionPlan.targetParticles
                = static_cast<double>(targetParticles) * (physicalMasses[region] / physicalMass);
            regionPlan.inputDensity = inputs[region].density;
            regionPlan.effectiveSpacing = selected.effectiveSpacing;
            regionPlan.discreteDensity
                = selection.particleMass * static_cast<double>(selected.numParticles) / volumes[region];
            regionPlan.relativeDensityError
                = std::abs(regionPlan.discreteDensity - regionPlan.inputDensity) / regionPlan.inputDensity;
        }
        return plan;
    }
} // namespace pmacc::spearhed
