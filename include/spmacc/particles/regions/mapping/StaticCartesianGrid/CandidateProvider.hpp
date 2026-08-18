/* Copyright 2026 Tapish Narwal
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * (at your option) any later version.
 */

#pragma once

#include "spmacc/particles/regions/mapping/StaticCartesianGrid/FixedCartesianGrid.hpp"
#include "spmacc/particles/spatial/CandidateProvider.hpp"

#include <pmacc/attribute/FunctionSpecifier.hpp>

#include <array>
#include <cstdint>

namespace pmacc::spearhed
{
    /**
     * @brief Implicit same-grid candidate enumerator.
     *
     * This intentionally tests target and source cell geometry for every
     * possible source slot rather than deriving an unverified integer stencil
     * radius.  It is the Phase-5 correctness baseline: candidate data is not
     * materialised in CSR buffers, and a later specialised stencil may only
     * replace this implementation after proving equivalent coverage.
     */
    template<CoordinateSystem CS>
    struct FixedGridCandidateProviderView
    {
        using Axis = typename CS::T_Axis;
        using Grid = FixedCartesianGrid<CS>;
        using Vec = typename Grid::Vec;

        Grid grid;
        Axis cutoff;
        std::array<uint32_t, CS::dimension> imageRadius;

        template<typename Fn>
        HDINLINE void forEachCandidate(uint32_t targetBucketSlot, Fn&& fn) const
        {
            auto const targetSearchBounds = grid.ownershipBounds(targetBucketSlot).expand(cutoff);
            auto const targetOrigin = grid.chart(targetBucketSlot).origin;
            uint32_t const numSources = grid.bucketCount();
            uint32_t const numImages = imageCount();

            // Loop bounds depend only on the target slot and the provider value. Every worker in
            // an interaction block therefore follows identical control flow, as CandidateProvider
            // requires for callbacks containing block barriers.
            for(uint32_t sourceBucketSlot = 0u; sourceBucketSlot < numSources; ++sourceBucketSlot)
            {
                auto const sourceBounds = grid.ownershipBounds(sourceBucketSlot);
                auto const sourceOrigin = grid.chart(sourceBucketSlot).origin;
                for(uint32_t image = 0u; image < numImages; ++image)
                {
                    auto const translation = imageTranslation(image);
                    auto translatedBounds = sourceBounds;
                    translatedBounds.min = translatedBounds.min + translation;
                    translatedBounds.max = translatedBounds.max + translation;
                    if(intersects(targetSearchBounds, translatedBounds))
                    {
                        bool const isUnshiftedImage = isZero(translation);
                        fn(RegionCandidate{
                            sourceBucketSlot,
                            sourceOrigin + translation - targetOrigin,
                            isUnshiftedImage});
                    }
                }
            }
        }

    private:
        [[nodiscard]] HDINLINE uint32_t imageCount() const
        {
            uint32_t count = 1u;
            for(uint32_t axis = 0u; axis < CS::dimension; ++axis)
                count *= grid.periodic[axis] ? (2u * imageRadius[axis] + 1u) : 1u;
            return count;
        }

        [[nodiscard]] HDINLINE Vec imageTranslation(uint32_t image) const
        {
            Vec translation{Axis{0}};
            pmacc::spearhed::for_each_enum_tag<CS>(
                [&](auto i, auto tag)
                {
                    if(!grid.periodic[i.value])
                        return;
                    uint32_t const width = 2u * imageRadius[i.value] + 1u;
                    int const coordinate = static_cast<int>(image % width) - static_cast<int>(imageRadius[i.value]);
                    image /= width;
                    translation[tag] = static_cast<Axis>(coordinate) * (grid.domain.max[tag] - grid.domain.min[tag]);
                });
            return translation;
        }

        [[nodiscard]] HDINLINE bool isZero(Vec const& value) const
        {
            return pmacc::spearhed::all_of_tag<CS>([&](auto tag) { return value[tag] == Axis{0}; });
        }
    };

    /** @brief Host value for an implicit fixed-grid candidate provider. */
    template<CoordinateSystem CS>
    struct FixedGridCandidateProvider
    {
        using Axis = typename CS::T_Axis;
        using Grid = FixedCartesianGrid<CS>;

        FixedGridCandidateProvider(Grid const& grid, Axis cutoff) : grid(grid), cutoff(cutoff)
        {
            pmacc::spearhed::for_each_enum_tag<CS>(
                [&](auto i, auto tag)
                {
                    if(!grid.periodic[i.value])
                        return;
                    Axis const domainExtent = grid.domain.max[tag] - grid.domain.min[tag];
                    // Include every image that can reach the cutoff plus one boundary image. The
                    // geometry test in the view removes superfluous candidates.
                    imageRadius[i.value] = static_cast<uint32_t>(cutoff / domainExtent) + 1u;
                });
        }

        [[nodiscard]] CandidateProvider auto deviceView() const
        {
            return FixedGridCandidateProviderView<CS>{grid, cutoff, imageRadius};
        }

        Grid grid;
        Axis cutoff;
        std::array<uint32_t, CS::dimension> imageRadius{};
    };
} // namespace pmacc::spearhed
