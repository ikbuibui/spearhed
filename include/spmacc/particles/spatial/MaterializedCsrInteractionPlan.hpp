/* Copyright 2026 Tapish Narwal
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#pragma once

#include "spmacc/particles/regions/NeighbourRegions.hpp"
#include "spmacc/particles/spatial/InteractionPlan.hpp"
#include "spmacc/particles/spatial/InteractionQuery.hpp"

#include <cstdint>
#include <tuple>
#include <utility>

namespace pmacc::spearhed
{
    /**
     * @brief Prepared state whose particle buckets expose material occupancy for CSR fallback construction.
     *
     * This is a capability of the target/source pair rather than a requirement of
     * PreparedRegionSet. Fixed grids will provide specialised implicit providers;
     * material layouts from different decomposition groups use this fallback meanwhile.
     */
    template<typename T>
    concept MaterializedCsrPreparedRegionSet = PreparedRegionSet<T> && requires(T const& prepared, uint32_t slot) {
        prepared.store();
        prepared.store().getDeviceDataBox()[static_cast<int>(slot)].spatial.occupancy;
        prepared.store().getDeviceDataBox()[static_cast<int>(slot)].spatial.chart.origin;
    };

    /**
     * @brief Materialise a CSR plan for prepared material-like target and source states.
     *
     * The prepared states can belong to distinct decompositions and decomposition groups.
     * Construction only reads their current metadata; it never prepares or otherwise
     * mutates either group. The returned plan owns all CSR buffers and retains the
     * handles needed for generation checks and queued-kernel lifetime.
     */
    template<
        MaterializedCsrPreparedRegionSet T_Target,
        typename T_Radius,
        MaterializedCsrPreparedRegionSet... T_Sources>
    [[nodiscard]] auto makeMaterializedCsrInteractionPlan(
        T_Target const& target,
        InteractionQuery<T_Radius> const& query,
        T_Sources const&... sources)
    {
        target.assertCurrent();
        if constexpr(sizeof...(sources) > 0u)
            (sources.assertCurrent(), ...);

        auto bundle = calculateNeighbours(target.store(), query.radius, sources.store()...);
        using Plan = InteractionPlan<
            std::remove_cvref_t<decltype(bundle)>,
            std::remove_cvref_t<T_Target>,
            std::remove_cvref_t<T_Sources>...>;
        return Plan{std::move(bundle), std::tuple{target, sources...}};
    }

    /** @brief Generic materialised-CSR fallback for a typed interaction query. */
    template<
        MaterializedCsrPreparedRegionSet T_Target,
        typename T_Radius,
        MaterializedCsrPreparedRegionSet... T_Sources>
    [[nodiscard]] auto makeInteractionPlan(
        T_Target const& target,
        InteractionQuery<T_Radius> const& query,
        T_Sources const&... sources)
    {
        return makeMaterializedCsrInteractionPlan(target, query, sources...);
    }

    /** @brief Compatibility overload for callers that pass only an interaction radius. */
    template<
        MaterializedCsrPreparedRegionSet T_Target,
        typename T_Radius,
        MaterializedCsrPreparedRegionSet... T_Sources>
    [[nodiscard]] auto makeInteractionPlan(
        T_Target const& target,
        T_Radius interactionRadius,
        T_Sources const&... sources)
    {
        return makeInteractionPlan(target, InteractionQuery{interactionRadius}, sources...);
    }
} // namespace pmacc::spearhed
