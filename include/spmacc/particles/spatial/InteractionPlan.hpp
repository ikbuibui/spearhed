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

#include "spmacc/particles/regions/NeighbourBundle.hpp"
#include "spmacc/particles/spatial/PreparedRegionSet.hpp"

#include <tuple>
#include <type_traits>
#include <utility>

namespace pmacc::spearhed
{
    namespace detail
    {
        // Establish the customization name for the dependent call below. Concrete
        // mapping implementations provide overloads selected by their strategy tag.
        void makeInteractionPlanImpl() = delete;
    } // namespace detail

    /**
     * @brief Construct an interaction plan using the prepared mapping's strategy.
     *
     * Prepared mappings select their concrete implementation through
     * @c InteractionPlanStrategy. Control code therefore depends only on this
     * interface, while mapping headers provide the corresponding implementation.
     */
    template<typename T_Target, typename... T_Args>
    [[nodiscard]] constexpr decltype(auto) makeInteractionPlan(T_Target&& target, T_Args&&... args)
    {
        using detail::makeInteractionPlanImpl;
        return makeInteractionPlanImpl(
            typename std::remove_cvref_t<T_Target>::InteractionPlanStrategy{},
            std::forward<T_Target>(target),
            std::forward<T_Args>(args)...);
    }

    /**
     * @brief An owning interaction bundle tied to prepared decomposition states.
     *
     * The wrapped bundle owns candidate data (currently CSR buffers). Prepared
     * states retain the mapping metadata and make accidental use after a newer
     * preparation detectable in debug builds. Selected views retain the
     * existing NeighbourBundle lifetime rule: their parent plan must outlive
     * queued interaction kernels.
     */
    template<IsNeighbourBundle T_Bundle, PreparedRegionSet... T_Prepared>
    struct InteractionPlan
    {
        T_Bundle bundle;
        std::tuple<T_Prepared...> prepared;

        void assertCurrent() const
        {
            std::apply([](auto const&... state) { (state.assertCurrent(), ...); }, prepared);
        }

        template<typename Fn>
        void forEachEntry(Fn&& fn)
        {
            assertCurrent();
            bundle.forEachEntry(std::forward<Fn>(fn));
        }

        template<typename Fn>
        void forEachEntry(Fn&& fn) const
        {
            assertCurrent();
            bundle.forEachEntry(std::forward<Fn>(fn));
        }

        template<typename Fn>
        void forEachDeviceView(Fn&& fn)
        {
            assertCurrent();
            bundle.forEachDeviceView(std::forward<Fn>(fn));
        }

        template<typename Fn>
        void forEachDeviceView(Fn&& fn) const
        {
            assertCurrent();
            bundle.forEachDeviceView(std::forward<Fn>(fn));
        }

        auto makeDeviceViewTuple()
        {
            assertCurrent();
            return bundle.makeDeviceViewTuple();
        }

        auto makeDeviceViewTuple() const
        {
            assertCurrent();
            return bundle.makeDeviceViewTuple();
        }

        static constexpr std::size_t size()
        {
            return T_Bundle::size();
        }

        template<typename Pred>
        auto select(Pred predicate = {})
        {
            assertCurrent();
            return bundle.select(predicate);
        }

        template<typename Pred>
        auto select(Pred predicate = {}) const
        {
            assertCurrent();
            return bundle.select(predicate);
        }

        template<RoleTag R>
        auto selectByRole()
        {
            assertCurrent();
            return bundle.template selectByRole<R>();
        }

        template<RoleTag R>
        auto selectByRole() const
        {
            assertCurrent();
            return bundle.template selectByRole<R>();
        }

        template<SpeciesTag S>
        auto& bySpecies(S species = {})
        {
            assertCurrent();
            return bundle.bySpecies(species);
        }

        template<SpeciesTag S>
        auto const& bySpecies(S species = {}) const
        {
            assertCurrent();
            return bundle.bySpecies(species);
        }
    };

    namespace detail
    {
        template<IsNeighbourBundle Bundle, PreparedRegionSet... Prepared>
        struct IsNeighbourBundleImpl<InteractionPlan<Bundle, Prepared...>> : std::true_type
        {
        };
    } // namespace detail
} // namespace pmacc::spearhed
