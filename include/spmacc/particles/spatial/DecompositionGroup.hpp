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

#include "spmacc/particles/regions/RegionRole.hpp"

#include <concepts>
#include <tuple>
#include <type_traits>
#include <utility>

namespace pmacc::spearhed
{
    /**
     * @brief Setup-level declaration of species that share a material-AABB decomposition.
     *
     * The declaration names species membership only. It deliberately does not
     * identify particle storage, which remains one ParticleRegionBuffer per
     * species.
     */
    template<SpeciesTag... T_Species>
    struct MaterialAabbSpeciesDecompositionGroup
    {
        using SpeciesList = std::tuple<T_Species...>;
    };

    namespace detail
    {
        template<typename T>
        struct IsMaterialAabbSpeciesDecompositionGroup : std::false_type
        {
        };

        template<SpeciesTag... T_Species>
        struct IsMaterialAabbSpeciesDecompositionGroup<MaterialAabbSpeciesDecompositionGroup<T_Species...>>
            : std::true_type
        {
        };

        template<typename T>
        inline constexpr bool isMaterialAabbSpeciesDecompositionGroup
            = IsMaterialAabbSpeciesDecompositionGroup<T>::value;

        template<typename T_Species, typename T_Group>
        struct GroupSpeciesCount : std::integral_constant<std::size_t, 0u>
        {
        };

        template<typename T_Species, SpeciesTag... T_GroupSpecies>
        struct GroupSpeciesCount<T_Species, MaterialAabbSpeciesDecompositionGroup<T_GroupSpecies...>>
            : std::integral_constant<std::size_t, (std::size_t{0u} + ... + std::same_as<T_Species, T_GroupSpecies>)>
        {
        };

        template<typename T_Species, typename... T_Groups>
        inline constexpr std::size_t groupSpeciesCount
            = (std::size_t{0u} + ... + GroupSpeciesCount<T_Species, T_Groups>::value);

        template<typename T_Group, typename T_SpeciesList>
        struct GroupOnlyUses : std::false_type
        {
        };

        template<SpeciesTag... T_GroupSpecies, SpeciesTag... T_Species>
        struct GroupOnlyUses<MaterialAabbSpeciesDecompositionGroup<T_GroupSpecies...>, std::tuple<T_Species...>>
            : std::bool_constant<
                  (sizeof...(T_GroupSpecies) > 0u)
                  && ((GroupSpeciesCount<T_GroupSpecies, MaterialAabbSpeciesDecompositionGroup<T_Species...>>::value
                       == 1u)
                      && ...)>
        {
        };

        template<typename T_Assignment, typename T_SpeciesList>
        struct IsDecompositionGroupAssignmentFor : std::false_type
        {
        };

        template<typename... T_Groups, SpeciesTag... T_Species>
        struct IsDecompositionGroupAssignmentFor<std::tuple<T_Groups...>, std::tuple<T_Species...>>
            : std::bool_constant<
                  (sizeof...(T_Groups) > 0u) && (isMaterialAabbSpeciesDecompositionGroup<T_Groups> && ...)
                  && (GroupOnlyUses<T_Groups, std::tuple<T_Species...>>::value && ...)
                  && ((groupSpeciesCount<T_Species, T_Groups...> == 1u) && ...)>
        {
        };

        template<typename T_SpeciesList>
        struct DefaultDecompositionGroups;

        template<SpeciesTag... T_Species>
        struct DefaultDecompositionGroups<std::tuple<T_Species...>>
        {
            using type = std::tuple<MaterialAabbSpeciesDecompositionGroup<T_Species...>>;
        };

        template<typename T_Setup, typename T_Registry, typename = void>
        struct DecompositionGroupsFor
        {
            using type = typename DefaultDecompositionGroups<typename T_Registry::List>::type;
        };

        template<typename T_Setup, typename T_Registry>
        struct DecompositionGroupsFor<T_Setup, T_Registry, std::void_t<typename T_Setup::DecompositionGroups>>
        {
            using type = typename T_Setup::DecompositionGroups;
        };
    } // namespace detail

    /** @brief The setup's declared groups, or one backward-compatible material group when omitted. */
    template<typename T_Setup, SpeciesRegistryTag T_Registry>
    using DecompositionGroupsFor = typename detail::DecompositionGroupsFor<T_Setup, T_Registry>::type;

    /** @brief True iff every registered species occurs in exactly one declared decomposition group. */
    template<typename T_Assignment, typename T_Registry>
    concept DecompositionGroupAssignmentFor
        = SpeciesRegistryTag<T_Registry>
          && detail::IsDecompositionGroupAssignmentFor<T_Assignment, typename T_Registry::List>::value;

    /** @brief Controls when a decomposition group's mapping is prepared. */
    enum class DecompositionGroupPreparation
    {
        EveryMotion,
        OnInvalidation
    };

    /**
     * @brief Owns one decomposition lifecycle for a species decomposition group.
     *
     * A dynamic group is prepared on every call after particles move. A static
     * group is prepared once, then must be explicitly invalidated when its
     * storage or geometry changes. In both cases preparedFor() is read-only and
     * forwards the decomposition's generation-checked handle.
     */
    template<typename T_Decomposition>
    class DecompositionGroup
    {
    public:
        explicit DecompositionGroup(
            T_Decomposition decomposition,
            DecompositionGroupPreparation preparation = DecompositionGroupPreparation::EveryMotion)
            : m_decomposition(std::move(decomposition))
            , m_preparation(preparation)
        {
        }

        void prepareAfterMotion()
        {
            if(m_preparation == DecompositionGroupPreparation::OnInvalidation && m_isPrepared)
                return;

            m_decomposition.prepareAfterMotion();
            m_isPrepared = true;
        }

        /** @brief Require the next prepareAfterMotion() to establish a new mapping generation. */
        void invalidate()
        {
            m_isPrepared = false;
        }

        [[nodiscard]] bool isPrepared() const
        {
            return m_isPrepared;
        }

        template<typename T_Store>
        [[nodiscard]] auto preparedFor(T_Store& store) const
        {
            return m_decomposition.preparedFor(store);
        }

        [[nodiscard]] T_Decomposition& decomposition()
        {
            return m_decomposition;
        }

        [[nodiscard]] T_Decomposition const& decomposition() const
        {
            return m_decomposition;
        }

    private:
        T_Decomposition m_decomposition;
        DecompositionGroupPreparation m_preparation;
        bool m_isPrepared = false;
    };

    template<typename T_Decomposition>
    DecompositionGroup(T_Decomposition, DecompositionGroupPreparation = DecompositionGroupPreparation::EveryMotion)
        -> DecompositionGroup<T_Decomposition>;
} // namespace pmacc::spearhed
