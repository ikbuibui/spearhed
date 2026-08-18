/* Copyright 2026 Tapish Narwal
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * (at your option) any later version.
 */

#pragma once

#include "spmacc/memory/FramePointer.hpp"
#include "spmacc/particles/algorithms/FrameIndex.hpp"
#include "spmacc/particles/attributes/MultiMask.hpp"
#include "spmacc/particles/attributes/RelativePosition.hpp"
#include "spmacc/particles/regions/ParticleRegionBuffer.hpp"
#include "spmacc/particles/regions/mapping/StaticCartesianGrid/FixedCartesianGrid.hpp"

#include <pmacc/attribute/FunctionSpecifier.hpp>
#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/lockstep/ForEach.hpp>
#include <pmacc/lockstep/Kernel.hpp>
#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>
#include <pmacc/memory/shared/Allocate.hpp>

#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace pmacc::spearhed
{
    namespace detail
    {
        /** Translate a particle position from the chart implied by an existing fixed-grid bucket. */
        template<CoordinateSystem CS>
        struct FixedGridSourceChart
        {
            HDINLINE constexpr auto toWorld(
                FixedCartesianGrid<CS> const& grid,
                uint32_t sourceSlot,
                auto const& relativePosition) const
            {
                return grid.toWorld(sourceSlot, relativePosition);
            }
        };

        /** Translate a particle position from a setup material bucket's saved chart. */
        template<typename T_SourceCharts>
        struct MaterialSourceChart
        {
            T_SourceCharts charts;

            template<CoordinateSystem CS>
            HDINLINE constexpr auto toWorld(
                FixedCartesianGrid<CS> const& /* grid */,
                uint32_t sourceSlot,
                auto const& relativePosition) const
            {
                return charts[sourceSlot].toWorld(relativePosition);
            }
        };

        /** Count the destination cell of every live source particle without mutating storage. */
        template<typename T_Frame, CoordinateSystem CS, typename T_SourceChart>
        struct CountFixedCartesianDestinations
        {
            DINLINE void operator()(
                auto const& worker,
                auto sourceFramePtrs,
                auto sourceRegionIndices,
                uint32_t totalSourceFrames,
                T_SourceChart sourceChart,
                FixedCartesianGrid<CS> grid,
                auto destinationCounts,
                auto invalidDestination) const
            {
                uint32_t const frameIndex = worker.blockDomIdx();
                if(frameIndex >= totalSourceFrames)
                    return;

                pmacc::spearhed::memory::FramePointer<T_Frame> const sourceFrame{sourceFramePtrs[frameIndex]};
                uint32_t const sourceSlot = sourceRegionIndices[frameIndex];
                auto const forEachSlot = pmacc::lockstep::makeForEach<T_Frame::frameSize>(worker);
                forEachSlot(
                    [&](uint32_t slot)
                    {
                        auto particle = sourceFrame[slot];
                        if(!particle[tags::multiMask])
                            return;

                        typename FixedCartesianGrid<CS>::Location location{};
                        auto const worldPosition
                            = sourceChart.toWorld(grid, sourceSlot, particle[tags::relativePos].get());
                        if(!grid.locate(worldPosition, location))
                        {
                            alpaka::atomicAdd(
                                worker.getAcc(),
                                &invalidDestination[0],
                                1u,
                                ::alpaka::hierarchy::Blocks{});
                            return;
                        }
                        alpaka::atomicAdd(
                            worker.getAcc(),
                            &destinationCounts[location.slot],
                            1u,
                            ::alpaka::hierarchy::Blocks{});
                    });
            }
        };

        /** Replace every bucket chain by exactly enough fresh packed frames. */
        template<typename T_Frame>
        struct RebuildPackedFrameLists
        {
            DINLINE void operator()(
                auto const& worker,
                auto regions,
                uint32_t regionCount,
                auto destinationCounts,
                auto allocationFailed) const
            {
                uint32_t const regionSlot = worker.blockDomIdx();
                if(regionSlot >= regionCount)
                    return;

                auto onlyMaster = pmacc::lockstep::makeMaster(worker);
                onlyMaster(
                    [&]
                    {
                        auto& frameList = regions[regionSlot].particleFrameList;
                        frameList.detachAllFrames();

                        uint32_t const particleCount = destinationCounts[regionSlot];
                        frameList.setNumParticles(particleCount);
                        uint32_t const frameCount = frameList.numFrames();
                        for(uint32_t frameSlot = 0u; frameSlot < frameCount; ++frameSlot)
                        {
                            auto frame = frameList.getEmptyFrame(worker);
                            if(!frame.operator->())
                            {
                                alpaka::atomicAdd(
                                    worker.getAcc(),
                                    &allocationFailed[0],
                                    1u,
                                    ::alpaka::hierarchy::Blocks{});
                                frameList.setNumParticles(frameSlot * T_Frame::frameSize);
                                break;
                            }
                            uint32_t const remaining = particleCount - frameSlot * T_Frame::frameSize;
                            frame->liveParticles = remaining < T_Frame::frameSize ? remaining : T_Frame::frameSize;
                        }
                    });
            }
        };

        /** Copy whole particle records into the destination's contiguous packed slots and rebase positions. */
        template<typename T_Frame, CoordinateSystem CS, typename T_SourceChart>
        struct ScatterFixedCartesianParticles
        {
            DINLINE void operator()(
                auto const& worker,
                auto sourceFramePtrs,
                auto sourceRegionIndices,
                uint32_t totalSourceFrames,
                T_SourceChart sourceChart,
                FixedCartesianGrid<CS> grid,
                auto destinationFramePtrs,
                auto destinationFrameScan,
                auto destinationCursors) const
            {
                uint32_t const sourceFrameIndex = worker.blockDomIdx();
                if(sourceFrameIndex >= totalSourceFrames)
                    return;

                pmacc::spearhed::memory::FramePointer<T_Frame> const sourceFrame{sourceFramePtrs[sourceFrameIndex]};
                uint32_t const sourceSlot = sourceRegionIndices[sourceFrameIndex];
                auto const forEachSlot = pmacc::lockstep::makeForEach<T_Frame::frameSize>(worker);
                forEachSlot(
                    [&](uint32_t sourceParticleSlot)
                    {
                        auto sourceParticle = sourceFrame[sourceParticleSlot];
                        if(!sourceParticle[tags::multiMask])
                            return;

                        typename FixedCartesianGrid<CS>::Location location{};
                        auto const worldPosition
                            = sourceChart.toWorld(grid, sourceSlot, sourceParticle[tags::relativePos].get());
                        bool const located = grid.locate(worldPosition, location);
                        PMACC_ASSERT(located);
                        if(!located)
                            return;

                        uint32_t const destinationParticleIndex = alpaka::atomicAdd(
                            worker.getAcc(),
                            &destinationCursors[location.slot],
                            1u,
                            ::alpaka::hierarchy::Blocks{});
                        uint32_t const firstFrame
                            = location.slot == 0u ? 0u : destinationFrameScan[location.slot - 1u];
                        pmacc::spearhed::memory::FramePointer<T_Frame> const destinationFrame{
                            destinationFramePtrs[firstFrame + destinationParticleIndex / T_Frame::frameSize]};
                        auto destinationParticle = destinationFrame[destinationParticleIndex % T_Frame::frameSize];

                        // Copy every field (including the global ID) before changing only the
                        // coordinate representation and the already-known live marker.
                        sourceParticle.deepCopyTo(destinationParticle);
                        pmacc::spearhed::for_each_tag<CS>(
                            [&](auto tag)
                            { destinationParticle[tags::relativePos][tag] = location.localPosition[tag]; });
                        destinationParticle[tags::multiMask] = 1u;
                    });
            }
        };

        /** Release frame heap allocations which were detached after their scatter reads completed. */
        template<typename T_Frame>
        struct ReleaseDetachedFrames
        {
            DINLINE void operator()(auto const& worker, auto regions, auto sourceFramePtrs, uint32_t totalSourceFrames)
                const
            {
                uint32_t const sourceFrameIndex = worker.blockDomIdx();
                if(sourceFrameIndex >= totalSourceFrames)
                    return;

                auto onlyMaster = pmacc::lockstep::makeMaster(worker);
                onlyMaster(
                    [&]
                    {
                        // All frame lists in one buffer share the same allocator handle. The list
                        // no longer owns this frame, but it owns the allocator needed to free it.
                        auto* frame = sourceFramePtrs[sourceFrameIndex];
                        regions[0].particleFrameList.destroyDetachedFrame(worker, frame);
                    });
            }
        };

        /** Clear partially allocated replacement chains after an allocation failure. */
        struct ClearReplacementFrames
        {
            DINLINE void operator()(auto const& worker, auto regions, uint32_t regionCount) const
            {
                uint32_t const regionSlot = worker.blockDomIdx();
                if(regionSlot >= regionCount)
                    return;
                pmacc::lockstep::makeMaster(worker)(
                    [&] { regions[regionSlot].particleFrameList.destroyAllFrames(worker); });
            }
        };
    } // namespace detail

    /**
     * @brief Bulk single-accelerator relocation for a fixed dense Cartesian grid.
     *
     * The component treats every live particle as a source, counts destination
     * buckets, builds fresh contiguous frame chains, and scatters whole records.
     * It intentionally rebuilds every local bucket rather than incrementally
     * filling holes.  That keeps the PIConGPU-style invariant explicit: all
     * frames but the final frame are full and live slots form a prefix.  It also
     * gives coalesced destination writes and avoids a device-heap allocation per
     * crossing particle.
     *
     * A non-periodic out-of-domain particle is rejected before any list is
     * changed. Periodic coordinates are canonicalised into the grid domain.
     */
    template<CoordinateSystem CS>
    class FixedCartesianRelocator
    {
    public:
        explicit FixedCartesianRelocator(FixedCartesianGrid<CS> grid) : m_grid(grid)
        {
        }

        template<typename T_Region>
        void relocate(ParticleRegionBuffer<T_Region>& store) const
        {
            using Frame = typename T_Region::FrameType;
            constexpr uint32_t threadsPerBlock = Frame::frameSize;

            if(store.size == 0)
                return;
            if(static_cast<uint32_t>(store.size) != m_grid.bucketCount())
                throw std::invalid_argument("fixed Cartesian relocation requires one dense bucket per grid cell");

            FrameIndexBuffer<T_Region> sourceIndex{store};
            if(sourceIndex.totalFrames == 0u)
                return;

            pmacc::HostDeviceBuffer<uint32_t, DIM1> destinationCounts{
                pmacc::DataSpace<DIM1>{static_cast<int>(m_grid.bucketCount())}};
            destinationCounts.getHostBuffer().setValue(0u);
            destinationCounts.hostToDevice();
            pmacc::HostDeviceBuffer<uint32_t, DIM1> invalidDestination{pmacc::DataSpace<DIM1>{1}};
            invalidDestination.getHostBuffer().setValue(0u);
            invalidDestination.hostToDevice();

            using SourceChart = detail::FixedGridSourceChart<CS>;
            auto countDone = PMACC_LOCKSTEP_KERNEL(detail::CountFixedCartesianDestinations<Frame, CS, SourceChart>{})
                                 .template config<threadsPerBlock>(pmacc::DataSpace<DIM1>(sourceIndex.totalFrames))(
                                     sourceIndex.framePtrsBox(),
                                     sourceIndex.regionIdxBox(),
                                     sourceIndex.totalFrames,
                                     SourceChart{},
                                     m_grid,
                                     destinationCounts.getDeviceBuffer().getDataBox(),
                                     invalidDestination.getDeviceBuffer().getDataBox());
            countDone.waitForFinished();
            invalidDestination.deviceToHost();
            if(invalidDestination.getHostBuffer().getDataBox()[0] != 0u)
                throw std::out_of_range("particle left a non-periodic fixed Cartesian domain");

            pmacc::HostDeviceBuffer<uint32_t, DIM1> allocationFailed{pmacc::DataSpace<DIM1>{1}};
            allocationFailed.getHostBuffer().setValue(0u);
            allocationFailed.hostToDevice();
            auto allocateDone = PMACC_LOCKSTEP_KERNEL(detail::RebuildPackedFrameLists<Frame>{})
                                    .template config<1u>(pmacc::DataSpace<DIM1>(store.size))(
                                        store.getDeviceDataBox(),
                                        static_cast<uint32_t>(store.size),
                                        destinationCounts.getDeviceBuffer().getDataBox(),
                                        allocationFailed.getDeviceBuffer().getDataBox());
            allocateDone.waitForFinished();
            ++store.topologyVersion;

            allocationFailed.deviceToHost();
            if(allocationFailed.getHostBuffer().getDataBox()[0] != 0u)
            {
                auto clearDone = PMACC_LOCKSTEP_KERNEL(detail::ClearReplacementFrames{})
                                     .template config<1u>(pmacc::DataSpace<DIM1>(
                                         store.size))(store.getDeviceDataBox(), static_cast<uint32_t>(store.size));
                auto freeDone = PMACC_LOCKSTEP_KERNEL(detail::ReleaseDetachedFrames<Frame>{})
                                    .template config<1u>(pmacc::DataSpace<DIM1>(sourceIndex.totalFrames))(
                                        store.getDeviceDataBox(),
                                        sourceIndex.framePtrsBox(),
                                        sourceIndex.totalFrames);
                (clearDone + freeDone).waitForFinished();
                throw std::runtime_error("fixed Cartesian relocation could not allocate replacement frames");
            }

            FrameIndexBuffer<T_Region> destinationIndex{store};
            pmacc::HostDeviceBuffer<uint32_t, DIM1> destinationCursors{
                pmacc::DataSpace<DIM1>{static_cast<int>(m_grid.bucketCount())}};
            destinationCursors.getHostBuffer().setValue(0u);
            destinationCursors.hostToDevice();

            auto scatterDone = PMACC_LOCKSTEP_KERNEL(detail::ScatterFixedCartesianParticles<Frame, CS, SourceChart>{})
                                   .template config<threadsPerBlock>(pmacc::DataSpace<DIM1>(sourceIndex.totalFrames))(
                                       sourceIndex.framePtrsBox(),
                                       sourceIndex.regionIdxBox(),
                                       sourceIndex.totalFrames,
                                       SourceChart{},
                                       m_grid,
                                       destinationIndex.framePtrsBox(),
                                       destinationIndex.scanBox(),
                                       destinationCursors.getDeviceBuffer().getDataBox());
            scatterDone.waitForFinished();

            destinationCursors.deviceToHost();
            destinationCounts.deviceToHost();
            auto const cursors = destinationCursors.getHostBuffer().getDataBox();
            auto const counts = destinationCounts.getHostBuffer().getDataBox();
            for(uint32_t slot = 0u; slot < m_grid.bucketCount(); ++slot)
                if(cursors[slot] != counts[slot])
                    throw std::runtime_error("fixed Cartesian relocation lost or duplicated a particle");

            auto freeDone = PMACC_LOCKSTEP_KERNEL(detail::ReleaseDetachedFrames<Frame>{})
                                .template config<1u>(pmacc::DataSpace<DIM1>(sourceIndex.totalFrames))(
                                    store.getDeviceDataBox(),
                                    sourceIndex.framePtrsBox(),
                                    sourceIndex.totalFrames);
            freeDone.waitForFinished();
        }

    private:
        FixedCartesianGrid<CS> m_grid;
    };

    /**
     * @brief Convert setup-created material buckets into a dense fixed grid.
     *
     * Setup placement remains free to use its legacy material bucket geometry.
     * This component snapshots those source charts, classifies every resulting
     * world-space position, replaces the store's region array by stable grid
     * slots, and scatters full particle records into packed destination frames.
     * The source frame index keeps the mallocMC frame addresses alive while the
     * old region buffer is replaced.
     */
    template<CoordinateSystem CS>
    class FixedCartesianInitialClassifier
    {
    public:
        explicit FixedCartesianInitialClassifier(FixedCartesianGrid<CS> grid) : m_grid(grid)
        {
        }

        template<typename T_Region, typename T_AllocatorHandle>
        void classify(ParticleRegionBuffer<T_Region>& store, T_AllocatorHandle const& allocatorHandle) const
        {
            using Frame = typename T_Region::FrameType;
            using Chart = RegionChart<CS>;
            constexpr uint32_t threadsPerBlock = Frame::frameSize;

            FrameIndexBuffer<T_Region> sourceIndex{store};
            uint32_t const oldRegionCount = static_cast<uint32_t>(store.size);
            pmacc::HostDeviceBuffer<Chart, DIM1> sourceCharts{
                pmacc::DataSpace<DIM1>{static_cast<int>(oldRegionCount)}};
            if(oldRegionCount > 0u)
            {
                store.buffer->deviceToHost();
                auto const oldRegions = store.buffer->getHostBuffer().getDataBox();
                auto charts = sourceCharts.getHostBuffer().getDataBox();
                for(uint32_t slot = 0u; slot < oldRegionCount; ++slot)
                    charts[slot] = oldRegions[static_cast<int>(slot)].spatial.chart;
                sourceCharts.hostToDevice();
            }
            auto sourceChart = detail::MaterialSourceChart{sourceCharts.getDeviceBuffer().getDataBox()};

            pmacc::HostDeviceBuffer<uint32_t, DIM1> destinationCounts{
                pmacc::DataSpace<DIM1>{static_cast<int>(m_grid.bucketCount())}};
            destinationCounts.getHostBuffer().setValue(0u);
            destinationCounts.hostToDevice();
            pmacc::HostDeviceBuffer<uint32_t, DIM1> invalidDestination{pmacc::DataSpace<DIM1>{1}};
            invalidDestination.getHostBuffer().setValue(0u);
            invalidDestination.hostToDevice();

            if(sourceIndex.totalFrames > 0u)
            {
                auto countDone
                    = PMACC_LOCKSTEP_KERNEL(
                          detail::CountFixedCartesianDestinations<Frame, CS, decltype(sourceChart)>{})
                          .template config<threadsPerBlock>(pmacc::DataSpace<DIM1>(sourceIndex.totalFrames))(
                              sourceIndex.framePtrsBox(),
                              sourceIndex.regionIdxBox(),
                              sourceIndex.totalFrames,
                              sourceChart,
                              m_grid,
                              destinationCounts.getDeviceBuffer().getDataBox(),
                              invalidDestination.getDeviceBuffer().getDataBox());
                countDone.waitForFinished();
                invalidDestination.deviceToHost();
                if(invalidDestination.getHostBuffer().getDataBox()[0] != 0u)
                    throw std::out_of_range("initial particle lies outside a non-periodic fixed Cartesian domain");
            }

            // No old region metadata remains in the final layout. The fixed-grid code derives
            // ownership bounds and chart origins from m_grid and the dense slot; the metadata chart
            // is retained only for generic output/particle views during the Phase-5 transition.
            store.create(m_grid.bucketCount());
            for(uint32_t slot = 0u; slot < m_grid.bucketCount(); ++slot)
            {
                typename T_Region::VolumeType spatial{};
                spatial.chart = m_grid.chart(slot);
                store.pushBack(T_Region{allocatorHandle, spatial});
            }
            store.buffer->hostToDevice();

            pmacc::HostDeviceBuffer<uint32_t, DIM1> allocationFailed{pmacc::DataSpace<DIM1>{1}};
            allocationFailed.getHostBuffer().setValue(0u);
            allocationFailed.hostToDevice();
            auto allocateDone = PMACC_LOCKSTEP_KERNEL(detail::RebuildPackedFrameLists<Frame>{})
                                    .template config<1u>(pmacc::DataSpace<DIM1>(store.size))(
                                        store.getDeviceDataBox(),
                                        static_cast<uint32_t>(store.size),
                                        destinationCounts.getDeviceBuffer().getDataBox(),
                                        allocationFailed.getDeviceBuffer().getDataBox());
            allocateDone.waitForFinished();
            ++store.topologyVersion;

            allocationFailed.deviceToHost();
            if(allocationFailed.getHostBuffer().getDataBox()[0] != 0u)
            {
                auto clearDone = PMACC_LOCKSTEP_KERNEL(detail::ClearReplacementFrames{})
                                     .template config<1u>(pmacc::DataSpace<DIM1>(
                                         store.size))(store.getDeviceDataBox(), static_cast<uint32_t>(store.size));
                if(sourceIndex.totalFrames > 0u)
                {
                    auto freeDone = PMACC_LOCKSTEP_KERNEL(detail::ReleaseDetachedFrames<Frame>{})
                                        .template config<1u>(pmacc::DataSpace<DIM1>(sourceIndex.totalFrames))(
                                            store.getDeviceDataBox(),
                                            sourceIndex.framePtrsBox(),
                                            sourceIndex.totalFrames);
                    (clearDone + freeDone).waitForFinished();
                }
                else
                    clearDone.waitForFinished();
                throw std::runtime_error(
                    "fixed Cartesian initial classification could not allocate replacement frames");
            }

            if(sourceIndex.totalFrames == 0u)
                return;

            FrameIndexBuffer<T_Region> destinationIndex{store};
            pmacc::HostDeviceBuffer<uint32_t, DIM1> destinationCursors{
                pmacc::DataSpace<DIM1>{static_cast<int>(m_grid.bucketCount())}};
            destinationCursors.getHostBuffer().setValue(0u);
            destinationCursors.hostToDevice();
            auto scatterDone
                = PMACC_LOCKSTEP_KERNEL(detail::ScatterFixedCartesianParticles<Frame, CS, decltype(sourceChart)>{})
                      .template config<threadsPerBlock>(pmacc::DataSpace<DIM1>(sourceIndex.totalFrames))(
                          sourceIndex.framePtrsBox(),
                          sourceIndex.regionIdxBox(),
                          sourceIndex.totalFrames,
                          sourceChart,
                          m_grid,
                          destinationIndex.framePtrsBox(),
                          destinationIndex.scanBox(),
                          destinationCursors.getDeviceBuffer().getDataBox());
            scatterDone.waitForFinished();

            destinationCursors.deviceToHost();
            destinationCounts.deviceToHost();
            auto const cursors = destinationCursors.getHostBuffer().getDataBox();
            auto const counts = destinationCounts.getHostBuffer().getDataBox();
            for(uint32_t slot = 0u; slot < m_grid.bucketCount(); ++slot)
                if(cursors[slot] != counts[slot])
                    throw std::runtime_error("fixed Cartesian initial classification lost or duplicated a particle");

            auto freeDone = PMACC_LOCKSTEP_KERNEL(detail::ReleaseDetachedFrames<Frame>{})
                                .template config<1u>(pmacc::DataSpace<DIM1>(sourceIndex.totalFrames))(
                                    store.getDeviceDataBox(),
                                    sourceIndex.framePtrsBox(),
                                    sourceIndex.totalFrames);
            freeDone.waitForFinished();
        }

    private:
        FixedCartesianGrid<CS> m_grid;
    };
} // namespace pmacc::spearhed
