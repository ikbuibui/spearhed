# Migration Plan: Domain- and Count-Driven SC Initialization

## Goal

Make the physical domain, fluid states, and approximate fluid particle count the only inputs needed to initialize the Sod SC lattice.

The setup must no longer configure:

- left or right particle spacing;
- a hard-coded spacing ratio;
- a transverse extent separate from the fluid-domain AABB.

SC planning will derive per-region integer lattice shapes from the domain aspect ratio and target particle count. The resulting shapes, rather than a scalar spacing, will be the single source of truth for both particle allocation and placement.

## Relationship to the boundary-condition migration

General boundary-condition semantics, lifecycle hooks, and realization selection are broader than SC initialization and are specified separately in `BOUNDARY_PLAN.md`.

The execution order is:

1. Complete Phases 1-5 of this plan to establish `SCShape`, the equal-mass fluid plan, and domain-driven Sod fluid initialization.
2. Complete the boundary plan through its descriptor, manager, and legacy-particle-realization milestones.
3. Complete Phase 6 of this plan together with the aligned-ghost milestone in `BOUNDARY_PLAN.md`.
4. Complete the tests, validation, and documentation in both plans.
5. Add periodic, mirrored-wall, inflow, and outflow realizations as follow-up boundary work without changing the SC planner.

This ordering avoids first hard-coding aligned walls into Sod setup blocks and then immediately moving them behind a general boundary abstraction.

## Non-goals

- Do not change the Sod equation of state or analytical reference solution.
- Do not change the SPH force, density, or integration algorithms.
- Do not introduce random, glass, BCC, FCC, or other placement methods as part of this migration.
- Do not make the requested particle count exact by partially filling a lattice plane.
- Do not tune the physical validation result in the same changes that introduce the new planning API.

## Target configuration model

The Sod setup should expose only the following independent inputs:

1. The fluid-domain AABB, including all active-axis extents.
2. The x coordinate of the left/right discontinuity.
3. The left and right density and pressure.
4. An approximate target number of fluid particles.
5. The kernel and boundary policy.

All other quantities are derived:

- left and right fluid AABBs;
- weighted regional target counts;
- integer per-axis SC cell counts;
- actual regional and total particle counts;
- common particle mass;
- effective per-axis particle spacings;
- wall layer counts and outer boundary extents.

The setup dependency chain should be:

```text
fluid domain + split position + fluid states + approximate N
    -> left/right region volumes
    -> equal-mass SC plan
    -> integer lattice shapes and actual N
    -> particle mass and diagnostic effective spacings
    -> aligned boundary-lattice extension
    -> full simulation domain
```

## Core invariants

The implementation must preserve these invariants:

1. A region creates exactly `product(shape.cells)` particles.
2. Placement receives the same complete shape used to calculate the count.
3. Every particle index in `[0, product(shape.cells))` maps to one unique lattice site.
4. No scalar `numCells` placement path may wrap extra particles onto existing sites.
5. The requested total count is approximate; the actual count is always reported and used.
6. The left and right fluid use one common particle mass.
7. Integer lattice shapes follow each region's AABB aspect ratio as closely as possible.
8. Effective spacing is an output of the plan, not a setup parameter.
9. Boundary particles continue the adjacent fluid lattice instead of independently refitting a wall AABB.
10. Products and floating-to-integer conversions are checked before narrowing to `uint32_t`.

## Planning mathematics

For fluid regions `r`, define:

```text
physicalMass_r = rho_r * volume_r
physicalMass   = sum(physicalMass_r)
```

Given approximate total count `N_target`, the continuous equal-mass target for each region is:

```text
N_target_r = N_target * physicalMass_r / physicalMass
```

For a region with axis lengths `L_i`, volume `V`, dimension `D`, and continuous target `N`, the ideal isotropic SC counts are:

```text
idealCells_i = L_i * pow(N / V, 1 / D)
```

These satisfy, before integer rounding:

```text
product(idealCells_i) = N
idealCells_i / idealCells_j = L_i / L_j
```

The planner selects nearby positive integer counts and defines:

```text
N_actual_r = product(cells_r)
N_actual   = sum(N_actual_r)
particleMass = physicalMass / N_actual
rho_discrete_r = particleMass * N_actual_r / volume_r
effectiveSpacing_r_i = L_r_i / cells_r_i
```

The planner must evaluate the discrete density error and spacing anisotropy instead of assuming the rounded shape is exact.

## Shape selection policy

Implement shape selection as a small host-side integer optimization. The simulation dimensions are at most three and Sod has only two fluid regions, so exhaustive evaluation of a small candidate set is preferable to fragile floating-point truncation rules.

For each axis:

1. Compute `idealCells_i` in `double`.
2. Generate positive candidates around it, initially `floor`, `ceil`, and one neighboring integer on each side where valid.
3. Enumerate candidate count vectors for each fluid region.
4. Evaluate left/right candidate pairs together.

Use the following deterministic, lexicographic score:

1. Minimize the maximum relative discrete-density error across fluid regions.
2. Minimize the maximum spacing anisotropy within a region, measured from `L_i / cells_i`.
3. Minimize relative total-count error from `N_target`.
4. Use a stable lexicographic count-array tie break for reproducibility.

This ordering prioritizes the equal-mass density model and SC isotropy over hitting an exact particle count. Expand the candidate window only if the initial neighborhood cannot satisfy an agreed density-error tolerance.

Initial acceptance tolerance:

```text
max relative interior density error <= 1 percent
```

The existing 2D and 3D Sod geometries should admit effectively exact density ratios. The 1D integer allocation should remain much more accurate than this threshold.

## Phase 1: Characterize the current behavior

Before changing implementation code:

1. Add host-side checks or a small diagnostic test that records current fluid counts, masses, and effective spacings for all three Sod configurations.
2. Preserve the current resolution as the migration baseline:

   | Dimension | Initial target fluid count | Expected current fluid count |
   |---|---:|---:|
   | 1D | 32000 | approximately 32000 |
   | 2D | 3150 | 2800 left + 350 right |
   | 3D | 3375 | 3000 left + 375 right |

3. Treat using 32000 particles in every dimension as a separate resolution change. Do not accidentally increase the 2D/3D validation cost during the API migration.
4. Record initial openPMD particle counts and Sod L1 results for comparison after migration.

Deliverable: characterization tests and a documented baseline with no behavior change.

## Phase 2: Introduce a safe SC shape API

Modify `include/spmacc/particles/initialization/SC.hpp`.

### 2.1 Add a shape type

Introduce a trivially copyable, device-friendly descriptor:

```cpp
template<CoordinateSystem CS>
struct SCShape
{
    std::array<uint32_t, CS::dimension> cells;
};
```

Provide helpers for:

- checked 64-bit site count;
- checked conversion to the `uint32_t` count expected by the current frame list;
- effective per-axis spacing for host diagnostics;
- validation that every count is nonzero.

### 2.2 Separate planning from placement

Planning from approximate counts and AABBs must be host-side. Placement must only consume integer counts and perform integer mixed-radix decomposition plus coordinate writes.

Keep this device path small:

```cpp
SC<CS>{}(worker, particle, particleRegion, globalParticleIdx, shape);
```

The placement code must assert or otherwise rely on the proven precondition:

```text
globalParticleIdx < numSites(shape)
```

### 2.3 Remove the unsafe scalar interface

Remove:

- `computeSCNumCells(uint32_t, AABB)` returning only x cells;
- `SC::operator(..., uint32_t numCells)`;
- any setup or test that allocates an arbitrary count but places into a smaller derived lattice.

If a compatibility transition is needed, mark the scalar path deprecated for one commit and make it fail at compile time for multidimensional use. Do not retain a runtime path known to alias positions.

### 2.4 Replace the spacing helper

Remove the asymmetric x-truncate/y-z-round implementation from the public SC API. If a spacing-to-shape helper remains useful for other callers, give it an explicit policy name such as:

```cpp
makeSCShapeForTargetSpacing(...)
makeSCShapeForMaximumSpacing(...)
```

It must round every axis according to the same documented policy. Sod must not use this helper after migration.

Deliverable: SC placement has one complete-shape interface and cannot silently alias particles.

## Phase 3: Add an equal-mass fluid lattice planner

Add a host-side planner, preferably under:

```text
include/spmacc/particles/initialization/SCPlan.hpp
```

Keep generic SC geometry planning in `spmacc`; keep Sod-specific state and two-region policy in the validation setup or a spearhed helper.

Suggested generic records:

```cpp
template<CoordinateSystem CS>
struct SCRegionInput
{
    AABB<CS> volume;
    double density;
};

template<CoordinateSystem CS>
struct SCRegionPlan
{
    AABB<CS> volume;
    SCShape<CS> shape;
    uint64_t numParticles;
};

template<CoordinateSystem CS, std::size_t NumRegions>
struct EqualMassSCPlan
{
    std::array<SCRegionPlan<CS>, NumRegions> regions;
    uint64_t actualParticles;
    typename CS::T_Axis particleMass;
};
```

Requirements:

- Planning calculations use `double` even when particle coordinates use `float`.
- The returned plan is trivially copyable or can be converted into a small trivially copyable kernel argument.
- Invalid AABBs, non-positive densities, zero target count, and overflow produce clear host-side failures.
- The planner reports target count, actual count, cell arrays, effective spacings, particle mass, and discrete density errors.
- Planning does not allocate device memory.

Deliverable: a tested planner that derives complete integer shapes from regions, densities, and approximate total count.

## Phase 4: Make the fluid domain the geometry source of truth

Refactor `share/spearhed/ValidationSetups/SodShockTube.hpp`.

### 4.1 Define one primary fluid-domain AABB

Replace `transverseExtent()` and the duplicated `leftFluidVol`, `rightFluidVol`, and `fluidDomain` literals with:

1. One dimension-specific default fluid-domain AABB containing all axis extents.
2. One `splitAABB` helper that splits that domain at the configured x discontinuity.

The complete AABB still contains the numerical transverse-domain choice, but there is no separate scalar transverse extent that can disagree with it.

### 4.2 Derive all fluid geometry

From the primary fluid domain:

- derive left and right fluid AABBs by changing only their x bounds;
- validate that the split lies strictly inside the domain;
- preserve origin/min/max semantics;
- derive regional volumes and aspect ratios directly from those AABBs.

### 4.3 Define one resolution input

Replace `spacingLeft`, `spacingRatio`, `spacingRight`, and the unused common `defaultTotalParticles` with a dimension-specific approximate target fluid count.

Use defaults that preserve the current computational resolution during migration:

```text
1D: 32000
2D: 3150
3D: 3375
```

The value should be named `targetFluidParticles` to make its approximate nature explicit.

Deliverable: domain AABB plus target count are the only geometric/resolution inputs to fluid planning.

## Phase 5: Migrate Sod fluid counting and placement

Replace `SpacingNumParticles` and `SpacingPlaceParticle`.

### 5.1 Construct the plan once on the host

During `SodShockTube` construction:

1. Split the fluid domain.
2. Build the two-region equal-mass SC plan.
3. Construct the interior block from that plan.
4. Preserve the legacy boundary block temporarily until the boundary-manager foundation from `BOUNDARY_PLAN.md` is available.
5. Derive the final full simulation domain from the boundary realization rather than making the Sod setup own wall geometry.

Avoid the current pattern where top-level initial conditions are copied into blocks by default member initializers and can later diverge. Prefer an explicit default constructor that establishes all derived state in initialization order.

### 5.2 Use plan shapes in both phases

The count functor selects the left or right plan entry and returns its checked site count.

The placement functor selects the same entry and passes its `SCShape` to `SC<CS>`. It initializes:

- common planned particle mass;
- state density and pressure-derived internal energy;
- zero velocity;
- smoothing length.

Do not recompute cell counts from floating-point spacing in either kernel.

For the initial two-region Sod migration, selecting a plan entry from the region's x bounds is acceptable. If this pattern is generalized, extend the initialization interface to provide a stable block-local region index to both count and placement functors rather than repeatedly identifying regions geometrically.

### 5.3 Add startup diagnostics

Print once on rank zero:

- target and actual fluid particle count;
- left/right shapes and counts;
- common particle mass;
- effective spacing per axis;
- discrete density error per state.

Deliverable: fluid initialization contains no explicit spacing configuration and has a visible, reproducible plan.

## Phase 6: Integrate the SC plan with boundary realizations

This phase depends on the boundary descriptors, `BoundaryManager`, and legacy particle realization defined in `BOUNDARY_PLAN.md`. Sod setup code must describe boundary semantics; it must not regain ownership of wall species, wall AABBs, particle counts, or placement functors.

### 6.1 Add the aligned particle realization

Implement `AlignedGhostParticles` as a boundary realization that consumes the adjacent `EqualMassSCPlan`. Do not run the fluid target-count planner independently on arbitrary wall AABBs, because that would stretch wall lattices and recreate the current wall-density mismatch.

For each adjacent state and axis:

```text
effectiveSpacing_i = fluidExtent_i / fluidCells_i
requiredLayers_i = ceil(kernelSupportRadius / effectiveSpacing_i)
wallWidth_i = requiredLayers_i * effectiveSpacing_i
```

Use effective per-axis fluid spacing, not nominal scalar spacing.

### 6.2 Generate implementation-owned ghost regions

The realization, rather than `SodShockTube.hpp`, constructs non-overlapping ghost regions whose boundaries lie on cell boundaries of the adjacent fluid lattice. It must define deterministic ownership of faces, edges, and corners so no ghost particle is duplicated.

The existing face decomposition may be reused internally if every piece is snapped to the corresponding fluid grid. Otherwise use an integer-index shell generator.

Each ghost particle must:

- use the common fluid particle mass;
- receive state from the boundary model's state provider;
- lie on a lattice continuation with no phase jump at the fluid boundary;
- be exposed as an interaction source only through the boundary realization.

The setup's full diagnostic domain is the envelope reported by the compiled boundary configuration. The fluid-domain AABB remains the primary geometry input.

### 6.3 Preserve semantics during migration

First map the existing frozen-particle behavior to an explicitly named legacy or fixed-exterior model. Aligned placement fixes geometric support but does not by itself turn frozen particles into a reflecting, periodic, or outflow boundary.

Periodic transverse boundaries and dynamic mirrored walls follow the later milestones in `BOUNDARY_PLAN.md`. They should not be mixed into the initial SC behavior-preservation change.

Deliverable: Sod describes physical boundary models, while an aligned particle realization derives ghost geometry and particles from the fluid SC plan.

## Phase 7: Tests

### 7.1 Generic SC tests

Update `tests/spearhed/unit/ParticleInit/ParticleInit.cpp` and add focused host tests where appropriate.

Required cases:

- A target of 10 particles in a unit cube cannot create duplicate positions.
- The created particle count equals `numSites(shape)`.
- Every lattice position is unique and lies inside its AABB.
- Non-cubic AABBs produce count ratios consistent with permuted axis extents.
- Permuting AABB axes permutes the chosen count array rather than changing policy by axis name.
- Zero counts, invalid extents, invalid density, and overflow are rejected.
- One-dimensional planning behaves without special scalar placement code.
- Mixed-radix placement reaches the first and last expected lattice cells.

Do not infer uniqueness only from count and bounds. Copy positions to the host or use a host-side placement equivalent, sort them, and explicitly detect duplicates.

### 7.2 Planner tests

For synthetic one-, two-, and three-dimensional domains, verify:

- regional target counts are weighted by `rho * volume`;
- actual count is near the target;
- the common mass is `sum(rho * volume) / actualCount`;
- effective spacing follows the AABB aspect ratio;
- the density ratio is derived from the input densities, not hard-coded as 8;
- modifying domain extents changes the shape without changing setup spacing constants, because no such constants remain;
- modifying densities changes regional counts while preserving equal particle mass.

### 7.3 Sod regression tests

For each configured dimension, verify expected baseline shapes or equivalent invariants:

- 1D actual count remains approximately 32000;
- 2D remains approximately 2800 left and 350 right;
- 3D remains 3000 left and 375 right;
- 2D/3D left-to-right count ratio remains approximately 8 for equal region volumes;
- initialized pressure and internal energy match the Sod states;
- no fluid or wall positions overlap;
- wall layers cover at least one kernel support radius;
- boundary geometric mass density matches the adjacent fluid within tolerance.

### 7.4 Device compilation

Compile the SC placement tests with the CPU serial backend and CUDA. The planner itself should remain host-side, while `SCShape` and placement remain CUDA-copyable/device-callable.

## Phase 8: Validation and documentation

Update `share/spearhed/ValidationSetups/README.md`:

- replace `dxLeft` and `transverseExtent` tuning instructions;
- document `targetFluidParticles` as the resolution knob;
- document the fluid-domain AABB as the sole geometry source;
- explain that actual count and effective spacing are derived and printed;
- update expected particle-count tables from measured post-migration values;
- describe the equal-mass derivation and integer rounding tolerance.

Run validation in widening circles:

1. Build and run generic ParticleInit tests on CPU serial.
2. Compile the ParticleInit target with CUDA; do not run it without a GPU.
3. Build 1D, 2D, and 3D Sod configurations.
4. Dump step zero and verify counts, masses, positions, uniqueness, and state attributes.
5. Run the existing 1D validation to step 200 and compare L1 errors with the recorded baseline.
6. Run representative 2D and 3D validations and compare projected L1 errors and transverse-velocity diagnostics.

Use the documented build constraints:

```bash
cmake --build <build-dir> -j 4
```

and use the Pixi laptop environment for openPMD validation.

## Suggested commit sequence

1. `test: characterize current SC and Sod particle layouts`
2. `refactor: introduce checked SCShape placement API`
3. `test: cover arbitrary target counts and position uniqueness`
4. `feat: add host equal-mass SC planner`
5. `refactor: derive Sod fluid layout from domain and target count`
6. `feat: add boundary descriptors, manager, and legacy particle realization`
7. `refactor: realize aligned Sod ghosts from the fluid SC plan`
8. `docs: describe domain- and count-driven Sod initialization`
9. `test: record post-migration Sod validation baselines`

Each commit should leave CPU builds usable. CUDA changes are compile-only validated.

## Completion criteria

The migration is complete when:

- `spacingLeft`, `spacingRight`, `spacingRatio`, and `transverseExtent` no longer exist in the Sod setup;
- one fluid-domain AABB is the only source of all fluid extents;
- left/right fluid AABBs are derived by splitting that domain;
- the requested particle count is explicitly approximate;
- one host-generated plan owns all integer SC shapes;
- allocation and placement consume the same shapes;
- the unsafe scalar SC placement overload is gone;
- all created particle positions are unique;
- a common particle mass is derived from physical mass and actual count;
- wall particles align with and match the mass density of adjacent fluid lattices;
- 1D/2D/3D CPU builds and CUDA compilation pass;
- Sod validation remains within agreed regression tolerances;
- the README documents only domain geometry and target particle count as resolution inputs.
