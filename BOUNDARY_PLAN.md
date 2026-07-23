# Boundary-Condition Architecture Plan

## Purpose and relationship to `PLAN.md`

This plan introduces a general boundary-condition architecture in which setups describe physical boundary semantics and the simulation selects an implementation realization. Particle boundaries are one realization of that general concept.

This belongs in a separate file because it affects setup interfaces, simulation lifecycle, neighbour construction, particle transport, MPI topology, and multiple future boundary models. It is broader than the domain- and count-driven SC migration in `PLAN.md`.

The plans meet at aligned SC ghost generation. Use this execution order:

1. Complete Phases 1-5 of `PLAN.md` to obtain checked `SCShape` objects and an `EqualMassSCPlan` for the fluid.
2. Complete Milestones 1-3 below to introduce boundary descriptors, `BoundaryManager`, and a behavior-preserving legacy particle realization.
3. Implement Milestone 4 below together with Phase 6 of `PLAN.md`; this moves aligned ghost generation behind the general boundary abstraction.
4. Complete the regression tests and documentation for both plans.
5. Implement periodic, mirrored-wall, inflow, and outflow models in later milestones.

The architecture may be developed before the SC planner, but aligned ghost generation must not be finalized until the fluid plan provides exact integer lattice shapes.

## Goals

- Let setups describe what a boundary means physically rather than how to allocate particles for it.
- Separate boundary geometry, physical model, and implementation realization.
- Support particle, virtual-image, topological, and particle-lifecycle realizations under one setup-facing vocabulary.
- Keep dispatch compile-time and avoid virtual calls in device code.
- Make boundary actions explicit at the correct stages of the simulation loop.
- Keep species roles as operational capabilities rather than boundary semantics.
- Make global physical boundaries distinct from rank-local MPI decomposition boundaries.
- Permit setup defaults for periodicity without maintaining conflicting configuration in multiple places.

## Non-goals

- Do not implement every boundary model in the first migration.
- Do not claim that the current frozen particle layer is a mathematically reflecting wall.
- Do not force all boundary models to allocate a particle species.
- Do not add `Periodic`, `Reflecting`, `Outflow`, or similar semantic tags to the species role system.
- Do not add runtime polymorphism to particle interaction hot paths.
- Do not generalize immediately to arbitrary curved immersed geometry; design an extension point and implement box-domain faces first.
- Do not change Sod boundary physics in the same commit that introduces the abstraction.

## Core distinction

A boundary condition has four independent parts:

1. **Geometry**: where it applies.
2. **Physical model**: what mathematical condition it imposes.
3. **Realization**: how the code enforces or approximates that model.
4. **Operational resources**: particle buffers, transformed neighbour images, injection queues, or other data owned by the realization.

Setups normally specify only geometry and physical model. A default realization trait selects the implementation. Tests or advanced setups may explicitly override the realization.

## Target setup syntax

The desired high-level Sod description is:

```cpp
auto boundaryConditions() const
{
    using namespace boundary;

    return box(fluidDomain)
        .axis<x>(
            reservoir(initialConditions.left),
            reservoir(initialConditions.right))
        .transverseTo<x>(periodic());
}
```

A closed wall setup should be expressible as:

```cpp
auto boundaryConditions() const
{
    return boundary::box(fluidDomain)
        .allFaces(boundary::slipWall(boundary::adiabatic()));
}
```

The initial implementation may use a simpler tuple API before adding fluent syntax:

```cpp
auto boundaryConditions() const
{
    using namespace boundary;

    return makeSet(
        on(lowerFace<x>(), Reservoir{leftState}),
        on(upperFace<x>(), Reservoir{rightState}),
        periodicAxesExcept<CS, x>());
}
```

Setup code must not mention:

- a boundary species;
- `Frozen` or `Source` roles;
- wall or ghost AABBs;
- ghost particle counts;
- ghost placement functors;
- boundary interaction-buffer registration;
- boundary lifecycle hooks.

## Boundary descriptor model

Use statically typed descriptors:

```cpp
template<typename Geometry, typename Model, typename Realization = Automatic>
struct BoundaryCondition
{
    Geometry geometry;
    Model model;
    Realization realization;
};
```

A `BoundarySet` is a heterogeneous tuple of conditions with validation and iteration helpers. Setup boundary types are fixed at compile time, while positions, states, and other values may be runtime data stored inside descriptors.

### Geometry types

Implement box-domain geometry first:

```cpp
template<AxisTag Axis, Side SideValue>
struct DomainFace;

template<AxisTag Axis>
struct DomainAxisPair;
```

Suggested construction vocabulary:

```cpp
lowerFace<x>()
upperFace<x>()
axisPair<y>()
allFaces<CS>()
transverseFaces<CS, x>()
```

`DomainFace` derives its physical location and extent from the global fluid-domain AABB. A setup must not repeat the face coordinate.

Future geometry extension points may include:

```cpp
Plane
Sphere
SignedDistanceSurface
BoundaryPatch<Surface, Selector>
```

Do not require those types for the first implementation.

### Physical model types

Provide semantic models rather than implementation names:

```cpp
Periodic
Reservoir<StateProvider>
SlipWall<ThermalPolicy, WallMotion>
NoSlipWall<ThermalPolicy, WallMotion>
Outflow<ExtrapolationPolicy>
Inflow<StateProvider>
Open
LegacyFrozenExterior<StateProvider>
```

`LegacyFrozenExterior` is intentionally explicit. It describes the behavior currently implemented by frozen source particles without claiming reflection, no-slip, or a rigorous reservoir transport condition.

State providers should support constant and time-dependent states:

```cpp
struct ConstantPrimitiveState
{
    HDINLINE PrimitiveState operator()(Point position, Real time) const;
};
```

A primitive state should contain only physical state required by the model, such as density, pressure, velocity, and optional thermal data. The realization converts it into particle-record fields.

### Thermal and wall-motion policies

Keep wall semantics composable:

```cpp
Adiabatic
Isothermal{temperature}
PrescribedInternalEnergy{value}
StationaryWall
PrescribedWallVelocity{provider}
```

Do not expose every composition through fluent syntax initially. Start with complete named models such as `SlipWall{Adiabatic{}, StationaryWall{}}` and add convenience factories later.

## Realization selection

Define a trait or customization point:

```cpp
template<typename Geometry, typename Model, typename SimulationCapabilities>
struct DefaultBoundaryRealization;
```

Initial intended mappings are:

| Physical model | Default realization |
|---|---|
| `LegacyFrozenExterior` | Persistent frozen particle buffer |
| `Reservoir` | Aligned persistent ghost particles, later with transport handling |
| `Periodic` | Position wrapping plus paired/transformed neighbour access |
| `SlipWall` | Mirrored virtual or materialized ghosts plus crossing reflection |
| `NoSlipWall` | Mirrored ghosts with wall-relative velocity transform |
| `Outflow` | Particle removal plus extrapolated exterior support |
| `Inflow` | Particle injection plus prescribed exterior support |
| `Open` | No exterior support, with an explicit diagnostic warning where appropriate |

Permit explicit overrides for validation and implementation comparison:

```cpp
on(lowerFace<x>(), Reservoir{state})
    .realizeWith<AlignedGhostParticles>();
```

The setup should rarely need this override.

## Boundary condition contract

Each model and realization combination must define both of these aspects where relevant:

### Particle transport

What happens when a live interior particle crosses the boundary?

Examples:

- periodic: wrap to the paired face;
- slip wall: reflect position and normal velocity;
- reservoir outflow: remove the exiting particle;
- inflow: inject particles according to a flux/state policy;
- legacy frozen exterior: no explicit crossing rule, matching current behavior.

### Exterior SPH support

What state does an interior particle interact with outside the domain?

Examples:

- periodic: transformed images from the paired side;
- slip wall: mirrored interior state;
- reservoir: prescribed state;
- outflow: extrapolated state;
- legacy frozen exterior: fixed persistent particles.

A realization is incomplete if the model requires both aspects but only one is implemented.

## Boundary manager

Introduce a compile-time `BoundaryManager<BoundarySet>` owned by `Simulation`. It compiles descriptors into resources and invokes optional lifecycle hooks using concepts or detection rather than virtual functions.

Suggested lifecycle:

```cpp
boundaryManager.initialize(context);

for(each step)
{
    boundaryManager.beforePush(context);

    ParticlePush{}(...);

    boundaryManager.afterPush(context);
    // Wrap, reflect, remove, migrate, or inject particles.

    UpdateVolumes{}(...);

    boundaryManager.prepareInteractions(context);
    // Refresh ghost state and prepare periodic/mirrored sources.

    calculateNeighbours(..., boundaryManager.sources());

    UpdateDensity{}(...);
    UpdateHydroForces{}(...);

    boundaryManager.afterInteractions(context);
}
```

Not every realization implements every hook. Compile-time detection should remove empty work.

### Contexts

Pass focused contexts rather than the entire simulation object. Possible contexts include:

```cpp
BoundaryInitializationContext
BoundaryTransportContext
BoundaryInteractionContext
```

They may provide:

- the global fluid domain;
- timestep and simulation time;
- kernel support radius;
- target particle buffers;
- device heap and data connector access;
- MPI rank topology;
- SC initialization plans where available.

Keep device-friendly arguments small and trivially copyable.

### Resource ownership

The manager or compiled realization owns:

- persistent ghost particle buffers;
- transformed image metadata;
- injection/removal queues;
- topology mappings;
- derived full-domain envelopes.

Setups own semantic configuration, not operational resources.

## Integration with species and roles

Keep roles capability-based:

- `Source`: contributes to interactions;
- `Movable`: handled by the normal particle pusher;
- `Frozen`: never changed by normal transport or integration;
- `Thermodynamic`: carries required SPH fields.

Do not encode boundary semantics as roles.

During the first migration, the existing `species::Boundary` may implement `LegacyFrozenExterior`. The realization creates and registers that species buffer and exposes it as a source.

Later options are:

- rename it to `FixedGhost` to reflect its implementation role;
- add a boundary-managed ghost species that lacks `Movable` but may be refreshed by `BoundaryManager`;
- use virtual/transformed sources that allocate no particle species.

Dynamic mirrored ghosts are not `Frozen` in the semantic sense, even if they are excluded from normal integration. Their attributes are managed explicitly by the boundary realization.

## Interaction-source abstraction

Initially, `BoundaryManager::sources()` may return ordinary particle-region buffers so existing neighbour bundles continue to work.

Design toward a broader boundary-source concept capable of representing:

- a real particle buffer;
- a particle buffer with a coordinate/velocity transform;
- a periodic image mapping;
- a generated mirrored view;
- an analytic exterior contribution.

Do not block the initial migration on this generalization. Add it when periodic or virtual mirrored sources require it.

## Setup and simulation lifetime

Currently `Setup{}` is constructed locally during `fillSimulation()`. Boundary configuration, especially periodic topology, may be needed before device/environment initialization.

Refactor toward persistent ownership:

```cpp
class Simulation
{
    Setup setup;
    BoundaryManager<BoundarySetFor<Setup>> boundaryManager;
};
```

Requirements:

- construct setup early enough to query periodic axes;
- keep setup and boundary descriptors alive for the simulation;
- derive PMacc periodic flags from the compiled boundary set;
- allow an explicit CLI override only with validation and clear logging;
- reject conflicting setup and CLI periodicity rather than silently selecting one;
- distinguish global physical faces from internal MPI partition faces.

If persistent setup ownership is too invasive for the first milestone, provide a static or default-constructible topology query as a transition, then remove the duplicate path.

## Validation of boundary configurations

Compile or initialize a boundary set only after checking:

- every global outer face has exactly one condition, including explicit `Open` where desired;
- a periodic axis is represented as one paired-axis condition, not two unrelated faces;
- no face has conflicting models;
- the model has a supported realization;
- required particle fields are present;
- the domain AABB is valid;
- wall or ghost support covers the kernel radius;
- face/axis tags belong to the coordinate system;
- implementation constraints such as Cartesian-only geometry are diagnosed clearly;
- MPI periodic topology agrees with the physical periodic model.

Prefer diagnostics near setup construction over template failures inside kernels.

## Milestone 1: Characterize current behavior

Before introducing abstractions:

1. Record current frozen-boundary particle counts and AABBs.
2. Record which simulation phases act on `species::Boundary` through roles.
3. Add a uniform-state boundary test measuring density and acceleration near walls.
4. Retain the existing frozen-position and confinement tests, but document that confinement alone does not prove reflecting-wall semantics.
5. Record Sod boundary-sensitive diagnostics, including transverse velocity and density near wall support layers.

Deliverable: a behavior baseline with no boundary architecture changes.

## Milestone 2: Add semantic descriptors

Add a boundary namespace, suggested location:

```text
include/spearhed/boundary/
```

Initial files may include:

```text
BoundaryCondition.hpp
BoundarySet.hpp
DomainFace.hpp
Models.hpp
StateProvider.hpp
Validation.hpp
```

Tasks:

1. Define geometry, model, and realization descriptor types.
2. Add `boundaryConditions()` to the setup contract, initially with a compatibility default for existing setups.
3. Implement tuple iteration and configuration validation.
4. Add the low-level tuple construction API.
5. Add the fluent box builder only after descriptor behavior is stable.
6. Convert one test setup to descriptors without changing its particle implementation.

Deliverable: setups can describe boundary semantics, but the current execution path remains intact.

## Milestone 3: Add `BoundaryManager` and legacy realization

Implement:

```text
BoundaryManager.hpp
realizations/LegacyFrozenParticles.hpp
```

Tasks:

1. Move current wall-region creation, boundary particle initialization, and source registration into `LegacyFrozenParticles`.
2. Make the manager call initialization and expose the resulting source buffer.
3. Keep current roles and interaction algorithms working.
4. Remove explicit boundary blocks from the converted setup.
5. Integrate lifecycle hooks as no-ops except where legacy behavior needs them.
6. Preserve current numerical results before changing wall geometry.

Deliverable: current particle boundaries are an implementation selected from a semantic boundary descriptor.

## Milestone 4: Add aligned SC ghost realization

Implement this milestone together with Phase 6 of `PLAN.md`.

Suggested file:

```text
include/spearhed/boundary/realizations/AlignedGhostParticles.hpp
```

Tasks:

1. Consume `EqualMassSCPlan` and the adjacent fluid-region plan.
2. Continue fluid lattice positions across each boundary face.
3. Derive integer ghost layers from kernel support and effective per-axis spacing.
4. Define deterministic ownership of face, edge, and corner ghost sites.
5. Use the common fluid particle mass.
6. Initialize fields through the model's state provider.
7. Expose the ghost buffer as an interaction source.
8. Report the derived full-domain envelope to the simulation.
9. Replace Sod's manually defined wall AABBs and boundary initialization block.

For behavior preservation, first use `LegacyFrozenExterior` with the aligned realization. A later semantic change may map x caps to `Reservoir` and transverse axes to `Periodic` or `SlipWall`.

Deliverable: aligned particle boundaries are reusable realizations and no longer Sod setup implementation code.

## Milestone 5: Implement periodic axes

Periodic boundaries should not allocate frozen wall particles.

Tasks:

1. Represent periodicity as `DomainAxisPair<Axis>`.
2. Derive PMacc periodic topology from the boundary set before environment initialization.
3. Wrap particles crossing either global face.
4. Ensure region origins/AABBs remain consistent after wrapping.
5. Provide neighbour interactions across the paired faces, using transformed region mappings or virtual image sources.
6. Support corners where multiple axes are periodic.
7. Convert Sod transverse axes to periodic only in a separate validation change.

Deliverable: a physical periodic condition described in setup and enforced in transport and neighbour search.

## Milestone 6: Implement dynamic wall models

Implement a slip wall before no-slip behavior.

### Slip wall

- Reflect crossing position across the wall.
- Reverse wall-relative normal velocity.
- Preserve wall-relative tangential velocity.
- Mirror or extrapolate density and pressure according to the model.
- Update virtual or materialized ghosts before interactions.

### No-slip wall

- Enforce wall-relative tangential velocity according to the selected discretization.
- Support stationary and prescribed moving walls.
- Keep thermal behavior separate through `Adiabatic` or `Isothermal` policy.

Do not label fixed stale particles as either model.

Deliverable: explicit rigid-wall semantics independent of whether ghosts are virtual or materialized.

## Milestone 7: Implement reservoir, inflow, and outflow transport

The initial persistent reservoir ghosts provide exterior SPH support but not complete transport behavior once waves or particles reach the domain edge.

Tasks:

- remove particles crossing an outflow/reservoir exit;
- inject particles for inflow according to flux and state policy;
- assign globally unique IDs;
- update frame-list topology versions;
- preserve mass flux and avoid injection overlap;
- synchronize lifecycle changes before neighbour indexing;
- test multi-rank ownership at global faces.

Deliverable: reservoir/inflow/outflow models define both exterior support and particle transport.

## Milestone 8: Generalize geometry if needed

Only after box-face models are stable:

- introduce planes, moving surfaces, or signed-distance geometry;
- define local normal evaluation;
- support curved mirrored states;
- provide acceleration structures where per-particle geometry checks would otherwise be expensive.

This milestone is optional until a setup requires immersed or curved boundaries.

## Testing strategy

### Descriptor and validation tests

- Every domain face is covered exactly once.
- Conflicting face conditions are rejected.
- Periodic axes must be paired.
- `transverseTo<x>` expands only axes present in `CS`.
- Unsupported model/realization combinations fail with clear diagnostics.
- Setup and CLI periodicity conflicts are rejected.

### Legacy realization tests

- Numerical behavior matches the existing frozen boundary within tolerance.
- Boundary particles remain excluded from normal pushing/integration.
- Boundary sources remain included in density and hydro interactions.
- Setups contain no explicit boundary blocks after conversion.

### Aligned ghost tests

- Ghost positions exactly continue the adjacent fluid lattice.
- No duplicate or missing face/edge/corner sites exist.
- Geometric mass density matches the adjacent fluid.
- Ghost support reaches at least one kernel support radius.
- A uniform state has uniform density near the boundary.
- Initial acceleration in a uniform state is approximately zero.

### Periodic tests

- A crossing particle wraps to the paired face with unchanged physical velocity and attributes.
- Pair interactions include neighbours across the seam.
- Mass and particle count are conserved.
- Two- and three-axis corner wrapping works.
- Single-rank and multi-rank behavior agree.

### Wall-model tests

- Slip wall reverses normal velocity and preserves tangential velocity.
- Moving slip wall uses wall-relative velocity.
- No-slip behavior enforces the documented tangential condition.
- Uniform pressure does not create spurious normal acceleration.
- Dynamic ghost state follows evolving adjacent fluid rather than remaining stale.

### Reservoir and flow tests

- Fixed reservoir support reproduces the prescribed exterior state.
- Outflow does not produce a large reflected pressure wave.
- Inflow mass flux matches the configured flux.
- IDs remain unique after injection.
- Frame indexes are invalidated and rebuilt after topology changes.

## Performance requirements

- Use a heterogeneous tuple and compile-time dispatch; no device-side virtual calls.
- Fuse multiple face transport conditions into one particle traversal when practical.
- Do not regenerate persistent ghost topology every step when only attributes change.
- Prefer transformed virtual sources for periodic and mirrored boundaries when this avoids particle copies without complicating coalesced access.
- Keep boundary-specific branches outside pairwise interaction inner loops where possible.
- Reuse frame and neighbour indexes only while boundary lifecycle hooks have not changed topology.
- Measure ghost memory and neighbour-count overhead for 2D and 3D validation setups.

## Suggested commit sequence

1. `test: characterize frozen particle boundary behavior`
2. `feat: add boundary geometry and model descriptors`
3. `feat: add boundary configuration validation`
4. `refactor: own setup and boundary configuration for simulation lifetime`
5. `feat: add BoundaryManager lifecycle`
6. `refactor: realize legacy frozen boundaries through BoundaryManager`
7. `refactor: generate aligned SC ghosts from fluid plans`
8. `test: verify aligned ghost equilibrium and density`
9. `feat: add periodic axis transport and neighbour mapping`
10. `feat: add dynamic mirrored slip walls`
11. `feat: add reservoir and outflow particle lifecycle`
12. `docs: document setup boundary vocabulary and realizations`

Commits 1-8 are part of the current SC/Sod migration sequence. Later implementations may proceed independently.

## Completion criteria for the architecture foundation

The foundation needed by `PLAN.md` is complete when:

- setups describe boundary geometry and physical models through `boundaryConditions()`;
- Sod and boundary tests no longer define explicit boundary initialization blocks;
- `BoundaryManager` owns implementation resources and lifecycle hooks;
- the current frozen particle behavior exists as an explicitly named legacy realization;
- aligned ghost particles consume the fluid `EqualMassSCPlan`;
- setup code does not mention boundary species, roles, ghost AABBs, counts, or placement;
- uniform-state density and acceleration tests pass near aligned boundaries;
- CPU tests and CUDA compilation pass.

## Completion criteria for the broader boundary system

The broader migration is complete when:

- periodic axes control both particle transport and cross-face neighbour interactions;
- reflecting wall models update state dynamically and implement documented transport semantics;
- reservoir, inflow, and outflow models define complete transport and exterior-support behavior;
- setup periodicity and PMacc topology have one validated source of truth;
- species roles remain capability-based and contain no boundary-model semantics;
- particle, virtual-image, and topology-based realizations share the same setup-facing boundary vocabulary;
- documentation clearly distinguishes physical models from realizations.
