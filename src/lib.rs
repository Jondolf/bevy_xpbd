//! # Bevy XPBD
//!
//! **Bevy XPBD** is a 2D and 3D physics engine based on
//! [*Extended Position Based Dynamics* (XPBD)](#what-is-xpbd) for
//! the [Bevy game engine](https://bevyengine.org/).
//!
//! Check out the [GitHub repository](https://github.com/Jondolf/bevy_xpbd)
//! for more information about the design, read the [Getting started](#getting-started)
//! guide below to get up to speed, and take a look at the [Table of contents](#table-of-contents)
//! for an overview of the engine's features and their documentation.
//!
//! You can also check out the [FAQ](#frequently-asked-questions), and if you encounter
//! any further problems, consider saying hello on the [Bevy Discord](https://discord.gg/bevy)!
//!
//! ## Getting started
//!
//! This short guide should help you get started with Bevy XPBD.
//!
//! ### Add the dependency
//!
//! First, add `bevy_xpbd_2d` or `bevy_xpbd_3d` to the dependencies in your `Cargo.toml`:
//!  
//! ```toml
//! # For 2D applications:
//! [dependencies]
//! bevy_xpbd_2d = "0.4"
//!
//! # For 3D applications:
//! [dependencies]
//! bevy_xpbd_3d = "0.4"
//!
//! # If you want to use the most up-to-date version, you can follow the main branch:
//! [dependencies]
//! bevy_xpbd_3d = { git = "https://github.com/Jondolf/bevy_xpbd", branch = "main" }
//! ```
//!
//! You can specify features by disabling the default features and manually adding
//! the feature flags you want:
//!
//! ```toml
//! [dependencies]
//! # Add 3D Bevy XPBD with double-precision floating point numbers.
//! # `parry-f64` enables collision detection using Parry.
//! bevy_xpbd_3d = { version = "0.4", default-features = false, features = ["3d", "f64", "parry-f64"] }
//! ```
//!
//! ### Feature flags
//!
//! | Feature                | Description                                                                                                                                  | Default feature         |
//! | ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------- |
//! | `2d`                   | Enables 2D physics. Incompatible with `3d`.                                                                                                  | Yes (`bevy_xpbd_2d`)    |
//! | `3d`                   | Enables 3D physics. Incompatible with `2d`.                                                                                                  | Yes (`bevy_xpbd_3d`)    |
//! | `f32`                  | Enables `f32` precision for physics. Incompatible with `f64`.                                                                                | Yes                     |
//! | `f64`                  | Enables `f64` precision for physics. Incompatible with `f32`.                                                                                | No                      |
//! | `default-collider`     | Enables the default [`Collider`]. Required for [spatial queries](spatial_query). Requires either the `parry-f32` or `parry-f64` feature.     | Yes                     |
//! | `parry-f32`            | Enables the `f32` version of the Parry collision detection library. Also enables the `default-collider` feature.                             | Yes                     |
//! | `parry-f64`            | Enables the `f64` version of the Parry collision detection library. Also enables the `default-collider` feature.                             | No                      |
#![cfg_attr(
    feature = "3d",
    doc = "| `collider-from-mesh`   | Allows you to create [`Collider`]s from `Mesh`es.                                                                                | Yes                     |"
)]
#![cfg_attr(
    feature = "3d",
    doc = "| `async-collider`       | Allows you to generate [`Collider`]s from mesh handles and scenes.                                                               | Yes                     |"
)]
//! | `debug-plugin`         | Enables physics debug rendering using the [`PhysicsDebugPlugin`]. The plugin must be added separately.                           | Yes                     |
//! | `enhanced-determinism` | Enables increased determinism.                                                                                                   | No                      |
//! | `parallel`             | Enables some extra multithreading, which improves performance for larger simulations but can add some overhead for smaller ones. | Yes                     |
//! | `simd`                 | Enables [SIMD] optimizations.                                                                                                    | No                      |
//! | `serialize`            | Enables support for serialization and deserialization using Serde.                                                               | No                      |
//!
//! [SIMD]: https://en.wikipedia.org/wiki/Single_instruction,_multiple_data
//!
//! ### Install the plugin
//!
//! Bevy XPBD is designed to be very modular. It is built from several [plugins](PhysicsPlugins) that
//! manage different parts of the engine. These plugins can be easily initialized and configured through
//! the [`PhysicsPlugins`] plugin group.
//!
//! ```no_run
//! use bevy::prelude::*;
#![cfg_attr(feature = "2d", doc = "use bevy_xpbd_2d::prelude::*;")]
#![cfg_attr(feature = "3d", doc = "use bevy_xpbd_3d::prelude::*;")]
//!
//! fn main() {
//!     App::new()
//!         .add_plugins((DefaultPlugins, PhysicsPlugins::default()))
//!         // ...your other plugins, systems and resources
//!         .run();
//! }
//! ```
//!
//! Now you can use all of Bevy XPBD's components and resources to build whatever you want!
//!
//! For example, adding a [rigid body](RigidBody) with a [collider](Collider) is as simple as spawning an entity
//! with the [`RigidBody`] and [`Collider`] components:
//!
//! ```
//! use bevy::prelude::*;
#![cfg_attr(feature = "2d", doc = "use bevy_xpbd_2d::prelude::*;")]
#![cfg_attr(feature = "3d", doc = "use bevy_xpbd_3d::prelude::*;")]
//!
//! fn setup(mut commands: Commands) {
#![cfg_attr(
    feature = "2d",
    doc = "    commands.spawn((RigidBody::Dynamic, Collider::circle(0.5)));"
)]
#![cfg_attr(
    feature = "3d",
    doc = "    commands.spawn((RigidBody::Dynamic, Collider::sphere(0.5)));"
)]
//! }
//! ```
//!
//! You can find lots of [usage examples](https://github.com/Jondolf/bevy_xpbd#more-examples)
//! in the project's [repository](https://github.com/Jondolf/bevy_xpbd).
//!
//! ## Table of contents
//!
//! Below is a structured overview of the documentation for the various
//! features of the engine.
//!
//! ### Rigid bodies
//!
//! - [Rigid body types](RigidBody#rigid-body-types)
//! - [Creating rigid bodies](RigidBody#creation)
//! - [Movement](RigidBody#movement)
//!     - [Linear](LinearVelocity) and [angular](AngularVelocity) velocity
//!     - [Forces](ExternalForce), [torque](ExternalTorque), and [linear](ExternalImpulse) and [angular](ExternalAngularImpulse) impulses
//! - [Gravity] and [gravity scale](GravityScale)
//! - [Mass properties](RigidBody#mass-properties)
//! - [Linear](LinearDamping) and [angular](AngularDamping) velocity damping
//! - [Lock translational and rotational axes](LockedAxes)
//! - [Dominance]
//! - [Automatic deactivation with sleeping](Sleeping)
//!
//! ### Collision detection
//!
//! - [Colliders](Collider)
//!     - [Creation](Collider#creation)
//!     - [Density](ColliderDensity)
//!     - [Friction] and [restitution](Restitution) (bounciness)
//!     - [Collision layers](CollisionLayers)
//!     - [Sensors](Sensor)
#![cfg_attr(
    feature = "3d",
    doc = "    - Creating colliders from meshes with [`AsyncCollider`] and [`AsyncSceneCollider`]"
)]
//! - [Get colliding entities](CollidingEntities)
//! - [Collision events](ContactReportingPlugin#collision-events)
//! - [Accessing, filtering and modifying collisions](Collisions)
//! - [Manual contact queries](contact_query)
//!
//! ### Constraints and joints
//!
//! - [Constraints](solver::xpbd#constraints) (advanced)
//! - [Joints](solver::joints)
//!     - [Fixed joint](FixedJoint)
//!     - [Distance joint](DistanceJoint)
//!     - [Prismatic joint](PrismaticJoint)
//!     - [Revolute joint](RevoluteJoint)
//!     - [Spherical joint](SphericalJoint)
//!
//! Joint motors and articulations are not supported yet, but they will be implemented in a future release.
//!
//! ### Spatial queries
//!
//! - [Spatial query types](spatial_query)
//!     - [Raycasting](spatial_query#raycasting) and [`RayCaster`]
//!     - [Shapecasting](spatial_query#shapecasting) and [`ShapeCaster`]
//!     - [Point projection](spatial_query#point-projection)
//!     - [Intersection tests](spatial_query#intersection-tests)
//! - [Spatial query filters](SpatialQueryFilter)
//! - [The `SpatialQuery` system parameter](SpatialQuery)
//!
//! ### Configuration
//!
//! - [Gravity]
//! - [Physics timestep](Physics#usage)
//! - [Physics speed](Physics#physics-speed)
//! - [Configure simulation fidelity with substeps](SubstepCount)
//! - [Render physics objects for debugging](PhysicsDebugPlugin)
//!
//! ### Scheduling
//!
//! - [Schedules and sets](PhysicsSchedulePlugin#schedules-and-sets)
//!     - [`PhysicsSet`]
//!     - [`PhysicsSchedule`] and [`PhysicsStepSet`]
//!     - [`SubstepSchedule`] and [`SubstepSet`]
//!     - [`PostProcessCollisions`] schedule
//!     - [`PrepareSet`](prepare::PrepareSet)
//! - [Configure the schedule used for running physics](PhysicsPlugins#custom-schedule)
//! - [Pausing, resuming and stepping physics](Physics#pausing-resuming-and-stepping-physics)
//! - [Usage on servers](#can-the-engine-be-used-on-servers)
//!
//! ### Architecture
//!
//! - [List of plugins and their responsibilities](PhysicsPlugins)
//! - [What is Extended Position Based Dynamics?](#what-is-xpbd)
//! - Extending and modifying the engine
//!     - [Custom plugins](PhysicsPlugins#custom-plugins)
//!     - [Custom constraints](solver::xpbd#custom-constraints)
//!     - [Custom joints](solver::joints#custom-joints)
//!
//! ## Frequently asked questions
//!
//! - [How does Bevy XPBD compare to Rapier and bevy_rapier?](#how-does-bevy-xpbd-compare-to-rapier-and-bevy_rapier)
//! - [Why is nothing happening?](#why-is-nothing-happening)
//! - [Why is everything moving so slowly?](#why-is-everything-moving-so-slowly)
//! - [Why did my rigid body suddenly vanish?](#why-did-my-rigid-body-suddenly-vanish)
//! - [Why is performance so bad?](#why-is-performance-so-bad)
//! - [Why does my camera following jitter?](#why-does-my-camera-following-jitter)
//! - [Is there a character controller?](#is-there-a-character-controller)
//! - [Why are there separate `Position` and `Rotation` components?](#why-are-there-separate-position-and-rotation-components)
//! - [Can the engine be used on servers?](#can-the-engine-be-used-on-servers)
//! - [Something else?](#something-else)
//!
//! ### How does Bevy XPBD compare to Rapier and bevy_rapier?
//!
//! Rapier is the biggest and most used physics engine in the Rust ecosystem, and it is currently
//! the most mature and feature-rich option.
//!
//! `bevy_rapier` is a great physics integration for Bevy, but it does have several problems:
//!
//! - It has to maintain a separate physics world and synchronize a ton of data with Bevy each frame
//! - The source code is difficult to inspect, as the vast majority of it is glue code and wrappers
//! for Bevy
//! - It has poor docs.rs documentation, and the documentation on rapier.rs is often outdated and
//! missing features
//! - It is hard to extend as it's not very modular or composable in design
//! - Overall, it doesn't have a native ECS-like feel outside of its public API
//!
//! Bevy XPBD on the other hand is built *for* Bevy *with* Bevy, and it uses the ECS for both the internals
//! and the public API. This removes the need for a separate physics world, reduces overhead, and makes
//! the source code much more approachable and easy to inspect for Bevy users.
//!
//! In part thanks to Bevy's modular architecture and the ECS, Bevy XPBD is also highly composable,
//! as it consists of several independent plugins and provides lots of options for configuration and extensions,
//! from [custom schedules](PhysicsPlugins#custom-schedule) and [plugins](PhysicsPlugins#custom-plugins) to
//! [custom joints](solver::joints#custom-joints) and [constraints](solver::xpbd#custom-constraints).
//!
//! In terms of the physics implementation, Rapier uses an impulse/velocity based solver, while Bevy XPBD uses
//! [Extended Position Based Dynamics](#what-is-xpbd). On paper, XPBD should be more stable and robust,
//! but it hasn't been widely adopted in mainstream usage yet.
//!
//! One of the biggest disadvantages of Bevy XPBD is that it is still very young, so it can have lots of bugs,
//! some missing features, and fewer community resources and third party crates. However, it is growing quite
//! rapidly, and it is already pretty close to feature-parity with Rapier.
//!
//! At the end of the day, both engines are very solid options. If you are looking for a more mature and tested
//! physics integration, `bevy_rapier` is the better choice, but if you prefer an engine with less overhead
//! and a more native Bevy integration, consider using Bevy XPBD. Their core APIs are also quite similar,
//! so switching between them should be straightforward.
//!
//! ### Why is nothing happening?
//!
//! Make sure you have added the [`PhysicsPlugins`] plugin group and you have given your rigid bodies
//! a [`RigidBody`] component. See the [getting started](#getting-started) section.
//!
//! ### Why is everything moving so slowly?
//!
//! If your application is in 2D, you might be using pixels as length units. This will require you to use
//! larger velocities and forces than you would in 3D. Make sure you set [`Gravity`] to some larger value
//! as well, because its magnitude is 9.81 by default, which is tiny in pixels.
//!
//! Bevy XPBD doesn't have a "physics scale" yet, but it will most likely be added in the future
//! so that it's possible to define some kind of pixels per meter configuration.
//!
//! ### Why did my rigid body suddenly vanish?
//!
//! Make sure to [give your rigid bodies some mass](RigidBody#adding-mass-properties), either by adding a [`Collider`]
//! or a [`MassPropertiesBundle`]. If your bodies don't have any mass, any physical interaction is likely to
//! instantly give them infinite velocity.
//!
//! Bevy XPBD should automatically print warnings when it detects bodies with an invalid mass or inertia.
//!
//! ### Why is performance so bad?
//!
//! Make sure you are building your project in release mode using `cargo build --release`.
//!
//! You can further optimize builds by setting the number of codegen units in your `Cargo.toml` to 1,
//! although this will also increase build times.
//!
//! ```toml
//! [profile.release]
//! codegen-units = 1
//! ```
//!
//! Note that Bevy XPBD simply isn't very optimized yet, and it mostly runs on a single thread for now.
//! This will be addressed in future releases.
//!
//! ### Why does my camera following jitter?
//!
//! When you write a system that makes the camera follow a physics entity, you might notice some jitter.
//!
//! To fix this, the system needs to:
//!
//! - Run after physics so that it has the up-to-date position of the player.
//! - Run before transform propagation so that your changes to the camera's `Transform` are written
//! to the camera's `GlobalTransform` before the end of the frame.
//!
//! The following ordering constraints should resolve the issue.
//!
//! ```
//! # use bevy::prelude::*;
//! # use bevy::transform::TransformSystem;
#![cfg_attr(feature = "2d", doc = "# use bevy_xpbd_2d::prelude::*;")]
#![cfg_attr(feature = "3d", doc = "# use bevy_xpbd_3d::prelude::*;")]
//! #
//! # fn main() {
//! #     let mut app = App::new();
//! #
//! app.add_systems(
//!     PostUpdate,
//!     camera_follow_player
//!         .after(PhysicsSet::Sync)
//!         .before(TransformSystem::TransformPropagate),
//! );
//! # }
//! #
//! # fn camera_follow_player() {}
//! ```
//!
//! ### Is there a character controller?
//!
//! Bevy XPBD does not have a built-in character controller, so if you need one,
//! you will need to implement it yourself. However, third party character controllers
//! like [`bevy_tnua`](https://github.com/idanarye/bevy-tnua) support Bevy XPBD, and [`bevy_mod_wanderlust`](https://github.com/PROMETHIA-27/bevy_mod_wanderlust)
//! and others are also likely to get Bevy XPBD support soon.
//!
//! For custom character controllers, you can take a look at the
#![cfg_attr(
    feature = "2d",
    doc = "[`dynamic_character_2d`] and [`kinematic_character_2d`] examples to get started."
)]
#![cfg_attr(
    feature = "3d",
    doc = "[`dynamic_character_3d`] and [`kinematic_character_3d`] examples to get started."
)]
//!
#![cfg_attr(
    feature = "2d",
    doc = "[`dynamic_character_2d`]: https://github.com/Jondolf/bevy_xpbd/tree/main/crates/bevy_xpbd_2d/examples/dynamic_character_2d
[`kinematic_character_2d`]: https://github.com/Jondolf/bevy_xpbd/tree/main/crates/bevy_xpbd_2d/examples/kinematic_character_2d"
)]
#![cfg_attr(
    feature = "3d",
    doc = "[`dynamic_character_3d`]: https://github.com/Jondolf/bevy_xpbd/tree/main/crates/bevy_xpbd_3d/examples/dynamic_character_3d
[`kinematic_character_3d`]: https://github.com/Jondolf/bevy_xpbd/tree/main/crates/bevy_xpbd_3d/examples/kinematic_character_3d"
)]
//!
//! ### Why are there separate `Position` and `Rotation` components?
//!
//! While `Transform` can be used for the vast majority of things, Bevy XPBD internally
//! uses separate [`Position`] and [`Rotation`] components. These are automatically
//! kept in sync by the [`SyncPlugin`].
//!
//! There are several reasons why the separate components are currently used.
//!
//! - Position and rotation should be global from the physics engine's point of view.
//! - Transform scale and shearing can cause issues and rounding errors in physics.
//! - Transform hierarchies can be problematic.
//! - There is no `f64` version of `Transform`.
//! - There is no 2D version of `Transform` (yet), and having a 2D version can optimize several computations.
//! - When position and rotation are separate, we can technically have more systems running in parallel.
//! - Only rigid bodies have rotation, particles typically don't (although we don't make a distinction yet).
//!
//! In external projects however, using [`Position`] and [`Rotation`] is only necessary when you
//! need to manage positions in the [`SubstepSchedule`]. Elsewhere, you should be able to use `Transform`.
//!
//! There is also a possibility that we will revisit this if/when Bevy has a `Transform2d` component.
//! Using `Transform` feels more idiomatic and simple, so it would be nice if it could be used directly
//! as long as we can get around the drawbacks.
//!
//! ### Can the engine be used on servers?
//!
//! Yes! Networking often requires running the simulation in a specific schedule, and in Bevy XPBD you can
//! [set the schedule that runs physics](PhysicsPlugins#custom-schedule) and [configure the timestep](Physics)
//! to whatever you want.
//!
//! One configuration is to run the client in `FixedUpdate`, and to use a fixed timestep for [`Time<Physics>`](Physics)
//! on both the server and the client to make sure the physics simulation is only advanced by one step
//! each time the schedule runs.
//!
//! Note that while Bevy XPBD should be locally deterministic, it can produce slightly different results on different
//! machines.
//!
//! ### Something else?
//!
//! Physics engines are very large and Bevy XPBD is young, so stability issues and bugs are to be expected.
//!
//! If you encounter issues, please consider first taking a look at the
//! [issues on GitHub](https://github.com/Jondolf/bevy_xpbd/issues) and
//! [open a new issue](https://github.com/Jondolf/bevy_xpbd/issues/new) if there already isn't one regarding your problem.
//!
//! You can also come and say hello on the [Bevy Discord server](https://discord.com/invite/gMUk5Ph).
//! There you can find a bevy_xpbd thread on the crate-help channel where you can ask questions.
//!
//! ## What is XPBD?
//!
//! *XPBD* or *Extended Position Based Dynamics* is a physics simulation method that extends
//! the traditional *PBD* to be more physically accurate and less dependent on time step size
//! and iteration count.
//!
//! Unlike force or impulse based physics simulation methods, XPBD mostly operates at the position-level,
//! which can produce more stable and reliable results, while allowing straightforward coupling
//! of [rigid bodies](RigidBody), soft bodies and fluids.
//!
//! ### Simulation loop
//!
//! At a high level, XPBD consists of a broad phase followed by a substepping loop that handles position
//! [integration](integrator), [constraint solving](solver), velocity updates, and a velocity solver that
//! handles dynamic friction and restitution.
//!
//! It looks roughly like this:
//!
//! ```ignore
//! while simulating:
//!     // Substep size
//!     h = ∆t / substep_count
//!
//!     // Broad phase
//!     broad_collision_pairs = collect_collision_pairs()
//!
//!     for substep_count:
//!         // Integrate
//!         for n particles and bodies:
//!             // Integrate position
//!             x_prev = x
//!             v = v + h * f_ext / m
//!             x = x + h * v
//!
//!             // Integrate rotation
//!             q_prev = q
//!             ω = ω + h * I^-1 * (τ_ext - (ω x (I * ω)))
//!             q = q + h * 0.5 * [ω_x, ω_y, ω_z, 0] * q
//!             q = q / |q|
//!
//!         // Narrow phase
//!         for pair in broad_collision_pairs:
//!             compute_contacts(pair)
//!
//!         // Solve constraints (contacts, joints etc.)
//!         solve_constraints(particles and bodies)
//!
//!         // Update velocities
//!         for n particles and bodies:
//!             v = (x - x_prev) / h
//!             ∆q = q * q_prev^-1
//!             ω = 2 * [∆q_x, ∆q_y, ∆q_z] / h
//!             ω = ∆q_w >= 0 ? ω : -ω
//!
//!         // Solve velocity constraints (dynamic friction and restitution)
//!         solve_velocities(particles and bodies)
//! ```
//!
//! where `h` is the substep size, `q` is the [rotation](Rotation) as a quaternion,
//! `ω` is the [angular velocity](AngularVelocity), `I` is the [angular inertia tensor](`Inertia`) and `τ` is the
//! [external torque](ExternalTorque).
//!
//! In Bevy XPBD, the simulation loop is handled by various plugins. The [`PhysicsSchedulePlugin`] sets up
//! the Bevy schedules[^1][^2] and system sets[^3][^4][^5], the [`BroadPhasePlugin`] manages the broad phase,
//! the [`IntegratorPlugin`] handles position and velocity integration, and so on. You can find all of the plugins
//! and their responsibilities [here](PhysicsPlugins).
//!
//! ### See also
//!
//! - [XPBD integration step](integrator)
//! - [Constraints and how to create them](solver::xpbd#constraints)
//! - [Schedules and sets used for the simulation loop](PhysicsSchedulePlugin#schedules-and-sets)
//!
//! ### Learning resources
//!
//! If you want to learn more about XPBD, I recommend taking a look at some of the papers.
//! Especially the first one from 2020 was used heavily for the simulation loop and constraints in Bevy XPBD.
//!
//! - XPBD: Müller M, Macklin M, Chentanez N, Jeschke S, Kim T. 2020. *[Detailed Rigid Body Simulation with Extended Position Based Dynamics](https://matthias-research.github.io/pages/publications/PBDBodies.pdf)*.
//! - XPBD: Macklin M, Müller M, Chentanez N. 2016. *[XPBD: Position-Based Simulation of Compliant Constrained Dynamics](http://mmacklin.com/xpbd.pdf)*.
//!
//! The papers are quite academic, so you might instead prefer some videos and articles.
//! The first one by Ten Minute Physics (Matthias Müller, one of the XPBD researchers) is great for understanding
//! how XPBD differs from other simulation methods and how the constraints work.
//!
//! - Video: Ten Minute Physics. 2022. *[Getting ready to simulate the world with XPBD](https://youtu.be/jrociOAYqxA)*.
//! - Notes: Nolan Tait. *[Bevy Physics: XPBD](https://taintedcoders.com/bevy/xpbd/)*.
//! - Tutorial series: Johan Helsing. *[Tutorial: Making a physics engine with Bevy](https://johanhelsing.studio/posts/bevy-xpbd)*.
//! (inspired this project)
//!
//! ## License
//!
//! Bevy XPBD is free and open source. All code in the Bevy XPBD repository is dual-licensed under either:
//!
//! - MIT License ([LICENSE-MIT](https://github.com/Jondolf/bevy_xpbd/blob/main/LICENSE-MIT)
//! or <http://opensource.org/licenses/MIT>)
//! - Apache License, Version 2.0 ([LICENSE-APACHE](https://github.com/Jondolf/bevy_xpbd/blob/main/LICENSE-APACHE)
//! or <http://www.apache.org/licenses/LICENSE-2.0>)
//!
//! at your option.
//!
//! [^1]: [`PhysicsSchedule`]
//!
//! [^2]: [`SubstepSchedule`]
//!
//! [^3]: [`PhysicsSet`]
//!
//! [^4]: [`PhysicsStepSet`]
//!
//! [^5]: [`SubstepSet`]

#![allow(rustdoc::invalid_rust_codeblocks)]
#![warn(clippy::doc_markdown, missing_docs)]

#[cfg(all(not(feature = "f32"), not(feature = "f64")))]
compile_error!("either feature \"f32\" or \"f64\" must be enabled");

#[cfg(all(feature = "f32", feature = "f64"))]
compile_error!("feature \"f32\" and feature \"f64\" cannot be enabled at the same time");

#[cfg(all(not(feature = "2d"), not(feature = "3d")))]
compile_error!("either feature \"2d\" or \"3d\" must be enabled");

#[cfg(all(feature = "2d", feature = "3d"))]
compile_error!("feature \"2d\" and feature \"3d\" cannot be enabled at the same time");

#[cfg(all(
    feature = "default-collider",
    feature = "f32",
    not(feature = "parry-f32")
))]
compile_error!(
    "feature \"default-collider\" requires the feature \"parry-f32\" when \"f32\" is enabled"
);

#[cfg(all(
    feature = "default-collider",
    feature = "f64",
    not(feature = "parry-f64")
))]
compile_error!(
    "feature \"default-collider\" requires the feature \"parry-f64\" when \"f64\" is enabled"
);

#[cfg(all(feature = "2d", feature = "parry-f32"))]
pub extern crate parry2d as parry;

#[cfg(all(feature = "2d", feature = "parry-f64"))]
pub extern crate parry2d_f64 as parry;

#[cfg(all(feature = "3d", feature = "parry-f32"))]
pub extern crate parry3d as parry;

#[cfg(all(feature = "3d", feature = "parry-f64"))]
pub extern crate parry3d_f64 as parry;

pub mod collision;
#[cfg(feature = "debug-plugin")]
pub mod debug_render;
pub mod dynamics;
pub mod math;
pub mod position;
pub mod prepare;
pub mod schedule;
pub mod spatial_query;
pub mod sync;

/// Re-exports common components, bundles, resources, plugins and types.
pub mod prelude {
    #[cfg(feature = "debug-plugin")]
    pub use crate::debug_render::*;
    pub use crate::{
        collision::{
            self,
            broad_phase::{BroadCollisionPairs, BroadPhasePlugin},
            collider_backend::ColliderBackendPlugin,
            contact_reporting::{
                Collision, CollisionEnded, CollisionStarted, ContactReportingPlugin,
            },
            narrow_phase::{NarrowPhaseConfig, NarrowPhasePlugin},
            *,
        },
        dynamics::{
            self,
            sleeping::SleepingPlugin,
            solver::{joints::*, SolverPlugin},
            *,
        },
        position::{Position, Rotation},
        prepare::{init_transforms, update_mass_properties, PrepareConfig, PreparePlugin},
        schedule::*,
        spatial_query::{self, *},
        sync::SyncPlugin,
        type_registration::PhysicsTypeRegistrationPlugin,
        PhysicsPlugins,
    };
    pub(crate) use crate::{
        math::*,
        position::{PreviousPosition, PreviousRotation, RotationValue},
    };
    pub use bevy_xpbd_derive::*;
}

mod utils;

#[cfg(test)]
mod tests;

use bevy::{
    app::PluginGroupBuilder, ecs::intern::Interned, ecs::schedule::ScheduleLabel, prelude::*,
};
#[allow(unused_imports)]
use prelude::*;

mod type_registration;

/// A plugin group containing all of Bevy XPBD's plugins.
///
/// By default, the following plugins will be added:
///
/// - [`PhysicsSchedulePlugin`]: Sets up the physics engine by initializing the necessary schedules, sets and resources.
/// - [`PhysicsTypeRegistrationPlugin`]: Registers physics types to the `TypeRegistry` resource in `bevy_reflect`.
/// - [`PreparePlugin`]: Runs systems at the start of each physics frame; initializes [rigid bodies](RigidBody)
/// and [colliders](Collider) and updates components.
/// - [`ColliderBackendPlugin`]: Handles generic collider backend logic, like initializing colliders and AABBs
/// and updating related components.
/// - [`BroadPhasePlugin`]: Collects pairs of potentially colliding entities into [`BroadCollisionPairs`] using
/// [AABB](ColliderAabb) intersection checks.
/// - [`NarrowPhasePlugin`]: Computes contacts between entities and sends collision events.
/// - [`ContactReportingPlugin`]: Sends collision events and updates [`CollidingEntities`].
/// - [`IntegratorPlugin`]: Handles motion caused by velocity, and applies external forces and gravity.
/// - [`SolverPlugin`]: Solves [constraints](solver::xpbd#constraints) (contacts and joints).
/// (dynamic [friction](Friction) and [restitution](Restitution)).
/// - [`SleepingPlugin`]: Manages sleeping and waking for bodies, automatically deactivating them to save computational resources.
/// - [`SpatialQueryPlugin`]: Handles spatial queries like [raycasting](spatial_query#raycasting) and [shapecasting](spatial_query#shapecasting).
/// - [`SyncPlugin`]: Keeps [`Position`] and [`Rotation`] in sync with `Transform`.
/// - [`PhysicsDebugPlugin`]: Renders physics objects and events like [AABBs](ColliderAabb) and [contacts](Collision)
/// for debugging purposes (only with `debug-plugin` feature enabled).
///
/// Refer to the documentation of the plugins for more information about their responsibilities and implementations.
///
/// ## Custom schedule
///
/// You can run the [`PhysicsSchedule`] in any schedule you want by specifying the schedule when adding the plugin group:
///
/// ```no_run
/// use bevy::prelude::*;
#[cfg_attr(feature = "2d", doc = "use bevy_xpbd_2d::prelude::*;")]
#[cfg_attr(feature = "3d", doc = "use bevy_xpbd_3d::prelude::*;")]
///
/// fn main() {
///     App::new()
///         .add_plugins((DefaultPlugins, PhysicsPlugins::new(FixedUpdate)))
///         .run();
/// }
/// ```
///
/// Note that using `FixedUpdate` with a fixed [physics timestep](Physics) can produce unexpected results due to two separate
/// fixed timesteps. However, using `FixedUpdate` can be useful for [networking usage](crate#can-the-engine-be-used-on-servers)
/// when you need to keep the client and server in sync.
///
/// ## Custom plugins
///
/// First, create a new plugin. If you want to run your systems in the engine's schedules, get either the [`PhysicsSchedule`]
/// or the [`SubstepSchedule`]. Then you can add your systems to that schedule and control system ordering with
/// [`PhysicsSet`] or [`SubstepSet`].
///
/// Here we will create a custom broad phase plugin that will replace the default [`BroadPhasePlugin`]:
///
/// ```
/// use bevy::prelude::*;
#[cfg_attr(
    feature = "2d",
    doc = "use bevy_xpbd_2d::{prelude::*, PhysicsSchedule, PhysicsStepSet};"
)]
#[cfg_attr(
    feature = "3d",
    doc = "use bevy_xpbd_3d::{prelude::*, PhysicsSchedule, PhysicsStepSet};"
)]
///
/// pub struct CustomBroadPhasePlugin;
///
/// impl Plugin for CustomBroadPhasePlugin {
///     fn build(&self, app: &mut App) {
///         // Make sure the PhysicsSchedule is available
///         let physics_schedule = app
///             .get_schedule_mut(PhysicsSchedule)
///             .expect("add PhysicsSchedule first");
///
///         // Add the system into the broad phase system set
///         physics_schedule.add_systems(collect_collision_pairs.in_set(PhysicsStepSet::BroadPhase));
///     }
/// }
///
/// fn collect_collision_pairs() {
///     // Implementation goes here
/// }
/// ```
///
/// Next, when creating your app, simply disable the default [`BroadPhasePlugin`] and add your custom plugin:
///
/// ```no_run
/// use bevy::prelude::*;
#[cfg_attr(feature = "2d", doc = "use bevy_xpbd_2d::prelude::*;")]
#[cfg_attr(feature = "3d", doc = "use bevy_xpbd_3d::prelude::*;")]
///
/// # struct CustomBroadPhasePlugin;
/// # impl Plugin for CustomBroadPhasePlugin {
/// #     fn build(&self, app: &mut App) {}
/// # }
/// #
/// fn main() {
///     let mut app = App::new();
///
///     app.add_plugins(DefaultPlugins);
///
///     // Add PhysicsPlugins and replace default broad phase with our custom broad phase
///     app.add_plugins(
///         PhysicsPlugins::default()
///             .build()
///             .disable::<BroadPhasePlugin>()
///             .add(CustomBroadPhasePlugin),
///     );
///
///     app.run();
/// }
/// ```
///
/// You can find a full working example
/// [here](https://github.com/Jondolf/bevy_xpbd/blob/main/crates/bevy_xpbd_3d/examples/custom_broad_phase.rs).
pub struct PhysicsPlugins {
    schedule: Interned<dyn ScheduleLabel>,
}

impl PhysicsPlugins {
    /// Creates a [`PhysicsPlugins`] plugin group using the given schedule for running the [`PhysicsSchedule`].
    ///
    /// The default schedule is `PostUpdate`.
    pub fn new(schedule: impl ScheduleLabel) -> Self {
        Self {
            schedule: schedule.intern(),
        }
    }
}

impl Default for PhysicsPlugins {
    fn default() -> Self {
        Self::new(PostUpdate)
    }
}

impl PluginGroup for PhysicsPlugins {
    fn build(self) -> PluginGroupBuilder {
        let builder = PluginGroupBuilder::start::<Self>()
            .add(PhysicsSchedulePlugin::new(self.schedule))
            .add(PhysicsTypeRegistrationPlugin)
            .add(PreparePlugin::new(self.schedule));

        #[cfg(all(
            feature = "default-collider",
            any(feature = "parry-f32", feature = "parry-f64")
        ))]
        let builder = builder
            .add(ColliderBackendPlugin::<Collider>::new(self.schedule))
            .add(NarrowPhasePlugin::<Collider>::default());

        builder
            .add(BroadPhasePlugin)
            .add(IntegratorPlugin)
            .add(ContactReportingPlugin)
            .add(SolverPlugin)
            .add(SleepingPlugin)
            .add(SpatialQueryPlugin::new(self.schedule))
            .add(SyncPlugin::new(self.schedule))
    }
}
