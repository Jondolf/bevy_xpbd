//! Computes contacts between entities.
//!
//! See [`NarrowPhasePlugin`].

mod contact_data;
pub mod contact_query;

pub use contact_data::*;
pub use contact_query::*;

use crate::prelude::*;
#[cfg(feature = "parallel")]
use bevy::tasks::{ComputeTaskPool, ParallelSlice};

/// Computes contacts between entities.
///
/// Collisions are only checked between entities contained in [`BroadCollisionPairs`],
/// which is handled by the [`BroadPhasePlugin`].
///
/// The results of the narrow phase are added into [`Collisions`].
pub struct NarrowPhasePlugin;

impl Plugin for NarrowPhasePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<NarrowPhaseConfig>()
            .init_resource::<Collisions>()
            .register_type::<NarrowPhaseConfig>();

        let physics_schedule = app
            .get_schedule_mut(PhysicsSchedule)
            .expect("add PhysicsSchedule first");

        physics_schedule.add_systems(
            (
                // Reset collision states before the narrow phase
                (|mut collisions: ResMut<Collisions>| {
                    collisions.iter_mut().for_each(|contacts| {
                        contacts.during_previous_frame = contacts.during_current_frame;
                        contacts.during_current_frame = false;
                        contacts.during_current_substep = false;
                    })
                })
                .after(PhysicsStepSet::BroadPhase)
                .before(PhysicsStepSet::Substeps),
                // Remove ended collisions
                (|mut collisions: ResMut<Collisions>| {
                    collisions.retain(|contacts| contacts.during_current_frame)
                })
                .after(PhysicsStepSet::ReportContacts)
                .before(PhysicsStepSet::Sleeping),
            )
                .chain(),
        );

        let substep_schedule = app
            .get_schedule_mut(SubstepSchedule)
            .expect("add SubstepSchedule first");

        substep_schedule.add_systems(
            (reset_substep_collision_states, collect_collisions)
                .chain()
                .in_set(SubstepSet::NarrowPhase),
        );

        // Remove collisions against removed colliders from `Collisions`
        app.add_systems(
            Last,
            |mut removals: RemovedComponents<Collider>, mut collisions: ResMut<Collisions>| {
                for removed in removals.iter() {
                    collisions.remove_collisions_with_entity(removed);
                }
            },
        );
    }
}

/// A resource for configuring the [narrow phase](NarrowPhasePlugin).
#[derive(Resource, Reflect, Clone, Debug, PartialEq)]
#[reflect(Resource)]
pub struct NarrowPhaseConfig {
    /// The maximum separation distance allowed for a collision to be accepted.
    ///
    /// This can be used for things like **speculative contacts** where the contacts should
    /// include pairs of entities that *might* be in contact after constraint solving or
    /// other positional changes.
    pub prediction_distance: Scalar,
}

impl Default for NarrowPhaseConfig {
    fn default() -> Self {
        Self {
            #[cfg(feature = "2d")]
            prediction_distance: 5.0,
            #[cfg(feature = "3d")]
            prediction_distance: 0.005,
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
fn collect_collisions(
    bodies: Query<(
        Option<&RigidBody>,
        &Position,
        Option<&AccumulatedTranslation>,
        &Rotation,
        &Collider,
        Option<&CollisionLayers>,
        Option<&Sleeping>,
    )>,
    broad_collision_pairs: Res<BroadCollisionPairs>,
    mut collisions: ResMut<Collisions>,
    narrow_phase_config: Res<NarrowPhaseConfig>,
) {
    #[cfg(feature = "parallel")]
    {
        let pool = ComputeTaskPool::get();
        // Todo: Verify if `par_splat_map` is deterministic. If not, sort the collisions.
        let new_collisions = broad_collision_pairs
            .0
            .par_splat_map(pool, None, |chunks| {
                let mut new_collisions: Vec<Contacts> = vec![];
                for (entity1, entity2) in chunks {
                    if let Ok([bundle1, bundle2]) = bodies.get_many([*entity1, *entity2]) {
                        let (
                            rb1,
                            position1,
                            accumulated_translation1,
                            rotation1,
                            collider1,
                            layers1,
                            sleeping1,
                        ) = bundle1;
                        let (
                            rb2,
                            position2,
                            accumulated_translation2,
                            rotation2,
                            collider2,
                            layers2,
                            sleeping2,
                        ) = bundle2;

                        if check_collision_validity(
                            rb1, rb2, layers1, layers2, sleeping1, sleeping2,
                        ) {
                            let position1 = position1.0
                                + accumulated_translation1.copied().unwrap_or_default().0;
                            let position2 = position2.0
                                + accumulated_translation2.copied().unwrap_or_default().0;

                            let during_previous_frame = collisions
                                .get_internal()
                                .get(&(*entity1, *entity2))
                                .map_or(false, |c| c.during_previous_frame);

                            let contacts = Contacts {
                                entity1: *entity1,
                                entity2: *entity2,
                                during_current_frame: true,
                                during_current_substep: true,
                                during_previous_frame,
                                manifolds: contact_query::contact_manifolds(
                                    collider1,
                                    position1,
                                    *rotation1,
                                    collider2,
                                    position2,
                                    *rotation2,
                                    narrow_phase_config.prediction_distance,
                                ),
                            };

                            if !contacts.manifolds.is_empty() {
                                new_collisions.push(contacts);
                            }
                        }
                    }
                }
                new_collisions
            })
            .into_iter()
            .flatten();
        collisions.extend(new_collisions);
    }
    #[cfg(not(feature = "parallel"))]
    {
        for (entity1, entity2) in broad_collision_pairs.0.iter() {
            if let Ok([bundle1, bundle2]) = bodies.get_many([*entity1, *entity2]) {
                let (
                    rb1,
                    position1,
                    accumulated_translation1,
                    rotation1,
                    collider1,
                    layers1,
                    sleeping1,
                ) = bundle1;
                let (
                    rb2,
                    position2,
                    accumulated_translation2,
                    rotation2,
                    collider2,
                    layers2,
                    sleeping2,
                ) = bundle2;

                if check_collision_validity(rb1, rb2, layers1, layers2, sleeping1, sleeping2) {
                    let position1 =
                        position1.0 + accumulated_translation1.copied().unwrap_or_default().0;
                    let position2 =
                        position2.0 + accumulated_translation2.copied().unwrap_or_default().0;

                    let during_previous_frame = collisions
                        .get_internal()
                        .get(&(*entity1, *entity2))
                        .map_or(false, |c| c.during_previous_frame);

                    let contacts = Contacts {
                        entity1: *entity1,
                        entity2: *entity2,
                        during_current_frame: true,
                        during_current_substep: true,
                        during_previous_frame,
                        manifolds: contact_query::contact_manifolds(
                            collider1,
                            position1,
                            *rotation1,
                            collider2,
                            position2,
                            *rotation2,
                            narrow_phase_config.prediction_distance,
                        ),
                    };

                    if !contacts.manifolds.is_empty() {
                        collisions.insert_collision_pair(contacts);
                    }
                }
            }
        }
    }
}

fn check_collision_validity(
    rb1: Option<&RigidBody>,
    rb2: Option<&RigidBody>,
    layers1: Option<&CollisionLayers>,
    layers2: Option<&CollisionLayers>,
    sleeping1: Option<&Sleeping>,
    sleeping2: Option<&Sleeping>,
) -> bool {
    let layers1 = layers1.map_or(CollisionLayers::default(), |l| *l);
    let layers2 = layers2.map_or(CollisionLayers::default(), |l| *l);

    // Skip collision if collision layers are incompatible
    if !layers1.interacts_with(layers2) {
        return false;
    }

    let inactive1 = rb1.map_or(false, |rb| rb.is_static()) || sleeping1.is_some();
    let inactive2 = rb2.map_or(false, |rb| rb.is_static()) || sleeping2.is_some();

    // No collision if the bodies are static or sleeping
    if inactive1 && inactive2 {
        return false;
    }

    true
}

fn reset_substep_collision_states(mut collisions: ResMut<Collisions>) {
    for contacts in collisions.get_internal_mut().values_mut() {
        contacts.during_current_substep = false;
    }
}
