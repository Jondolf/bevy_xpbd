//! Pluggable dispatch for the geometric queries between
//! [`Collider`](crate::collision::collider::Collider) shapes.

#[cfg(any(feature = "parry-f32", feature = "parry-f64"))]
use crate::math::Real;
use bevy::ecs::resource::Resource;
#[cfg(any(feature = "parry-f32", feature = "parry-f64"))]
use parry::{
    math::{Pose, Vector},
    query::{
        ClosestPoints, Contact, ContactManifold, ContactManifoldsWorkspace, DefaultQueryDispatcher,
        NonlinearRigidMotion, PersistentQueryDispatcher, QueryDispatcher as _, ShapeCastHit,
        ShapeCastOptions, Unsupported, details::NormalConstraints,
    },
    shape::Shape,
};

/// The Bevy resource specifying the dispatcher used for all pairwise geometric queries.
///
/// By default the resource is empty and queries are dispatched with Parry's
/// [`DefaultQueryDispatcher`], which supports every Parry built-in shape. To support
/// a custom [`Shape`], install a dispatcher that recognizes the
/// custom shape — usually chained with the default dispatcher as a fallback.
///
#[cfg_attr(any(feature = "parry-f32", feature = "parry-f64"), doc = "```no_run")]
#[cfg_attr(
    not(any(feature = "parry-f32", feature = "parry-f64")),
    doc = "```ignore"
)]
#[cfg_attr(feature = "2d", doc = "# use avian2d::prelude::*;")]
#[cfg_attr(feature = "3d", doc = "# use avian3d::prelude::*;")]
#[cfg_attr(
    feature = "2d",
    doc = "use avian2d::parry::query::DefaultQueryDispatcher;"
)]
#[cfg_attr(
    feature = "3d",
    doc = "use avian3d::parry::query::DefaultQueryDispatcher;"
)]
/// use bevy::prelude::*;
///
/// fn main() {
///     App::new()
///         .add_plugins((DefaultPlugins, PhysicsPlugins::default()))
///         // A real application would chain its own dispatcher in front, e.g.
///         // `MyVoxelDispatcher.chain(DefaultQueryDispatcher)`; installing the
///         // default explicitly is shown here for brevity.
///         .insert_resource(QueryDispatcher::new(Box::new(DefaultQueryDispatcher)))
///         .run();
/// }
/// ```
#[derive(Resource, Default)]
pub struct QueryDispatcher(
    #[cfg(any(feature = "parry-f32", feature = "parry-f64"))]
    Option<Box<dyn PersistentQueryDispatcher<(), ()>>>,
);

#[cfg(any(feature = "parry-f32", feature = "parry-f64"))]
impl QueryDispatcher {
    /// Creates a resource dispatching queries with `dispatcher`.
    ///
    /// Dispatchers for custom shapes should handle the pairs they recognize and fall
    /// back to [`DefaultQueryDispatcher`] for everything else, typically by chaining with
    /// [`QueryDispatcher::chain`](parry::query::QueryDispatcher::chain) *before* boxing.
    pub fn new(dispatcher: Box<dyn PersistentQueryDispatcher<(), ()>>) -> Self {
        Self(Some(dispatcher))
    }
}

#[cfg(any(feature = "parry-f32", feature = "parry-f64"))]
macro_rules! dispatch_query {
    ($query_dispatcher:expr, $method:ident, $($arg:expr),+) => {
        match $query_dispatcher.0 {
            Some(ref dispatcher) => dispatcher.$method($($arg),+),
            None => DefaultQueryDispatcher.$method($($arg),+),
        }
    };
}

#[cfg(any(feature = "parry-f32", feature = "parry-f64"))]
impl QueryDispatcher {
    /// Computes one pair of contact points between two shapes, like
    /// [`parry::query::contact`](fn@parry::query::contact) but routed through the installed dispatcher.
    ///
    /// Returns `None` if the shapes are separated by a distance greater than
    /// `prediction`, and [`Unsupported`] if the dispatcher doesn't handle this shape
    /// pair. The contact is expressed in world space.
    pub fn contact(
        &self,
        pos1: &Pose,
        g1: &dyn Shape,
        pos2: &Pose,
        g2: &dyn Shape,
        prediction: Real,
    ) -> Result<Option<Contact>, Unsupported> {
        let pos12 = pos1.inv_mul(pos2);
        let mut result = dispatch_query!(self, contact, &pos12, g1, g2, prediction);
        if let Ok(Some(contact)) = &mut result {
            contact.transform_by_mut(pos1, pos2);
        }
        result
    }

    /// Computes all [`ContactManifold`]s between two shapes, exploiting spatial and
    /// temporal coherence: passing the `manifolds` and `workspace` produced by the
    /// previous call for the same pair lets the underlying algorithms update contacts
    /// incrementally instead of recomputing them from scratch.
    pub fn contact_manifolds(
        &self,
        pos12: &Pose,
        g1: &dyn Shape,
        g2: &dyn Shape,
        prediction: Real,
        manifolds: &mut Vec<ContactManifold<(), ()>>,
        workspace: &mut Option<ContactManifoldsWorkspace>,
    ) -> Result<(), Unsupported> {
        dispatch_query!(
            self,
            contact_manifolds,
            &pos12,
            g1,
            g2,
            prediction,
            manifolds,
            workspace
        )
    }

    /// Computes the single [`ContactManifold`] between two convex shapes, in the raw
    /// [`PersistentQueryDispatcher::contact_manifold_convex_convex`] shape (`pos12` is
    /// the pose of the second shape relative to the first).
    pub fn contact_manifold_convex_convex(
        &self,
        pos12: &Pose,
        g1: &dyn Shape,
        g2: &dyn Shape,
        normal_constraints1: Option<&dyn NormalConstraints>,
        normal_constraints2: Option<&dyn NormalConstraints>,
        prediction: Real,
        manifold: &mut ContactManifold<(), ()>,
    ) -> Result<(), Unsupported> {
        dispatch_query!(
            self,
            contact_manifold_convex_convex,
            &pos12,
            g1,
            g2,
            normal_constraints1,
            normal_constraints2,
            prediction,
            manifold
        )
    }

    /// Computes the closest points between two shapes, like
    /// [`parry::query::closest_points`](fn@parry::query::closest_points) but routed through the installed dispatcher.
    ///
    /// Returns [`ClosestPoints::Disjoint`] if the shapes are separated by a distance
    /// greater than `max_dist`. The points are expressed in world space.
    pub fn closest_points(
        &self,
        pos1: &Pose,
        g1: &dyn Shape,
        pos2: &Pose,
        g2: &dyn Shape,
        max_dist: Real,
    ) -> Result<ClosestPoints, Unsupported> {
        let pos12 = pos1.inv_mul(pos2);
        dispatch_query!(self, closest_points, &pos12, g1, g2, max_dist)
            .map(|res| res.transform_by(pos1, pos2))
    }

    /// Computes the minimum distance separating two shapes, like
    /// [`parry::query::distance`] but routed through the installed dispatcher.
    ///
    /// Returns `0.0` if the shapes are touching or penetrating.
    pub fn distance(
        &self,
        pos1: &Pose,
        g1: &dyn Shape,
        pos2: &Pose,
        g2: &dyn Shape,
    ) -> Result<Real, Unsupported> {
        let pos12 = pos1.inv_mul(pos2);
        dispatch_query!(self, distance, &pos12, g1, g2)
    }

    /// Tests whether two shapes are intersecting, like
    /// [`parry::query::intersection_test`] but routed through the installed dispatcher.
    pub fn intersection_test(
        &self,
        pos1: &Pose,
        g1: &dyn Shape,
        pos2: &Pose,
        g2: &dyn Shape,
    ) -> Result<bool, Unsupported> {
        let pos12 = pos1.inv_mul(pos2);
        dispatch_query!(self, intersection_test, &pos12, g1, g2)
    }
    /// Casts two shapes moving with the given linear velocities against each other,
    /// like [`parry::query::cast_shapes`] but routed through the installed dispatcher.
    ///
    /// Returns the first [`ShapeCastHit`] within `options.max_time_of_impact`, if any.
    pub fn cast_shapes(
        &self,
        pos1: &Pose,
        vel1: Vector,
        g1: &dyn Shape,
        pos2: &Pose,
        vel2: Vector,
        g2: &dyn Shape,
        options: ShapeCastOptions,
    ) -> Result<Option<ShapeCastHit>, Unsupported> {
        let pos12 = pos1.inv_mul(pos2);
        let vel12 = pos1.rotation.inverse() * (vel2 - vel1);
        dispatch_query!(self, cast_shapes, &pos12, vel12, g1, g2, options)
    }

    /// Casts two shapes under arbitrary rigid motions (translation and rotation)
    /// against each other, like [`parry::query::cast_shapes_nonlinear`] but routed
    /// through the installed dispatcher.
    pub fn cast_shapes_nonlinear(
        &self,
        motion1: &NonlinearRigidMotion,
        g1: &dyn Shape,
        motion2: &NonlinearRigidMotion,
        g2: &dyn Shape,
        start_time: Real,
        end_time: Real,
        stop_at_penetration: bool,
    ) -> Result<Option<ShapeCastHit>, Unsupported> {
        dispatch_query!(
            self,
            cast_shapes_nonlinear,
            motion1,
            g1,
            motion2,
            g2,
            start_time,
            end_time,
            stop_at_penetration
        )
    }
}
