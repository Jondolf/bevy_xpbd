use crate::prelude::*;
#[cfg(all(feature = "3d", feature = "lazy-collider"))]
use bevy::utils::HashMap;
use bevy::{
    ecs::entity::{EntityMapper, MapEntities},
    prelude::*,
    utils::HashSet,
};

/// The default [`Collider`] that uses Parry.
#[cfg(all(
    feature = "default-collider",
    any(feature = "parry-f32", feature = "parry-f64")
))]
mod parry;
#[cfg(all(
    feature = "default-collider",
    any(feature = "parry-f32", feature = "parry-f64")
))]
pub use parry::*;

/// A trait for creating colliders from other types.
pub trait IntoCollider<C: AnyCollider> {
    /// Creates a collider from `self`.
    fn collider(&self) -> C;
}

/// A trait that generalizes over colliders. Implementing this trait
/// allows colliders to be used with the physics engine.
pub trait AnyCollider: Component {
    /// Computes the [Axis-Aligned Bounding Box](ColliderAabb) of the collider
    /// with the given position and rotation.
    #[cfg_attr(
        feature = "2d",
        doc = "\n\nThe rotation is counterclockwise and in radians."
    )]
    fn aabb(&self, position: Vector, rotation: impl Into<Rotation>) -> ColliderAabb;

    /// Computes the swept [Axis-Aligned Bounding Box](ColliderAabb) of the collider.
    /// This corresponds to the space the shape would occupy if it moved from the given
    /// start position to the given end position.
    #[cfg_attr(
        feature = "2d",
        doc = "\n\nThe rotation is counterclockwise and in radians."
    )]
    fn swept_aabb(
        &self,
        start_position: Vector,
        start_rotation: impl Into<Rotation>,
        end_position: Vector,
        end_rotation: impl Into<Rotation>,
    ) -> ColliderAabb {
        self.aabb(start_position, start_rotation)
            .merged(self.aabb(end_position, end_rotation))
    }

    /// Computes the collider's mass properties based on its shape and a given density.
    fn mass_properties(&self, density: Scalar) -> ColliderMassProperties;

    /// Computes all [`ContactManifold`]s between two colliders.
    ///
    /// Returns an empty vector if the colliders are separated by a distance greater than `prediction_distance`
    /// or if the given shapes are invalid.
    fn contact_manifolds(
        &self,
        other: &Self,
        position1: Vector,
        rotation1: impl Into<Rotation>,
        position2: Vector,
        rotation2: impl Into<Rotation>,
        prediction_distance: Scalar,
    ) -> Vec<ContactManifold>;
}

/// A trait for colliders that support scaling.
pub trait ScalableCollider: AnyCollider {
    /// Returns the global scaling factor of the collider.
    fn scale(&self) -> Vector;

    /// Sets the global scaling factor of the collider.
    ///
    /// If the scaling factor is not uniform and the resulting scaled shape
    /// can not be represented exactly, the given `detail` is used for an approximation.
    fn set_scale(&mut self, scale: Vector, detail: u32);

    /// Scales the collider by the given scaling factor.
    ///
    /// If the scaling factor is not uniform and the resulting scaled shape
    /// can not be represented exactly, the given `detail` is used for an approximation.
    fn scale_by(&mut self, factor: Vector, detail: u32) {
        self.set_scale(factor * self.scale(), detail)
    }
}

/// A component that will automatically generate a [`Collider`] at runtime.
/// The type of the generated collider can be specified using [`ColliderConstructor`].
/// This supports computing the shape dynamically from the mesh.
///
/// Since [`Collider`] is not [`Reflect`], you can use this type to statically statically
/// specify a collider's shape instead.
///
/// This component will never override a pre-existing [`Collider`] component on the same entity.
///
/// ## See also
///
/// For inserting colliders on an entity's descendants, use [`LazyColliderHierarchy`].
///
/// ## Example
///
/// ```
/// use bevy::prelude::*;
/// use bevy_xpbd_3d::prelude::*;
///
/// fn setup(mut commands: Commands, mut assets: ResMut<AssetServer>, mut meshes: Assets<Mesh>) {
///     // Spawn a cube with a convex hull collider generated from the mesh
///     commands.spawn((
///         LazyCollider(ColliderConstructor::ConvexHullFromMesh),
///         PbrBundle {
///             mesh: meshes.add(Mesh::from(Cuboid::default())),
///             ..default()
///         },
///     ));
/// }
/// ```
#[cfg(feature = "lazy-collider")]
#[derive(Component, Clone, Debug, Default, Deref, DerefMut, PartialEq, Reflect)]
#[reflect(Debug, PartialEq, Component)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serialize", reflect(Serialize, Deserialize))]
pub struct LazyCollider(pub ColliderConstructor);

/// A component that will automatically generate [`Collider`]s on its descendants at runtime.
/// The type of the generated collider can be specified using [`ColliderConstructor`].
/// This supports computing the shape dynamically from the mesh, in which case only the descendants
/// with a [`Mesh`] will have colliders generated.
///
/// In contrast to [`LazyCollider`], this component will *not* generate a collider on its own entity.
///
/// If this component is used on a scene, such as one spawned by a [`SceneBundle`], it will
/// wait until the scene is loaded before generating colliders.
///
/// The exact configuration for each descendant can be specified using [`LazyColliderHierarchyData`].
///
/// This component will only override a pre-existing [`Collider`] component on a descendant entity
/// when it has been explicitly mentioned in the `meshes_by_name`.
///
/// ## See also
///
/// For inserting colliders on the same entity, use [`LazyCollider`].
///
/// ## Caveats
///
/// When a component has multiple ancestors with [`LazyColliderHierarchy`], the insertion order is undefined.
///
/// ## Example
///
/// ```
/// use bevy::prelude::*;
/// use bevy_xpbd_3d::prelude::*;
///
/// fn setup(mut commands: Commands, mut assets: ResMut<AssetServer>) {
///     let scene = assets.load("my_model.gltf#Scene0");
///
///     // Spawn the scene and automatically generate triangle mesh colliders
///     commands.spawn((
///         SceneBundle { scene: scene.clone(), ..default() },
///         LazyColliderHierarchy::new(Some(ColliderConstructor::TrimeshFromMesh)),
///     ));
///
///     // Specify configuration for specific meshes by name
///     commands.spawn((
///         SceneBundle { scene: scene.clone(), ..default() },
///         LazyColliderHierarchy::new(Some(ColliderConstructor::TrimeshFromMesh))
///             .with_shape_for_name("Tree", ColliderConstructor::ConvexHullFromMesh)
///             .with_layers_for_name("Tree", CollisionLayers::from_bits(0b0010, 0b1111))
///             .with_density_for_name("Tree", 2.5),
///     ));
///
///     // Only generate colliders for specific meshes by name
///     commands.spawn((
///         SceneBundle { scene: scene.clone(), ..default() },
///         LazyColliderHierarchy::new(None)
///             .with_shape_for_name("Tree", ColliderConstructor::ConvexHullFromMesh),
///     ));
///
///     // Generate colliders for everything except specific meshes by name
///     commands.spawn((
///         SceneBundle { scene, ..default() },
///         LazyColliderHierarchy::new(Some(ColliderConstructor::TrimeshFromMeshWithConfig(
///             TriMeshFlags::MERGE_DUPLICATE_VERTICES
///         )))
///         .without_shape_with_name("Tree"),
///     ));
/// }
/// ```
#[cfg(feature = "lazy-collider")]
#[derive(Component, Clone, Debug, Default, PartialEq, Reflect)]
#[reflect(Debug, Component, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serialize", reflect(Serialize, Deserialize))]
pub struct LazyColliderHierarchy {
    /// The default collider type used for each mesh that isn't included in [`meshes_by_name`](#structfield.meshes_by_name).
    /// If `None`, all meshes except the ones in [`meshes_by_name`](#structfield.meshes_by_name) will be skipped.
    pub default_shape: Option<ColliderConstructor>,
    /// Specifies data like the collider type and [`CollisionLayers`] for meshes by name.
    /// Entries with a `None` value will be skipped.
    /// For the meshes not found in this `HashMap`, [`default_shape`](#structfield.default_shape)
    /// and all collision layers will be used instead.
    pub meshes_by_name: HashMap<String, Option<LazyColliderHierarchyData>>,
}

#[cfg(feature = "lazy-collider")]
impl LazyColliderHierarchy {
    /// Creates a new [`LazyColliderHierarchy`] with the default collider type used for
    /// meshes set to the given `default_shape`.
    ///
    /// If the given collider type is `None`, all meshes except the ones in
    /// [`meshes_by_name`](#structfield.meshes_by_name) will be skipped.
    /// You can add named shapes using [`with_shape_for_name`](Self::with_shape_for_name).
    pub fn new(default_shape: impl Into<Option<ColliderConstructor>>) -> Self {
        Self {
            default_shape: default_shape.into(),
            meshes_by_name: default(),
        }
    }

    /// Specifies the collider type used for a mesh with the given `name`.
    pub fn with_shape_for_name(mut self, name: &str, shape: ColliderConstructor) -> Self {
        if let Some(Some(data)) = self.meshes_by_name.get_mut(name) {
            data.shape = shape;
        } else {
            self.meshes_by_name.insert(
                name.to_string(),
                Some(LazyColliderHierarchyData { shape, ..default() }),
            );
        }
        self
    }

    /// Specifies the [`CollisionLayers`] used for a mesh with the given `name`.
    pub fn with_layers_for_name(mut self, name: &str, layers: CollisionLayers) -> Self {
        if let Some(Some(data)) = self.meshes_by_name.get_mut(name) {
            data.layers = layers;
        } else {
            self.meshes_by_name.insert(
                name.to_string(),
                Some(LazyColliderHierarchyData {
                    layers,
                    ..default()
                }),
            );
        }
        self
    }

    /// Specifies the [`ColliderDensity`] used for a mesh with the given `name`.
    pub fn with_density_for_name(mut self, name: &str, density: Scalar) -> Self {
        if let Some(Some(data)) = self.meshes_by_name.get_mut(name) {
            data.density = density;
        } else {
            self.meshes_by_name.insert(
                name.to_string(),
                Some(LazyColliderHierarchyData {
                    density,
                    ..default()
                }),
            );
        }
        self
    }

    /// Sets collider for the mesh associated with the given `name` to `None`, skipping
    /// collider generation for it.
    pub fn without_shape_with_name(mut self, name: &str) -> Self {
        self.meshes_by_name.insert(name.to_string(), None);
        self
    }
}

/// Configuration for a specific collider generated from a scene using [`LazyColliderHierarchy`].
#[cfg(all(feature = "3d", feature = "lazy-collider"))]
#[derive(Clone, Debug, PartialEq, Reflect)]
#[reflect(Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serialize", reflect(Serialize, Deserialize))]
#[cfg_attr(all(feature = "3d", feature = "collider-from-mesh"), reflect(Default))]
pub struct LazyColliderHierarchyData {
    /// The type of collider generated for the mesh.
    pub shape: ColliderConstructor,
    /// The [`CollisionLayers`] used for this collider.
    pub layers: CollisionLayers,
    /// The [`ColliderDensity`] used for this collider.
    pub density: Scalar,
}

#[cfg(all(
    feature = "3d",
    feature = "lazy-collider",
    feature = "collider-from-mesh"
))]
impl Default for LazyColliderHierarchyData {
    fn default() -> Self {
        Self {
            shape: ColliderConstructor::TrimeshFromMesh,
            layers: CollisionLayers::default(),
            density: 1.0,
        }
    }
}

/// Determines how a [`Collider`] is generated from a `Mesh`.
///
/// Colliders can be created from meshes with the following components and methods:
///
/// - [`LazyCollider`] (requires `lazy-collider` features)
/// - [`LazyColliderHierarchy`] (requires `lazy-collider` features)
/// - [`Collider::trimesh_from_mesh`]
/// - [`Collider::convex_hull_from_mesh`]
/// - [`Collider::convex_decomposition_from_mesh`]
#[derive(Clone, Debug, PartialEq, Reflect)]
#[reflect(Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serialize", reflect(Serialize, Deserialize))]
#[cfg_attr(all(feature = "3d", feature = "collider-from-mesh"), derive(Default))]
#[cfg_attr(all(feature = "3d", feature = "collider-from-mesh"), reflect(Default))]
pub enum ColliderConstructor {
    /// Constructs a collider with [`Collider::circle`].
    #[cfg(feature = "2d")]
    Circle {
        #[allow(missing_docs)]
        radius: Scalar,
    },
    /// Constructs a collider with [`Collider::sphere`].
    #[cfg(feature = "3d")]
    Sphere {
        #[allow(missing_docs)]
        radius: Scalar,
    },
    /// Constructs a collider with [`Collider::ellipse`].
    #[cfg(feature = "2d")]
    Ellipse {
        #[allow(missing_docs)]
        half_width: Scalar,
        #[allow(missing_docs)]
        half_height: Scalar,
    },
    /// Constructs a collider with [`Collider::rectangle`].
    #[cfg(feature = "2d")]
    Rectangle {
        #[allow(missing_docs)]
        x_length: Scalar,
        #[allow(missing_docs)]
        y_length: Scalar,
    },
    /// Constructs a collider with [`Collider::cuboid`].
    #[cfg(feature = "3d")]
    Cuboid {
        #[allow(missing_docs)]
        x_length: Scalar,
        #[allow(missing_docs)]
        y_length: Scalar,
        #[allow(missing_docs)]
        z_length: Scalar,
    },
    /// Constructs a collider with [`Collider::round_rectangle`].
    #[cfg(feature = "2d")]
    RoundRectangle {
        #[allow(missing_docs)]
        x_length: Scalar,
        #[allow(missing_docs)]
        y_length: Scalar,
        #[allow(missing_docs)]
        border_radius: Scalar,
    },
    /// Constructs a collider with [`Collider::round_cuboid`].
    #[cfg(feature = "3d")]
    RoundCuboid {
        #[allow(missing_docs)]
        x_length: Scalar,
        #[allow(missing_docs)]
        y_length: Scalar,
        #[allow(missing_docs)]
        z_length: Scalar,
        #[allow(missing_docs)]
        border_radius: Scalar,
    },
    /// Constructs a collider with [`Collider::cylinder`].
    #[cfg(feature = "3d")]
    Cylinder {
        #[allow(missing_docs)]
        height: Scalar,
        #[allow(missing_docs)]
        radius: Scalar,
    },
    /// Constructs a collider with [`Collider::cone`].
    #[cfg(feature = "3d")]
    Cone {
        #[allow(missing_docs)]
        height: Scalar,
        #[allow(missing_docs)]
        radius: Scalar,
    },
    /// Constructs a collider with [`Collider::capsule`].
    Capsule {
        #[allow(missing_docs)]
        height: Scalar,
        #[allow(missing_docs)]
        radius: Scalar,
    },
    /// Constructs a collider with [`Collider::capsule_endpoints`].
    CapsuleEndpoints {
        #[allow(missing_docs)]
        a: Vector,
        #[allow(missing_docs)]
        b: Vector,
        #[allow(missing_docs)]
        radius: Scalar,
    },
    /// Constructs a collider with [`Collider::halfspace`].
    Halfspace {
        #[allow(missing_docs)]
        outward_normal: Vector,
    },
    /// Constructs a collider with [`Collider::segment`].
    Segment {
        #[allow(missing_docs)]
        a: Vector,
        #[allow(missing_docs)]
        b: Vector,
    },
    /// Constructs a collider with [`Collider::triangle`].
    Triangle {
        #[allow(missing_docs)]
        a: Vector,
        #[allow(missing_docs)]
        b: Vector,
        #[allow(missing_docs)]
        c: Vector,
    },
    /// Constructs a collider with [`Collider::regular_polygon`].
    #[cfg(feature = "2d")]
    RegularPolygon {
        #[allow(missing_docs)]
        circumradius: f32,
        #[allow(missing_docs)]
        size: usize,
    },
    /// Constructs a collider with [`Collider::polyline`].
    Polyline {
        #[allow(missing_docs)]
        vertices: Vec<Vector>,
        #[allow(missing_docs)]
        indices: Option<Vec<[u32; 2]>>,
    },
    /// Constructs a collider with [`Collider::trimesh`].
    Trimesh {
        #[allow(missing_docs)]
        vertices: Vec<Vector>,
        #[allow(missing_docs)]
        indices: Vec<[u32; 3]>,
    },
    /// Constructs a collider with [`Collider::trimesh_with_config`].
    TrimeshWithConfig {
        #[allow(missing_docs)]
        vertices: Vec<Vector>,
        #[allow(missing_docs)]
        indices: Vec<[u32; 3]>,
        #[allow(missing_docs)]
        flags: TriMeshFlags,
    },
    /// Constructs a collider with [`Collider::convex_decomposition`].
    #[cfg(feature = "2d")]
    ConvexDecomposition {
        #[allow(missing_docs)]
        vertices: Vec<Vector>,
        #[allow(missing_docs)]
        indices: Vec<[u32; 2]>,
    },
    /// Constructs a collider with [`Collider::convex_decomposition`].
    #[cfg(feature = "3d")]
    ConvexDecomposition {
        #[allow(missing_docs)]
        vertices: Vec<Vector>,
        #[allow(missing_docs)]
        indices: Vec<[u32; 3]>,
    },
    /// Constructs a collider with [`Collider::convex_decomposition_with_config`].
    #[cfg(feature = "2d")]
    ConvexDecompositionWithConfig {
        #[allow(missing_docs)]
        vertices: Vec<Vector>,
        #[allow(missing_docs)]
        indices: Vec<[u32; 2]>,
        #[allow(missing_docs)]
        params: VhacdParameters,
    },
    /// Constructs a collider with [`Collider::convex_decomposition_with_config`].
    #[cfg(feature = "3d")]
    ConvexDecompositionWithConfig {
        #[allow(missing_docs)]
        vertices: Vec<Vector>,
        #[allow(missing_docs)]
        indices: Vec<[u32; 3]>,
        #[allow(missing_docs)]
        params: VhacdParameters,
    },
    /// Constructs a collider with [`Collider::convex_hull`].
    #[cfg(feature = "2d")]
    ConvexHull {
        #[allow(missing_docs)]
        points: Vec<Vector>,
    },
    /// Constructs a collider with [`Collider::convex_hull`].
    #[cfg(feature = "3d")]
    ConvexHull {
        #[allow(missing_docs)]
        points: Vec<Vector>,
    },
    /// Constructs a collider with [`Collider::heightfield`].
    #[cfg(feature = "2d")]
    Heightfield {
        #[allow(missing_docs)]
        heights: Vec<Scalar>,
        #[allow(missing_docs)]
        scale: Vector,
    },
    /// Constructs a collider with [`Collider::heightfield`].
    #[cfg(feature = "3d")]
    Heightfield {
        #[allow(missing_docs)]
        heights: Vec<Vec<Scalar>>,
        #[allow(missing_docs)]
        scale: Vector,
    },
    /// Constructs a collider with [`Collider::trimesh_from_mesh`].
    #[cfg(all(feature = "3d", feature = "collider-from-mesh"))]
    #[default]
    TrimeshFromMesh,
    /// Constructs a collider with [`Collider::trimesh_from_mesh_with_config`].
    #[cfg(all(
        feature = "3d",
        feature = "collider-from-mesh",
        feature = "default-collider"
    ))]
    TrimeshFromMeshWithConfig(TriMeshFlags),
    /// Constructs a collider with [`Collider::convex_decomposition_from_mesh`].
    #[cfg(all(feature = "3d", feature = "collider-from-mesh"))]
    ConvexDecompositionFromMesh,
    /// Constructs a collider with [`Collider::convex_decomposition_from_mesh_with_config`].
    #[cfg(all(
        feature = "3d",
        feature = "collider-from-mesh",
        feature = "default-collider"
    ))]
    ConvexDecompositionFromMeshWithConfig(VhacdParameters),
    /// Constructs a collider with [`Collider::convex_hull_from_mesh`].
    #[cfg(all(feature = "3d", feature = "collider-from-mesh"))]
    ConvexHullFromMesh,
}
impl ColliderConstructor {
    /// Returns `true` if the collider type requires a mesh to be generated.
    pub fn requires_mesh(&self) -> bool {
        #[cfg(all(feature = "3d", feature = "collider-from-mesh"))]
        {
            matches!(
                self,
                Self::TrimeshFromMesh
                    | Self::TrimeshFromMeshWithConfig(_)
                    | Self::ConvexDecompositionFromMesh
                    | Self::ConvexDecompositionFromMeshWithConfig(_)
                    | Self::ConvexHullFromMesh
            )
        }
        #[cfg(not(all(feature = "3d", feature = "collider-from-mesh")))]
        {
            false
        }
    }
}

/// A component that stores the `Entity` ID of the [`RigidBody`] that a [`Collider`] is attached to.
///
/// If the collider is a child of a rigid body, this points to the body's `Entity` ID.
/// If the [`Collider`] component is instead on the same entity as the [`RigidBody`] component,
/// this points to the collider's own `Entity` ID.
///
/// This component is added and updated automatically based on entity hierarchies and should not
/// be modified directly.
///
/// ## Example
///
/// ```
/// use bevy::prelude::*;
#[cfg_attr(feature = "2d", doc = "use bevy_xpbd_2d::prelude::*;")]
#[cfg_attr(feature = "3d", doc = "use bevy_xpbd_3d::prelude::*;")]
///
/// fn setup(mut commands: Commands) {
///     // Spawn a rigid body with one collider on the same entity and two as children.
///     // Each entity will have a ColliderParent component that has the same rigid body entity.
///     commands
///         .spawn((RigidBody::Dynamic, Collider::ball(0.5)))
///         .with_children(|children| {
///             children.spawn((Collider::ball(0.5), Transform::from_xyz(2.0, 0.0, 0.0)));
///             children.spawn((Collider::ball(0.5), Transform::from_xyz(-2.0, 0.0, 0.0)));
///         });
/// }
/// ```
#[derive(Reflect, Clone, Copy, Component, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ColliderParent(pub(crate) Entity);

impl ColliderParent {
    /// Gets the `Entity` ID of the [`RigidBody`] that this [`Collider`] is attached to.
    pub const fn get(&self) -> Entity {
        self.0
    }
}

impl MapEntities for ColliderParent {
    fn map_entities<M: EntityMapper>(&mut self, entity_mapper: &mut M) {
        self.0 = entity_mapper.map_entity(self.0)
    }
}

/// The transform of a collider relative to the rigid body it's attached to.
/// This is in the local space of the body, not the collider itself.
///
/// This is used for computing things like contact positions and a body's center of mass
/// without having to traverse deeply nested hierarchies. It's updated automatically,
/// so you shouldn't modify it manually.
#[derive(Reflect, Clone, Copy, Component, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ColliderTransform {
    /// The translation of a collider in a rigid body's frame of reference.
    pub translation: Vector,
    /// The rotation of a collider in a rigid body's frame of reference.
    pub rotation: Rotation,
    /// The global scale of a collider. Equivalent to the `GlobalTransform` scale.
    pub scale: Vector,
}

impl ColliderTransform {
    /// Transforms a given point by applying the translation, rotation and scale of
    /// this [`ColliderTransform`].
    pub fn transform_point(&self, mut point: Vector) -> Vector {
        point *= self.scale;
        point = self.rotation.rotate(point);
        point += self.translation;
        point
    }
}

impl Default for ColliderTransform {
    fn default() -> Self {
        Self {
            translation: Vector::ZERO,
            rotation: Rotation::default(),
            scale: Vector::ONE,
        }
    }
}

impl From<Transform> for ColliderTransform {
    fn from(value: Transform) -> Self {
        Self {
            #[cfg(feature = "2d")]
            translation: value.translation.truncate().adjust_precision(),
            #[cfg(feature = "3d")]
            translation: value.translation.adjust_precision(),
            rotation: Rotation::from(value.rotation.adjust_precision()),
            #[cfg(feature = "2d")]
            scale: value.scale.truncate().adjust_precision(),
            #[cfg(feature = "3d")]
            scale: value.scale.adjust_precision(),
        }
    }
}

/// A component that marks a [`Collider`] as a sensor, also known as a trigger.
///
/// Sensor colliders send [collision events](ContactReportingPlugin#collision-events) and register intersections,
/// but allow other bodies to pass through them. This is often used to detect when something enters
/// or leaves an area or is intersecting some shape.
///
/// ## Example
///
/// ```
/// use bevy::prelude::*;
#[cfg_attr(feature = "2d", doc = "use bevy_xpbd_2d::prelude::*;")]
#[cfg_attr(feature = "3d", doc = "use bevy_xpbd_3d::prelude::*;")]
///
/// fn setup(mut commands: Commands) {
///     // Spawn a static body with a sensor collider.
///     // Other bodies will pass through, but it will still send collision events.
///     commands.spawn((RigidBody::Static, Collider::ball(0.5), Sensor));
/// }
/// ```
#[doc(alias = "Trigger")]
#[derive(Reflect, Clone, Component, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[reflect(Component)]
pub struct Sensor;

/// The Axis-Aligned Bounding Box of a [collider](Collider).
#[derive(Clone, Copy, Component, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ColliderAabb {
    /// The minimum point of the AABB.
    pub min: Vector,
    /// The maximum point of thr AABB.
    pub max: Vector,
}

impl ColliderAabb {
    /// Creates a new [`ColliderAabb`] from the given `center` and `half_size`.
    pub fn new(center: Vector, half_size: Vector) -> Self {
        Self {
            min: center - half_size,
            max: center + half_size,
        }
    }

    /// Creates a new [`ColliderAabb`] from its minimum and maximum points.
    pub fn from_min_max(min: Vector, max: Vector) -> Self {
        Self { min, max }
    }

    /// Creates a new [`ColliderAabb`] from a given [`SharedShape`].
    #[cfg(all(
        feature = "default-collider",
        any(feature = "parry-f32", feature = "parry-f64")
    ))]
    pub fn from_shape(shape: &crate::parry::shape::SharedShape) -> Self {
        let aabb = shape.compute_local_aabb();
        Self {
            min: aabb.mins.into(),
            max: aabb.maxs.into(),
        }
    }

    /// Computes the center of the AABB,
    pub fn center(self) -> Vector {
        (self.min + self.max) / 2.0
    }

    /// Computes the size of the AABB.
    pub fn size(self) -> Vector {
        self.max - self.min
    }

    /// Merges this AABB with another one.
    pub fn merged(self, other: Self) -> Self {
        ColliderAabb {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    /// Checks if `self` intersects with `other`.
    #[inline(always)]
    #[cfg(feature = "2d")]
    pub fn intersects(&self, other: &Self) -> bool {
        let x_overlaps = self.min.x <= other.max.x && self.max.x >= other.min.x;
        let y_overlaps = self.min.y <= other.max.y && self.max.y >= other.min.y;
        x_overlaps && y_overlaps
    }

    /// Checks if `self` intersects with `other`.
    #[inline(always)]
    #[cfg(feature = "3d")]
    pub fn intersects(&self, other: &Self) -> bool {
        let x_overlaps = self.min.x <= other.max.x && self.max.x >= other.min.x;
        let y_overlaps = self.min.y <= other.max.y && self.max.y >= other.min.y;
        let z_overlaps = self.min.z <= other.max.z && self.max.z >= other.min.z;
        x_overlaps && y_overlaps && z_overlaps
    }
}

impl Default for ColliderAabb {
    fn default() -> Self {
        ColliderAabb {
            min: Vector::INFINITY,
            max: Vector::NEG_INFINITY,
        }
    }
}

/// A component that stores the entities that are colliding with an entity.
///
/// This component is automatically added for all entities with a [`Collider`],
/// but it will only be filled if the [`ContactReportingPlugin`] is enabled (by default, it is).
///
/// ## Example
///
/// ```
/// use bevy::prelude::*;
#[cfg_attr(feature = "2d", doc = "use bevy_xpbd_2d::prelude::*;")]
#[cfg_attr(feature = "3d", doc = "use bevy_xpbd_3d::prelude::*;")]
///
/// fn my_system(query: Query<(Entity, &CollidingEntities)>) {
///     for (entity, colliding_entities) in &query {
///         println!(
///             "{:?} is colliding with the following entities: {:?}",
///             entity,
///             colliding_entities
///         );
///     }
/// }
/// ```
#[derive(Reflect, Clone, Component, Debug, Default, Deref, DerefMut, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[reflect(Component)]
pub struct CollidingEntities(pub HashSet<Entity>);

impl MapEntities for CollidingEntities {
    fn map_entities<M: EntityMapper>(&mut self, entity_mapper: &mut M) {
        self.0 = self
            .0
            .clone()
            .into_iter()
            .map(|e| entity_mapper.map_entity(e))
            .collect()
    }
}

#[cfg(all(feature = "lazy-collider", test))]
mod tests {
    use super::*;
    use bevy::{ecs::query::QueryData, scene::ScenePlugin};

    #[test]
    fn lazy_collider_requires_no_mesh_on_primitive() {
        let mut app = create_test_app();

        let entity = app
            .world
            .spawn(LazyCollider(PRIMITIVE_COLLIDER.clone()))
            .id();

        app.update();

        assert!(app.query_ok::<&Collider>(entity));
        assert!(app.query_err::<&LazyCollider>(entity));
    }

    #[test]
    #[should_panic]
    fn lazy_collider_requires_mesh_on_computed() {
        let mut app = create_test_app();

        app.world.spawn(LazyCollider(COMPUTED_COLLIDER.clone()));

        app.update();
    }

    #[test]
    fn lazy_collider_converts_mesh_on_computed() {
        let mut app = create_test_app();

        let mesh_handle = app.add_mesh();
        let entity = app
            .world
            .spawn((LazyCollider(COMPUTED_COLLIDER.clone()), mesh_handle))
            .id();

        app.update();

        assert!(app.query_ok::<&Collider>(entity));
        assert!(app.query_ok::<&Handle<Mesh>>(entity));
        assert!(app.query_err::<&LazyCollider>(entity));
    }

    #[test]
    fn lazy_collider_hierarchy_does_nothing_on_self_with_primitive() {
        let mut app = create_test_app();

        let entity = app
            .world
            .spawn(LazyColliderHierarchy::new(PRIMITIVE_COLLIDER.clone()))
            .id();

        app.update();

        assert!(app.query_err::<&LazyColliderHierarchy>(entity));
        assert!(app.query_err::<&Collider>(entity));
    }

    #[test]
    fn lazy_collider_hierarchy_does_nothing_on_self_with_computed() {
        let mut app = create_test_app();

        let mesh_handle = app.add_mesh();
        let entity = app
            .world
            .spawn((
                LazyColliderHierarchy::new(COMPUTED_COLLIDER.clone()),
                mesh_handle,
            ))
            .id();

        app.update();

        assert!(app.query_ok::<&Handle<Mesh>>(entity));
        assert!(app.query_err::<&LazyColliderHierarchy>(entity));
        assert!(app.query_err::<&Collider>(entity));
    }

    #[test]
    fn lazy_collider_hierarchy_does_not_require_mesh_on_self_with_computed() {
        let mut app = create_test_app();

        let entity = app
            .world
            .spawn(LazyColliderHierarchy::new(COMPUTED_COLLIDER.clone()))
            .id();

        app.update();

        assert!(app.query_err::<&Collider>(entity));
        assert!(app.query_err::<&LazyColliderHierarchy>(entity));
    }

    #[test]
    fn lazy_collider_hierarchy_inserts_primitive_colliders_on_all_descendants() {
        let mut app = create_test_app();

        // Hierarchy:
        // - parent
        //   - child1
        //   - child2
        //     - child3

        let parent = app
            .world
            .spawn(LazyColliderHierarchy::new(PRIMITIVE_COLLIDER.clone()))
            .id();
        let child1 = app.world.spawn(()).id();
        let child2 = app.world.spawn(()).id();
        let child3 = app.world.spawn(()).id();

        app.world
            .entity_mut(parent)
            .push_children(&[child1, child2]);
        app.world.entity_mut(child2).push_children(&[child3]);

        app.update();

        // No entities should have LazyColliderHierarchy
        assert!(app.query_err::<&LazyColliderHierarchy>(parent));
        assert!(app.query_err::<&LazyColliderHierarchy>(child1));
        assert!(app.query_err::<&LazyColliderHierarchy>(child2));
        assert!(app.query_err::<&LazyColliderHierarchy>(child3));

        assert!(app.query_err::<&Collider>(parent));
        assert!(app.query_ok::<&Collider>(child1));
        assert!(app.query_ok::<&Collider>(child2));
        assert!(app.query_ok::<&Collider>(child3));
    }

    #[test]
    fn lazy_collider_hierarchy_inserts_computed_colliders_only_on_descendants_with_mesh() {
        let mut app = create_test_app();
        let mesh_handle = app.add_mesh();

        // Hierarchy:
        // - parent
        //   - child1 (no mesh)
        //   - child2 (no mesh)
        //     - child3 (mesh)
        //   - child4 (mesh)
        //     - child5 (no mesh)
        //   - child6 (mesh)
        //   - child7 (mesh)
        //     - child8 (mesh)

        let parent = app
            .world
            .spawn(LazyColliderHierarchy::new(COMPUTED_COLLIDER.clone()))
            .id();
        let child1 = app.world.spawn(()).id();
        let child2 = app.world.spawn(()).id();
        let child3 = app.world.spawn(mesh_handle.clone()).id();
        let child4 = app.world.spawn(mesh_handle.clone()).id();
        let child5 = app.world.spawn(()).id();
        let child6 = app.world.spawn(mesh_handle.clone()).id();
        let child7 = app.world.spawn(mesh_handle.clone()).id();
        let child8 = app.world.spawn(mesh_handle.clone()).id();

        app.world
            .entity_mut(parent)
            .push_children(&[child1, child2, child4, child6, child7]);
        app.world.entity_mut(child2).push_children(&[child3]);
        app.world.entity_mut(child4).push_children(&[child5]);
        app.world.entity_mut(child7).push_children(&[child8]);

        app.update();

        // No entities should have LazyColliderHierarchy
        assert!(app.query_err::<&LazyColliderHierarchy>(parent));
        assert!(app.query_err::<&LazyColliderHierarchy>(child1));
        assert!(app.query_err::<&LazyColliderHierarchy>(child2));
        assert!(app.query_err::<&LazyColliderHierarchy>(child3));
        assert!(app.query_err::<&LazyColliderHierarchy>(child4));
        assert!(app.query_err::<&LazyColliderHierarchy>(child5));
        assert!(app.query_err::<&LazyColliderHierarchy>(child6));
        assert!(app.query_err::<&LazyColliderHierarchy>(child7));
        assert!(app.query_err::<&LazyColliderHierarchy>(child8));

        assert!(app.query_err::<&Collider>(parent));
        assert!(app.query_err::<&Collider>(child1));
        assert!(app.query_err::<&Collider>(child2));
        assert!(app.query_ok::<&Collider>(child3));
        assert!(app.query_ok::<&Collider>(child4));
        assert!(app.query_err::<&Collider>(child5));
        assert!(app.query_ok::<&Collider>(child6));
        assert!(app.query_ok::<&Collider>(child7));
        assert!(app.query_ok::<&Collider>(child8));
    }

    const PRIMITIVE_COLLIDER: ColliderConstructor = ColliderConstructor::Capsule {
        height: 1.0,
        radius: 0.5,
    };

    const COMPUTED_COLLIDER: ColliderConstructor = ColliderConstructor::TrimeshFromMesh;

    fn create_test_app() -> App {
        let mut app = App::new();
        app.add_plugins((
            MinimalPlugins,
            AssetPlugin::default(),
            ScenePlugin,
            PhysicsPlugins::default(),
        ))
        .init_resource::<Assets<Mesh>>();

        app
    }

    trait AppExt {
        fn query_ok<D: QueryData>(&mut self, entity: Entity) -> bool;
        fn query_err<D: QueryData>(&mut self, entity: Entity) -> bool {
            !self.query_ok::<D>(entity)
        }
        fn add_mesh(&mut self) -> Handle<Mesh>;
    }

    impl AppExt for App {
        fn query_ok<D: QueryData>(&mut self, entity: Entity) -> bool {
            let mut query = self.world.query::<D>();
            let component = query.get(&self.world, entity);
            component.is_ok()
        }

        fn add_mesh(&mut self) -> Handle<Mesh> {
            self.world
                .get_resource_mut::<Assets<Mesh>>()
                .unwrap()
                .add(Mesh::from(Cuboid::default()))
        }
    }
}
