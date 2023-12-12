#![allow(clippy::unnecessary_cast)]

use bevy::{
    prelude::*,
    sprite::{MaterialMesh2dBundle, Mesh2dHandle},
    window::PrimaryWindow,
};
use bevy_xpbd_2d::{math::*, prelude::*};
use examples_common_2d::XpbdExamplePlugin;

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, XpbdExamplePlugin))
        .insert_resource(ClearColor(Color::rgb(0.05, 0.05, 0.1)))
        .insert_resource(SubstepCount(50))
        .insert_resource(Gravity(Vector::NEG_Y * 1000.0))
        .add_systems(Startup, setup)
        .add_systems(Update, update_cursor_position)
        .run();
}

#[derive(Component)]
struct FollowMouse;

fn setup(
    mut commands: Commands,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    commands.spawn(Camera2dBundle::default());

    let particle_count = 100;
    let particle_radius = 1.2;
    let particle_mesh: Mesh2dHandle = meshes
        .add(shape::Circle::new(particle_radius as f32).into())
        .into();
    let particle_material = materials.add(ColorMaterial::from(Color::rgb(0.2, 0.7, 0.9)));

    // Spawn kinematic particle that can follow the mouse
    let mut previous_particle = commands
        .spawn((
            RigidBody::Kinematic,
            FollowMouse,
            MaterialMesh2dBundle {
                mesh: particle_mesh.clone(),
                material: particle_material.clone(),
                ..default()
            },
        ))
        .id();

    // Spawn the rest of the particles, connecting each one to the previous one with joints
    for i in 1..particle_count {
        let current_particle = commands
            .spawn((
                RigidBody::Dynamic,
                MassPropertiesBundle::new_computed(&Collider::ball(particle_radius), 1.0),
                MaterialMesh2dBundle {
                    mesh: particle_mesh.clone(),
                    material: particle_material.clone(),
                    transform: Transform::from_xyz(
                        0.0,
                        -i as f32 * (particle_radius as f32 * 2.0 + 1.0),
                        0.0,
                    ),
                    ..default()
                },
            ))
            .id();

        commands.spawn(JointBundle {
            entities: [previous_particle, current_particle].into(),
            joint: RevoluteJoint::new().with_compliance(0.0000001),
            anchors: JointAnchors::from_second(Vector::Y * (particle_radius * 2.0 + 1.0)),
            ..default()
        });

        previous_particle = current_particle;
    }
}

#[derive(Resource)]
pub struct CursorPosition(pub Vec2);

fn update_cursor_position(
    windows: Query<&Window, With<PrimaryWindow>>,
    camera: Query<(&Camera, &GlobalTransform)>,
    mut cursor_pos: ResMut<CursorPosition>,
) {
    let window = windows.single();
    let (camera, camera_transform) = camera.single();

    if let Some(cursor_world_pos) = window
        .cursor_position()
        .and_then(|cursor| camera.viewport_to_world_2d(camera_transform, cursor))
    {
        cursor_pos.0 = cursor_world_pos;
    }
}
