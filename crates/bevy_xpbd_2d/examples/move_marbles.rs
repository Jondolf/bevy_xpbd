use bevy::prelude::*;
use bevy_xpbd_2d::prelude::*;
use examples_common_2d::XpbdExamplePlugin;

#[derive(Component)]
struct Player;

#[derive(Component, Deref, DerefMut)]
pub struct MoveAcceleration(pub f32);

#[derive(Component, Deref, DerefMut)]
pub struct MaxVelocity(pub Vec2);

pub enum MovementEvent {
    Up(Entity),
    Down(Entity),
    Left(Entity),
    Right(Entity),
}

fn setup(
    mut commands: Commands,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    let sphere = meshes.add(Mesh::from(shape::Icosphere {
        radius: 1.0,
        subdivisions: 4,
    }));

    let white = materials.add(StandardMaterial {
        base_color: Color::rgb(0.8, 0.8, 1.0),
        unlit: true,
        ..default()
    });

    let blue = materials.add(StandardMaterial {
        base_color: Color::rgb(0.2, 0.6, 0.8),
        unlit: true,
        ..default()
    });

    let _floor = commands
        .spawn_bundle(PbrBundle {
            mesh: meshes.add(Mesh::from(shape::Quad::new(Vec2::ONE))),
            material: white.clone(),
            transform: Transform::from_scale(Vec3::new(20.0, 1.0, 1.0)),
            ..default()
        })
        .insert_bundle(RigidBodyBundle::new_static().with_pos(Vec2::new(0.0, -7.5)))
        .insert_bundle(ColliderBundle::new(&Shape::cuboid(10.0, 0.5), 1.0));

    let _ceiling = commands
        .spawn_bundle(PbrBundle {
            mesh: meshes.add(Mesh::from(shape::Quad::new(Vec2::ONE))),
            material: white.clone(),
            transform: Transform::from_scale(Vec3::new(20.0, 1.0, 1.0)),
            ..default()
        })
        .insert_bundle(RigidBodyBundle::new_static().with_pos(Vec2::new(0.0, 7.5)))
        .insert_bundle(ColliderBundle::new(&Shape::cuboid(10.0, 0.5), 1.0));

    let _left_wall = commands
        .spawn_bundle(PbrBundle {
            mesh: meshes.add(Mesh::from(shape::Quad::new(Vec2::ONE))),
            material: white.clone(),
            transform: Transform::from_scale(Vec3::new(1.0, 15.0, 1.0)),
            ..default()
        })
        .insert_bundle(RigidBodyBundle::new_static().with_pos(Vec2::new(-9.5, 0.0)))
        .insert_bundle(ColliderBundle::new(&Shape::cuboid(0.5, 10.0), 1.0));

    let _right_wall = commands
        .spawn_bundle(PbrBundle {
            mesh: meshes.add(Mesh::from(shape::Quad::new(Vec2::ONE))),
            material: white,
            transform: Transform::from_scale(Vec3::new(1.0, 15.0, 1.0)),
            ..default()
        })
        .insert_bundle(RigidBodyBundle::new_static().with_pos(Vec2::new(9.5, 0.0)))
        .insert_bundle(ColliderBundle::new(&Shape::cuboid(0.5, 10.0), 1.0));

    let radius = 0.15;
    let stacks = 20;
    for i in 0..20 {
        for j in 0..stacks {
            let pos = Vec2::new(
                (j as f32 - stacks as f32 * 0.5) * 2.5 * radius,
                2.0 * radius * i as f32 - 2.0,
            );
            commands
                .spawn_bundle(PbrBundle {
                    mesh: sphere.clone(),
                    material: blue.clone(),
                    transform: Transform {
                        scale: Vec3::splat(radius),
                        translation: pos.extend(0.0),
                        ..default()
                    },
                    ..default()
                })
                .insert_bundle(RigidBodyBundle {
                    restitution: Restitution(0.3),
                    ..RigidBodyBundle::new_dynamic().with_pos(pos)
                })
                .insert_bundle(ColliderBundle::new(&Shape::ball(radius), 1.0))
                .insert(Player)
                .insert(MoveAcceleration(0.005))
                .insert(MaxVelocity(Vec2::new(30.0, 30.0)));
        }
    }

    commands.spawn_bundle(OrthographicCameraBundle {
        transform: Transform::from_translation(Vec3::new(0.0, 0.0, 100.0)),
        orthographic_projection: OrthographicProjection {
            scale: 0.025,
            ..default()
        },
        ..OrthographicCameraBundle::new_3d()
    });
}

fn handle_input(
    keyboard_input: Res<Input<KeyCode>>,
    mut ev_movement: EventWriter<MovementEvent>,
    query: Query<Entity, With<Player>>,
) {
    for entity in query.iter() {
        if keyboard_input.pressed(KeyCode::Up) {
            ev_movement.send(MovementEvent::Up(entity));
        }
        if keyboard_input.pressed(KeyCode::Down) {
            ev_movement.send(MovementEvent::Down(entity));
        }
        if keyboard_input.pressed(KeyCode::Left) {
            ev_movement.send(MovementEvent::Left(entity));
        }
        if keyboard_input.pressed(KeyCode::Right) {
            ev_movement.send(MovementEvent::Right(entity));
        }
    }
}

fn player_movement(
    mut ev_movement: EventReader<MovementEvent>,
    mut query: Query<(&mut LinVel, &MaxVelocity, &MoveAcceleration), With<Player>>,
) {
    for ev in ev_movement.iter() {
        for (mut vel, max_vel, move_acceleration) in query.iter_mut() {
            match ev {
                MovementEvent::Up(_ent) => vel.y += move_acceleration.0,
                MovementEvent::Down(_ent) => vel.y -= move_acceleration.0,
                MovementEvent::Left(_ent) => vel.x -= move_acceleration.0,
                MovementEvent::Right(_ent) => vel.x += move_acceleration.0,
            }
            vel.0 = vel.0.clamp(-max_vel.0, max_vel.0);
        }
    }
}

fn main() {
    #[cfg(target_arch = "wasm32")]
    console_error_panic_hook::set_once();

    App::new()
        .insert_resource(ClearColor(Color::BLACK))
        .insert_resource(Msaa { samples: 4 })
        .insert_resource(Gravity(Vec2::new(0.0, -9.81)))
        .insert_resource(NumSubsteps(6))
        .add_plugins(DefaultPlugins)
        .add_plugin(XpbdExamplePlugin)
        .add_event::<MovementEvent>()
        .add_startup_system(setup)
        .add_system(handle_input)
        .add_system(player_movement)
        .run();
}
