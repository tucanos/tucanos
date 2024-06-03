use nalgebra::Rotation3;
use std::{f64::consts::FRAC_PI_4, time::Instant};
use tucanos::{mesh::Point, test_meshes::test_mesh_3d, Result};

fn main() -> Result<()> {
    let mesh0 = test_mesh_3d()
        .split()
        .split()
        .split()
        .split()
        .split()
        .split();
    println!("Mesh 0: {} vertices", mesh0.n_verts());

    let rot = Rotation3::from_euler_angles(FRAC_PI_4, FRAC_PI_4, FRAC_PI_4);

    let mut mesh1 = test_mesh_3d();
    mesh1.mut_verts().for_each(|x| {
        let p = Point::<3>::new(0.5, 0.5, 0.5);
        let tmp = 0.5 * (rot * (*x - p));
        *x = p + tmp;
    });
    let mesh1 = mesh1.split().split().split().split().split().split();
    println!("Mesh 1: {} vertices", mesh1.n_verts());

    println!("Linear interpolation");
    let now = Instant::now();
    let tree = mesh0.compute_elem_tree();
    println!("Tree built in {}s", now.elapsed().as_secs_f64());

    let f0 = mesh0
        .verts()
        .map(|x| 1.0 * x[0] + 2.0 * x[1] + 3.0 * x[2])
        .collect::<Vec<_>>();
    let now = Instant::now();
    let f1 = mesh0.interpolate_linear(&tree, &mesh1, &f0, None)?;
    println!("Interpolation done in {}s", now.elapsed().as_secs_f64());

    let err = mesh1
        .verts()
        .enumerate()
        .map(|(i, x)| {
            let f = 1.0 * x[0] + 2.0 * x[1] + 3.0 * x[2];
            (f1[i] - f).abs()
        })
        .fold(f64::NEG_INFINITY, f64::max);

    println!("max. err: {err:2.3e}");

    println!("Nearest interpolation");
    let now = Instant::now();
    let tree = mesh0.compute_vert_tree();
    println!("Tree built in {}s", now.elapsed().as_secs_f64());

    let f0 = mesh0
        .verts()
        .map(|x| 1.0 * x[0] + 2.0 * x[1] + 3.0 * x[2])
        .collect::<Vec<_>>();
    let now = Instant::now();
    let f1 = mesh0.interpolate_nearest(&tree, &mesh1, &f0)?;
    println!("Interpolation done in {}s", now.elapsed().as_secs_f64());

    let err = mesh1
        .verts()
        .enumerate()
        .map(|(i, x)| {
            let f = 1.0 * x[0] + 2.0 * x[1] + 3.0 * x[2];
            (f1[i] - f).abs()
        })
        .fold(f64::NEG_INFINITY, f64::max);

    println!("max. err: {err:2.3e}");

    Ok(())
}
