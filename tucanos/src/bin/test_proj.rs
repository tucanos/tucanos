use std::{f64::consts::PI, time::Instant};

use rand::{Rng, SeedableRng, rngs::StdRng};
use tmesh::{
    Result, Vert3d,
    mesh::{
        BoundaryMesh3d, FastQuadraticBoundaryMesh3d, Mesh, QuadraticBoundaryMesh3d,
        quadratic_sphere_mesh, sphere_mesh,
    },
};
use tucanos::{
    geometry::{Geometry, MeshedGeometry},
    mesh::MeshTopology,
};

fn run(mesh: &impl Mesh<3>, pts: &[Vert3d], bdy: impl Mesh<3>) {
    println!("  # of verts = {}", bdy.n_verts());

    let start = Instant::now();
    let qgeom = MeshedGeometry::new(mesh, &MeshTopology::new(mesh), bdy).unwrap();
    println!("  geom built in : {:.2e}s", start.elapsed().as_secs_f64());

    let start = Instant::now();
    for &pt in pts {
        let mut pt = pt;
        qgeom.project(&mut pt, &(2, 1));
    }
    println!("  project in : {:.2e}s", start.elapsed().as_secs_f64());
}

fn main() -> Result<()> {
    let h = 0.01;
    let n = 1000;
    let m = 7;

    let mut rng = StdRng::seed_from_u64(1234);

    let pts = (0..n)
        .map(|_| {
            let r = 1.0 + h * rng.random::<f64>();
            let theta = PI * rng.random::<f64>();
            let phi = 0.5 * PI * rng.random::<f64>();
            Vert3d::new(
                r * theta.cos() * phi.cos(),
                r * theta.sin() * phi.cos(),
                phi.sin(),
            )
        })
        .collect::<Vec<_>>();

    let mut mesh: BoundaryMesh3d = sphere_mesh(1.0, m + 1);
    mesh.fix()?;

    let bdy = mesh.clone();
    let qbdy: QuadraticBoundaryMesh3d = quadratic_sphere_mesh(1.0, m);
    let qbdy_fast: FastQuadraticBoundaryMesh3d = quadratic_sphere_mesh(1.0, m);

    println!("Linear mesh:");
    run(&mesh, &pts, bdy);

    println!("\nQuadratic mesh:");
    run(&mesh, &pts, qbdy);

    println!("\nFast Quadratic mesh:");
    run(&mesh, &pts, qbdy_fast);

    Ok(())
}
