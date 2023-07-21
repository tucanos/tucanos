use std::time::Instant;
use tucanos::{
    mesh::SimplexMesh,
    meshb_io::read_solb,
    metric::{AnisoMetric3d, Metric},
    topo_elems::Tetrahedron,
    Result,
};

fn main() -> Result<()> {
    let mesh_fname = std::env::args().nth(1).expect("no mesh file given");
    let sol_fname = std::env::args().nth(2).expect("no sol file given");

    // Read the mesh
    let mut mesh = SimplexMesh::<3, Tetrahedron>::read_meshb(&mesh_fname)?;

    // Read the solution
    let (sol, m) = read_solb(&sol_fname)?;

    if m == 1 {
        // Compute the hessian
        mesh.compute_vertex_to_vertices();
        let now = Instant::now();
        let _ = mesh.hessian(&sol, Some(2), false)?;
        println!(
            "LS hessian: {:.3e}",
            1e-6 * now.elapsed().as_micros() as f64
        );
    } else if m == 6 {
        mesh.compute_edges();
        let mut m = sol
            .as_slice()
            .chunks(6)
            .map(AnisoMetric3d::from_slice)
            .collect::<Vec<_>>();
        let now = Instant::now();
        mesh.apply_metric_gradation(&mut m, 1.5, 10)?;
        println!("gradation: {:.3e}", 1e-6 * now.elapsed().as_micros() as f64);
    }

    Ok(())
}
