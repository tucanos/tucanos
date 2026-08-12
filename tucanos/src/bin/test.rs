use std::{sync::Arc, time::Instant};

use faer::Mat;
use ferreus_rbf::{
    RBFInterpolator,
    interpolant_config::{
        Drift, FittingAccuracy, FittingAccuracyType, InterpolantSettings, RBFKernelType,
    },
    progress::{ProgressMsg, ProgressSink, closure_sink},
};
use tmesh::{
    Result,
    mesh::{
        AdaptiveBoundsQuadraticTetrahedron, GSimplex, Mesh, Mesh3d, QuadraticBoundaryMesh3d,
        QuadraticMesh3d, Simplex, SubMesh, to_quadratic::to_quadratic_tetrahedron_mesh,
    },
};
use tucanos::{
    geometry::{Geometry, MeshedGeometry},
    mesh::MeshTopology,
};
// use tucanos::{
//     geometry::{Geometry, MeshedGeometry},
//     mesh::MeshTopology,
// };

fn quadratic_to_linear_mesh(msh: &QuadraticMesh3d) -> Mesh3d {
    let mut new_msh = Mesh3d::empty();
    new_msh.add_verts(msh.verts());
    new_msh.add_elems(msh.elems().map(|e| e.linear()), msh.etags());
    new_msh.add_faces(msh.faces().map(|e| e.linear()), msh.ftags());

    // remove the unused vertices
    let submesh = SubMesh::new(&new_msh, |_| true);
    submesh.mesh
}

/// Generates a callback closure_sink
fn get_callback_sink() -> Arc<dyn ProgressSink> {
    let (sink, _listener) = closure_sink(256, |msg| match msg {
        ProgressMsg::SolverIteration {
            iter,
            residual,
            progress,
        } => {
            println!(
                "Iteration: {:>3}    {:>.5E}    {:>.1}%",
                iter,
                residual,
                progress * 100.0
            );
        }
        ProgressMsg::SurfacingProgress {
            isovalue,
            stage,
            progress,
        } => {
            println!(
                "Isovalue: {:?}    Stage: {}    {:>.1}%",
                isovalue,
                stage,
                progress * 100.0
            );
        }
        ProgressMsg::DuplicatesRemoved { num_duplicates } => {
            println!("Removed {num_duplicates:>3} duplicate points");
        }

        ProgressMsg::Message { message } => {
            println!("{message}");
        }
    });

    sink
}

fn linear_to_quadratic_mesh(
    msh: &Mesh3d,
    bdy: &QuadraticBoundaryMesh3d,
) -> Result<(MeshTopology, QuadraticMesh3d)> {
    let mut msh: QuadraticMesh3d = to_quadratic_tetrahedron_mesh(msh);

    let topo = MeshTopology::new(&msh);

    let mut geom = MeshedGeometry::new(bdy)?;
    geom.set_topo_map(topo.topo());

    // Define the RBF kernel to use
    let kernel_type = RBFKernelType::Linear;

    // Define the desired fitting accuracy
    let fitting_accuracy = FittingAccuracy {
        tolerance: 0.001,
        tolerance_type: FittingAccuracyType::Absolute,
    };

    // Initialise an InterpolantSettings instance
    let interpolant_settings = InterpolantSettings::builder(kernel_type)
        .fitting_accuracy(fitting_accuracy)
        .drift(Drift::Linear)
        .build();

    // Create a callback to receive progress updates from the RBFInterpolator
    let callback = get_callback_sink();

    let vtags = topo.vtags();
    // Get the number of boundary vertices in the mesh
    let n = vtags.iter().filter(|tag| tag.0 < 3).count();

    let mut source_points = Mat::zeros(n, 3);
    let mut source_values = Mat::zeros(n, 3);
    let start = Instant::now();
    for (i, (pt, tag)) in msh
        .verts()
        .zip(vtags)
        .filter(|(_, tag)| tag.0 < 3)
        .enumerate()
    {
        if tag.0 < 3 {
            let mut pt_proj = pt;
            geom.project(&mut pt_proj, tag);
            for j in 0..3 {
                source_points[(i, j)] = pt[j];
                source_values[(i, j)] = pt_proj[j] - pt[j];
            }
        }
    }
    let duration = start.elapsed().as_secs_f64();
    println!("Projected {n} boundary vertices in {duration:.3}s");

    // Setup and solve the RBF system
    let rbfi = RBFInterpolator::builder(source_points, source_values, interpolant_settings)
        .progress_callback(callback.clone())
        .build();

    let mut target_points = Mat::zeros(msh.n_verts(), 3);
    for (i, pt) in msh.verts().enumerate() {
        for j in 0..3 {
            target_points[(i, j)] = pt[j];
        }
    }

    let start = Instant::now();
    let res = rbfi.evaluate(target_points.as_ref());
    let duration = start.elapsed().as_secs_f64();
    println!("Computed {} displacements in {duration:.3}s", msh.n_verts());

    for (i, pt) in msh.verts_mut().enumerate() {
        for j in 0..3 {
            pt[j] += res[(i, j)];
        }
    }

    Ok((topo, msh))
}

fn main() -> Result<()> {
    let fname = "ForXavier/1_p1_p2_curving_no_coda/geom.meshb";
    let geom = QuadraticBoundaryMesh3d::from_meshb(fname)?;
    println!(
        "Read mesh from {}: {} elems, {} faces, {} verts",
        fname,
        geom.n_elems(),
        geom.n_faces(),
        geom.n_verts()
    );

    let fname = "ForXavier/1_p1_p2_curving_no_coda/qm_deformed_folded.meshb";
    let msh = QuadraticMesh3d::from_meshb(fname)?;
    println!(
        "Read mesh from {}: {} elems, {} faces, {} verts",
        fname,
        msh.n_elems(),
        msh.n_faces(),
        msh.n_verts()
    );

    let lin_msh = quadratic_to_linear_mesh(&msh);

    let (topo, quad_msh) = linear_to_quadratic_mesh(&lin_msh, &geom)?;
    let vtags = topo.vtags();

    let lu = AdaptiveBoundsQuadraticTetrahedron::lagrange_to_bezier();

    let mut count = 0;
    for (i, ge) in quad_msh.gelems().enumerate() {
        // let ge = ge.flatten();
        let (_, (min, max)) =
            AdaptiveBoundsQuadraticTetrahedron::new(&ge, &lu).compute_bounds(None);
        let d = max / min;
        if d < 0.0 {
            let vol = ge.vol() * 6.0;
            let res = ge.linear().vol() * 6.0;
            println!(
                "Elem {i}: d = {d:.3e}, min = {min:.3e}, max = {max:.3e}, linear = {res:.3e}, vol = {vol:.3e}"
            );
            let flat = ge.flatten();
            let lin = ge.linear();
            let r = lin.radius();
            println!("  Linear radius = {r:.3e}");
            let e = quad_msh.elem(i);
            for i in 4..10 {
                let d = ge[i] - flat[i];
                let dist = d.norm();
                let i_vert = e.get(i);
                let tag = vtags[i_vert];
                println!("  Node {i}: dist = {dist:.3e}, i_vert = {i_vert}, tag = {tag:?}");
            }
            count += 1;
        }
    }
    println!("Found {count} elements with negative distortion");

    // let mut geom = MeshedGeometry::new(&geom)?;
    // let topo = MeshTopology::new(&msh);
    // geom.set_topo_map(topo.topo());

    // for (pt, tag) in msh.verts().zip(topo.vtags().iter()) {
    //     if tag.0 < 3 {
    //         let mut pt_proj = pt;
    //         geom.project(&mut pt_proj, tag);
    //         let dist = (pt - pt_proj).norm();
    //         if dist > 1e-3 {
    //             println!("{pt:?} (tag={tag:?}) projected to {pt_proj:?} with distance {dist:.3e}");
    //         }
    //     }
    // }

    // let pt = Vert3d::new(0.8236791848256956, 1.216911083404943, 0.005691081388471509);
    // let tag = (2, 3);
    // let mut pt_proj = pt;
    // geom.project(&mut pt_proj, &tag);
    // let dist = (pt - pt_proj).norm();
    // println!("{pt:?} (tag={tag:?}) projected to {pt_proj:?} with distance {dist:.3e}");

    // let pt = Vert3d::new(0.8236791848256956, 1.216911083404943, 0.005691081388471509);

    // for (i, ge) in geom.gelems().enumerate().skip(47702).take(1) {
    //     println!("Elem {i}");
    //     // let ge = ge.flatten();

    //     let _b = ge.bcoords(&pt);
    //     // let b2 = ge.bcoords_algebraic(&pt);
    //     // let p = ge.mapping(&b);
    //     // let p2 = ge.mapping(&b2);
    //     // let err = (pt - p).norm();
    //     // let err2 = (pt - p2).norm();
    //     // println!("err = {err:.3e}, err2 = {err2:.3e}");
    //     // let b = Vert3d::from_column_slice(&ge.bcoords(&pt));
    //     // let b2 = Vert3d::from_column_slice(&ge.bcoords_algebraic(&pt));
    //     // let err = (b - b2).norm();
    //     // assert!(
    //     //     err < 1e-2,
    //     //     "bcoords mismatch: {b:?} vs {b2:?} (err={err:.3e})"
    //     // );
    // }
    Ok(())
}
