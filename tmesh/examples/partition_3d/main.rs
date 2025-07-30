//! Mesh partition example
use std::{path::Path, process::Command, time::Instant};
#[cfg(feature = "metis")]
use tmesh::mesh::partition::{MetisKWay, MetisPartitioner, MetisRecursive};
use tmesh::{
    Result,
    mesh::{
        BoundaryMesh3d, Mesh, Mesh3d,
        partition::{
            HilbertBallPartitioner, HilbertPartitioner, KMeansPartitioner3d, RCMPartitioner,
        },
    },
};

/// .geo file to generate the input mesh with gmsh:
const GEO_FILE: &str = r#"// Gmsh project created on Tue Jun 10 20:58:23 2025
SetFactory("OpenCASCADE");
Cone(1) = {0, 0, 0, 1, 0, 0, 0.5, 0.1, 2*Pi};
Sphere(2) = {0, 0, 0, 0.1, -Pi/2, Pi/2, 2*Pi};
BooleanDifference{ Curve{2}; Volume{1}; Delete; }{ Volume{2}; Delete; }
MeshSize {3} = 0.01;
MeshSize {4} = 0.001;

Physical Surface("cone", 12) = {1};
Physical Surface("top", 13) = {2};
Physical Surface("bottom", 14) = {3};
Physical Surface("sphere", 15) = {4, 5};
Physical Volume("E", 16) = {1};

"#;

fn print_partition_cc(msh: &Mesh3d, n_parts: usize) {
    for i in 0..n_parts {
        let pmesh = msh.get_partition(i).mesh;
        let faces = pmesh.all_faces();
        let graph = pmesh.element_pairs(&faces);
        let cc = graph.connected_components().unwrap();
        let n_cc = cc.iter().copied().max().unwrap_or(0) + 1;
        println!("  part {i}: {n_cc} components");
    }
}

#[allow(clippy::too_many_lines)]
fn main() -> Result<()> {
    let fname = "geom3d.mesh";
    let fname = Path::new(fname);

    if !fname.exists() {
        std::fs::write("geom3d.geo", GEO_FILE)?;

        let output = Command::new("gmsh")
            .arg("geom3d.geo")
            .arg("-3")
            .arg("-o")
            .arg(fname.to_str().unwrap())
            .output()?;

        assert!(
            output.status.success(),
            "gmsh error: {}",
            String::from_utf8(output.stderr).unwrap()
        );
    }

    let msh = Mesh3d::from_meshb(fname.to_str().unwrap())?;

    let (mut msh, _, _, _) = msh.reorder_rcm();
    let (bdy, _): (BoundaryMesh3d, _) = msh.boundary();

    msh.write_vtk("geom3d.vtu")?;
    bdy.write_vtk("geom3d_bdy.vtu")?;

    println!("# of elements: {}", msh.n_elems());

    let n_parts = 4;

    let start = Instant::now();
    let (quality, imbalance) = msh.partition::<HilbertPartitioner>(n_parts, None)?;
    let t = start.elapsed();
    println!(
        "HilbertPartitioner: {:.2e}s, quality={:.2e}, imbalance={:.2e}",
        t.as_secs_f64(),
        quality,
        imbalance
    );
    print_partition_cc(&msh, n_parts);

    let start = Instant::now();
    let (quality, imbalance) = msh.partition::<HilbertBallPartitioner>(n_parts, None)?;
    let t = start.elapsed();
    println!(
        "HilbertBallPartitioner: {:.2e}s, quality={:.2e}, imbalance={:.2e}",
        t.as_secs_f64(),
        quality,
        imbalance
    );
    print_partition_cc(&msh, n_parts);

    let start = Instant::now();
    let (quality, imbalance) = msh.partition::<RCMPartitioner>(n_parts, None)?;
    let t = start.elapsed();
    println!(
        "RCMPartitioner: {:.2e}s, quality={:.2e}, imbalance={:.2e}",
        t.as_secs_f64(),
        quality,
        imbalance
    );
    print_partition_cc(&msh, n_parts);

    let start = Instant::now();
    let (quality, imbalance) = msh.partition::<KMeansPartitioner3d>(n_parts, None)?;
    let t = start.elapsed();
    println!(
        "KMeansPartitioner3d: {:.2e}s, quality={:.2e}, imbalance={:.2e}",
        t.as_secs_f64(),
        quality,
        imbalance
    );
    print_partition_cc(&msh, n_parts);

    #[cfg(feature = "metis")]
    {
        let start = Instant::now();
        let (quality, imbalance) =
            msh.partition::<MetisPartitioner<MetisRecursive>>(n_parts, None)?;
        let t = start.elapsed();
        println!(
            "MetisPartitioner<MetisRecursive>: {:.2e}s, quality={:.2e}, imbalance={:.2e}",
            t.as_secs_f64(),
            quality,
            imbalance
        );
        print_partition_cc(&msh, n_parts);

        let start = Instant::now();
        let (quality, imbalance) = msh.partition::<MetisPartitioner<MetisKWay>>(n_parts, None)?;
        let t = start.elapsed();
        println!(
            "MetisPartitioner<MetisKWay>: {:.2e}s, quality={:.2e}, imbalance={:.2e}",
            t.as_secs_f64(),
            quality,
            imbalance
        );
        print_partition_cc(&msh, n_parts);
    }

    Ok(())
}
