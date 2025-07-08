//! Dual meshes in 3d
use std::{path::Path, process::Command};
use tmesh::{
    Result,
    dual::{DualMesh, DualMesh3d, DualType, PolyMesh},
    mesh::{BoundaryMesh3d, Mesh, Mesh3d},
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

fn main() -> Result<()> {
    let fname = "geom3d.mesh";
    let fname = Path::new(fname);

    if !fname.exists() {
        std::fs::write("geom3d.geo", GEO_FILE)?;

        let output = Command::new("gmsh")
            .arg("geom3d.geo")
            .arg("-2")
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

    let (msh, _, _, _) = msh.reorder_rcm();
    let (bdy, _): (BoundaryMesh3d, _) = msh.boundary();

    msh.write_vtk("mesh.vtu")?;
    bdy.write_vtk("mesh_bdy.vtu")?;

    let dual = DualMesh3d::new(&msh, DualType::Median);
    dual.write_vtk("median.vtu")?;
    let (bdy, _): (BoundaryMesh3d, _) = dual.boundary();
    bdy.write_vtk("median_bdy.vtu")?;

    println!("Median");
    println!("dual: {} faces", dual.n_faces());
    let max_verts_per_elem = (0..dual.n_elems())
        .map(|i| dual.elem_n_verts(i))
        .max()
        .unwrap_or(0);
    println!("max # of verts per elem: {max_verts_per_elem}");

    let dual = DualMesh3d::new(&msh, DualType::Barth);
    dual.write_vtk("barth.vtu")?;
    let (bdy, _): (BoundaryMesh3d, _) = dual.boundary();
    bdy.write_vtk("barth_bdy.vtu")?;

    println!("Barth");
    println!("dual: {} faces", dual.n_faces());
    let max_verts_per_elem = (0..dual.n_elems())
        .map(|i| dual.elem_n_verts(i))
        .max()
        .unwrap_or(0);
    println!("max # of verts per elem: {max_verts_per_elem}");

    Ok(())
}
