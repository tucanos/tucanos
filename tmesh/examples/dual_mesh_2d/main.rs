//! Dual meshes in 2d
use std::{path::Path, process::Command};
use tmesh::{
    Result,
    dual::{DualMesh, DualMesh2d, DualType, PolyMesh, SimplePolyMesh},
    mesh::{BoundaryMesh2d, Mesh, Mesh2d},
};

/// .geo file to generate the input mesh with gmsh:
const GEO_FILE: &str = r#"Point(1) = {0, 0, 0, 1.0};
Point(2) = {1, 0, 0, 1.0};
Point(3) = {0, 1, 0, 1.0};
Circle(1) = {2, 1, 3};
Line(2) = {3, 1};
Line(3) = {1, 2};
Curve Loop(1) = {3, 1, 2};
Plane Surface(1) = {1};
Physical Curve("B1", 4) = {1};
Physical Curve("B2", 5) = {3, 2};
Physical Surface("S1", 6) = {1};
MeshSize {1, 2, 3} = 0.1;

Field[1] = BoundaryLayer;
Field[1].AnisoMax = 10;
Field[1].Quads = 0;
Field[1].Thickness = 0.2;
Field[1].CurvesList = {1};
Field[1].NbLayers = 10;
Field[1].Ratio = 1.1;
Field[1].Size = 0.025;
Field[1].SizeFar = 0.1;
Field[1].PointsList = {2, 3};

BoundaryLayer  Field = 1;"#;

fn main() -> Result<()> {
    let fname = "quarter_disk.mesh";
    let fname = Path::new(fname);

    if !fname.exists() {
        std::fs::write("quarter_disk.geo", GEO_FILE)?;

        let output = Command::new("gmsh")
            .arg("quarter_disk.geo")
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

    let msh = Mesh2d::from_meshb(fname.to_str().unwrap())?;

    let (msh, _, _, _) = msh.reorder_rcm();
    let (bdy, _): (BoundaryMesh2d, _) = msh.boundary();

    msh.write_vtk("mesh.vtu")?;
    bdy.write_vtk("mesh_bdy.vtu")?;

    let dual = DualMesh2d::new(&msh, DualType::Median);
    dual.write_vtk("median.vtu")?;
    let (bdy, _): (BoundaryMesh2d, _) = dual.boundary();
    bdy.write_vtk("median_bdy.vtu")?;

    let poly = SimplePolyMesh::simplify(&dual, true);
    poly.write_vtk("median_simplified.vtu")?;

    println!("dual: {} faces", dual.n_faces());
    println!("simplidied: {} faces", poly.n_faces());

    let dual = DualMesh2d::new(&msh, DualType::ThresholdBarth(0.1));
    dual.write_vtk("barth.vtu")?;
    let (bdy, _): (BoundaryMesh2d, _) = dual.boundary();
    bdy.write_vtk("barth_bdy.vtu")?;

    let poly = SimplePolyMesh::simplify(&dual, true);
    poly.write_vtk("barth_simplified.vtu")?;

    println!("dual: {} faces", dual.n_faces());
    println!("simplidied: {} faces", poly.n_faces());

    Ok(())
}
