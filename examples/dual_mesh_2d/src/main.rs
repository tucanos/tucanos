use simplex_mesh::{
    boundary_mesh_2d::BoundaryMesh2d,
    dual_mesh::{DualMesh, DualType},
    dual_mesh_2d::DualMesh2d,
    mesh::Mesh,
    mesh_2d::{Mesh2d, rectangle_mesh},
};

fn main() {
    let msh = rectangle_mesh::<Mesh2d>(2.0, 30, 1.0, 20);
    let (msh, _, _, _) = msh.reorder_rcm();
    let (bdy, _): (BoundaryMesh2d, _) = msh.boundary();

    msh.write_vtk("mesh.vtu", None, None).unwrap();
    bdy.write_vtk("bdy.vtu", None, None).unwrap();

    let dual = DualMesh2d::new(&msh, DualType::Median);
    dual.write_vtk("median.vtu").unwrap();
    let (bdy, _): (BoundaryMesh2d, _) = dual.boundary();
    bdy.write_vtk("median_bdy.vtu", None, None).unwrap();

    let dual = DualMesh2d::new(&msh, DualType::Barth);
    dual.write_vtk("barth.vtu").unwrap();
    let (bdy, _): (BoundaryMesh2d, _) = dual.boundary();
    bdy.write_vtk("barth_bdy.vtu", None, None).unwrap();
}
