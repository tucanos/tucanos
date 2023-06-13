use std::collections::HashMap;

use env_logger::Env;
use log::{info, warn};
use tucanos::{
    geometry::{Geometry, LinearGeometry},
    mesh::SimplexMesh,
    mesh_stl::orient_stl,
    meshb_io::read_solb,
    metric::{AnisoMetric3d, Metric},
    remesher::{Remesher, RemesherParams},
    topo_elems::{Tetrahedron, Triangle},
    Mesh, Result,
};

pub fn init_log(level: &str) {
    env_logger::Builder::from_env(Env::default().default_filter_or(level))
        .format_timestamp(None)
        .init();
}

fn main() -> Result<()> {
    init_log("debug");

    let beta = 2.0; // gradation
    let step = 4.0; //
    let r_h = 4.0;
    let p = 2;
    let target_h_min = 1e-5;
    let target_h_max = 15.0;
    let target_n_elems = 150000;

    let mesh_fname = std::env::args().nth(1).expect("no mesh file given");
    let bdy_fname = std::env::args().nth(2).expect("no geometry file given");
    let sol_fname = std::env::args().nth(3);

    // Read the mesh
    let mut mesh = SimplexMesh::<3, Tetrahedron>::read_meshb(&mesh_fname)?;
    mesh.compute_topology();

    mesh.compute_face_to_elems();
    mesh.check()?;

    mesh.boundary().0.write_vtk("boundary_in.vtu", None, None)?;

    // Read & prepare the geometry
    let mut geom = SimplexMesh::<3, Triangle>::read_meshb(&bdy_fname)?;

    orient_stl(&mesh, &mut geom);
    let mut geom = LinearGeometry::new(&mesh, geom).unwrap();
    geom.compute_curvature();
    geom.write_curvature("curvature.vtu")?;

    // Check the face normals
    let max_angle = geom.max_normal_angle(&mesh);
    if max_angle > 20.0 {
        warn!("Max angle: {} degrees", max_angle);
    } else {
        info!("Max angle: {} degrees", max_angle);
    }

    let mut metric = Vec::with_capacity(mesh.n_verts() as usize);

    mesh.compute_vertex_to_vertices();
    mesh.compute_vertex_to_elems();
    mesh.compute_volumes();

    let implied_metric = mesh.implied_metric()?;

    let curvature_metric = mesh.curvature_metric(&geom, r_h, beta, None, None)?;

    if let Some(sol_fname) = sol_fname {
        // Read the solution
        let (sol, m) = read_solb(&sol_fname)?;

        if m == 1 {
            // Compute the hessian
            mesh.compute_vertex_to_vertices();
            let hessian = mesh.hessian(&sol, Some(2), false)?;
            // let grad = mesh.gradient_l2proj(&sol)?;
            // let hessian = mesh.hessian_l2proj(&grad)?;

            // Compute the metric
            let exponent = -2.0 / (2.0 * p as f64 + 3.0);
            for i_vert in 0..mesh.n_verts() {
                let mut m_v = AnisoMetric3d::from_slice(&hessian, i_vert);
                let scale = f64::powf(m_v.vol(), exponent);
                if !scale.is_nan() {
                    m_v.scale(scale);
                }
                metric.push(m_v);
            }
            for _ in 0..2 {
                metric = mesh.smooth_metric(&metric)?;
            }

            let (h_min, h_max, aniso_max, complexity) = mesh.metric_info(&curvature_metric);
            info!(
                "curvature metric : h_min = {}, h_max = {}, aniso_max = {}, complexity = {}",
                h_min, h_max, aniso_max, complexity
            );

            mesh.scale_metric(
                &mut metric,
                target_h_min,
                target_h_max,
                target_n_elems,
                Some(&curvature_metric),
                Some(&implied_metric),
                Some(step),
                10,
            )?;

            let (h_min, h_max, aniso_max, complexity) = mesh.metric_info(&metric);
            info!(
                "before gradation : h_min = {}, h_max = {}, aniso_max = {}, complexity = {}",
                h_min, h_max, aniso_max, complexity
            );

            mesh.apply_metric_gradation(&mut metric, beta, 10)?;

            let (h_min, h_max, aniso_max, complexity) = mesh.metric_info(&metric);
            info!(
                "after gradation : h_min = {}, h_max = {}, aniso_max = {}, complexity = {}",
                h_min, h_max, aniso_max, complexity
            );
        } else if m == 6 {
            metric.extend((0..mesh.n_verts()).map(|i| AnisoMetric3d::from_slice(&sol, i)));
        } else {
            panic!("Invalid m = {}", m);
        }
    } else {
        metric = curvature_metric;
    }

    let hmin: Vec<_> = metric.iter().map(|&m| m.sizes()[0]).collect();
    let hmax: Vec<_> = metric.iter().map(|&m| m.sizes()[2]).collect();
    let aniso: Vec<_> = metric
        .iter()
        .map(|&m| m.sizes()[2] / m.sizes()[0])
        .collect();

    let mut data = HashMap::new();
    data.insert(String::from("aniso"), aniso.as_slice());
    data.insert(String::from("hmin"), hmin.as_slice());
    data.insert(String::from("hmax"), hmax.as_slice());

    mesh.write_vtk("input.vtu", Some(data), None)?;

    let (bdy, vert_ids) = mesh.boundary();

    let mut bdy_data = HashMap::new();
    let bdy_aniso = vert_ids
        .iter()
        .map(|&i| aniso[i as usize])
        .collect::<Vec<_>>();
    let bdy_hmin = vert_ids
        .iter()
        .map(|&i| hmin[i as usize])
        .collect::<Vec<_>>();
    let bdy_hmax = vert_ids
        .iter()
        .map(|&i| hmax[i as usize])
        .collect::<Vec<_>>();
    bdy_data.insert(String::from("aniso"), bdy_aniso.as_slice());
    bdy_data.insert(String::from("hmin"), bdy_hmin.as_slice());
    bdy_data.insert(String::from("hmax"), bdy_hmax.as_slice());

    bdy.write_vtk("input_bdy.vtu", Some(bdy_data), None)?;

    let mut remesher = Remesher::new(&mesh, &metric, geom)?;

    let params = RemesherParams {
        max_angle: f64::min(max_angle, 45.0),
        ..Default::default()
    };

    remesher.remesh(params);
    remesher.check()?;

    let mut mesh = remesher.to_mesh(true);
    mesh.compute_face_to_elems();
    mesh.check()?;

    // let mut geom = SimplexMesh::<3, Triangle>::read_meshb(&bdy_fname)?;

    // orient_stl(&mesh, &mut geom);
    // geom.compute_octree();
    // let geom = LinearGeometry::new(geom).unwrap();
    // let max_angle = geom.max_normal_angle(&mesh);
    // if max_angle > 20.0 {
    //     warn!("Max angle: {} degrees", max_angle);
    // } else {
    //     info!("Max angle: {} degrees", max_angle);
    // }

    mesh.write_vtk("output.vtu", None, None)?;
    mesh.write_meshb("output.meshb")?;

    mesh.boundary()
        .0
        .write_vtk("boundary_out.vtu", None, None)?;

    Ok(())
}
