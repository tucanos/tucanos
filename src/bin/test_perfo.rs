use env_logger::Env;
use log::info;
use rayon::prelude::{IntoParallelRefMutIterator, ParallelIterator};
use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader},
    time::Instant,
};
use tucanos::{
    geometry::{Geometry, LinearGeometry},
    mesh::SimplexMesh,
    mesh_stl::orient_stl,
    metric::{AnisoMetric3d, Metric},
    parallel::{ParallelRemesher, ParallelRemeshingParams},
    remesher::RemesherParams,
    topo_elems::{Tetrahedron, Triangle},
    Result,
};

fn init_log(level: &str) {
    env_logger::Builder::from_env(Env::default().default_filter_or(level))
        .format_timestamp(None)
        .init();
}

fn read_ma(fname: &str) -> Result<Vec<f64>> {
    let file = File::open(fname)?;
    let mut reader = BufReader::new(file);
    let mut line = String::new();

    let mut res = Vec::new();

    while reader.read_line(&mut line)? > 0 {
        let trimmed_line = line.trim();
        res.push(trimmed_line.parse()?);
        line.clear();
    }

    Ok(res)
}

fn main() -> Result<()> {
    init_log("info");

    let mut mesh = SimplexMesh::<3, Tetrahedron>::read_meshb("data/mesh.meshb")?;
    mesh.compute_topology();

    info!("{} vertices", mesh.n_verts());
    info!("{} elements", mesh.n_elems());
    info!("{} faces", mesh.n_faces());

    let ma_elems = read_ma("data/Ma.dat")?;
    assert_eq!(ma_elems.len(), mesh.n_elems() as usize);

    let start = Instant::now();

    mesh.compute_vertex_to_elems();
    mesh.compute_volumes();

    // Multiscale metric
    let now = Instant::now();
    let mut ma_verts = mesh.elem_data_to_vertex_data(&ma_elems)?;

    mesh.compute_vertex_to_vertices();
    for _ in 0..2 {
        ma_verts = mesh.smooth(&ma_verts, 0)?;
    }

    let hessian = mesh.hessian(&ma_verts, None, true)?;

    let mut metric = hessian
        .chunks(AnisoMetric3d::N)
        .map(AnisoMetric3d::from_slice)
        .collect::<Vec<_>>();

    let p = 2;
    let exponent = 2.0 / (2.0 * f64::from(p) + f64::from(3));
    metric.as_mut_slice().par_iter_mut().for_each(|m| {
        let scale = f64::powf(m.vol(), exponent);
        if !scale.is_nan() {
            m.scale(scale);
        }
    });

    for _ in 0..2 {
        metric = mesh.smooth_metric(&metric)?;
    }
    info!("Multiscale metric: {:.2e}s", now.elapsed().as_secs_f32());

    // Implied metric
    let now = Instant::now();
    let implied_metric = mesh.implied_metric()?;
    info!("Implied metric: {:.2e}s", now.elapsed().as_secs_f32());

    // Fixed metric
    let now = Instant::now();
    let r_h = 4.0;
    let beta = 1.3;
    let mut bdy = SimplexMesh::<3, Triangle>::read_meshb("data/bdy.meshb")?;
    orient_stl(&mesh, &mut bdy);
    let mut geom = LinearGeometry::new(&mesh, bdy)?;
    geom.compute_curvature();

    let mut fixed_metric = mesh.curvature_metric(&geom, r_h, beta, None, None)?;
    for _ in 0..2 {
        fixed_metric = mesh.smooth_metric(&fixed_metric)?;
    }
    info!("Fixed metric: {:.2e}s", now.elapsed().as_secs_f32());

    // Scaling
    let now = Instant::now();
    let step = 2.0;
    let h_min = 1e-4;
    let h_max = 20.0;
    let n_elems = 600000;
    let max_iter = 10;

    let _c = mesh.scale_metric(
        &mut metric,
        h_min,
        h_max,
        n_elems,
        Some(&fixed_metric),
        Some(&implied_metric),
        Some(step),
        max_iter,
    )?;
    info!("Scaling: {:.2e}s", now.elapsed().as_secs_f32());

    // Gradation
    let now = Instant::now();
    mesh.apply_metric_gradation(&mut metric, beta, max_iter)?;
    info!("Gradation: {:.2e}s", now.elapsed().as_secs_f32());

    info!(
        "Metric computation (total): {:.2e}s",
        start.elapsed().as_secs_f32()
    );
    if true {
        panic!("{:.2e}s", start.elapsed().as_secs_f32());
    }
    // Clear
    mesh.clear_vertex_to_vertices();

    // Max angle
    let max_angle = geom.max_normal_angle(&mesh);
    info!("max_angle = {max_angle:.2e}");
    let max_angle = max_angle.max(20.0);

    // Remesh
    let now = Instant::now();
    let remesher = ParallelRemesher::new(
        mesh,
        tucanos::mesh_partition::PartitionType::MetisRecursive(8),
    )?;
    let dd_params = ParallelRemeshingParams::new(2, 2, 10000);
    let remesh_params = RemesherParams {
        max_angle,
        ..RemesherParams::default()
    };

    let (new_mesh, _) = remesher.remesh(&metric, &geom, remesh_params, dd_params)?;
    info!("Remeshing: {:.2e}s", now.elapsed().as_secs_f32());

    // Interpolation
    let now = Instant::now();
    let mesh = remesher.partitionned_mesh();
    let tree = mesh.compute_elem_tree();
    let ma_new_vert = mesh.interpolate_linear(&tree, &new_mesh, &ma_verts, Some(1.0))?;
    let ma_new_elem = new_mesh.vertex_data_to_elem_data(&ma_new_vert)?;
    info!("Interpolation: {:.2e}s", now.elapsed().as_secs_f32());

    info!("Total: {:.2e}s", start.elapsed().as_secs_f32());

    let mut vert_data = HashMap::new();
    vert_data.insert(String::from("Ma"), ma_new_vert.as_slice());
    let mut elem_data = HashMap::new();
    elem_data.insert(String::from("Ma"), ma_new_elem.as_slice());
    new_mesh.write_vtk("output.vtu", Some(vert_data), Some(elem_data))?;

    Ok(())
}
