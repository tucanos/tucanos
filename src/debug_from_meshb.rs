use env_logger::Env;
use log::info;
use remesh::{
    geometry::LinearGeometry,
    mesh::SimplexMesh,
    mesh_stl::orient_stl,
    meshb_io::GmfWriter,
    meshb_io::{GmfElementTypes, GmfReader},
    metric::{AnisoMetric3d, Metric},
    min_max_iter,
    remesher::{Remesher, RemesherParams},
    topo_elems::{Tetrahedron, Triangle},
    Error, Mesh, Result,
};

pub fn init_log(level: &str) {
    env_logger::Builder::from_env(Env::default().default_filter_or(level))
        .format_timestamp(None)
        .init();
}

fn main() -> Result<()> {
    #[cfg(feature = "libmeshb-sys")]
    {
        init_log("debug");

        let p = 2;
        let h_min = 1e-4;
        let h_max = 100.0;
        let n_elems = 200000;
        let beta = 1.5;

        let mesh_fname = std::env::args().nth(1).expect("no mesh file given");
        let sol_fname = std::env::args().nth(2).expect("no sol file given");
        let bdy_fname = std::env::args().nth(3);

        // Read the mesh
        let reader = GmfReader::new(&mesh_fname);
        if reader.is_invalid() {
            return Err(Error::from("Cannot open the mesh file"));
        }
        assert_eq!(reader.dim(), 3);
        let coords = reader.read_vertices();
        let (elems, etags) = reader.read_elements(GmfElementTypes::Tetrahedron);
        let (faces, ftags) = reader.read_elements(GmfElementTypes::Triangle);

        let mut mesh = SimplexMesh::<3, Tetrahedron>::new(coords, elems, etags, faces, ftags);

        // Read the solution
        let reader = GmfReader::new(&sol_fname);
        if reader.is_invalid() {
            return Err(Error::from("Cannot open the solution file"));
        }

        let (sol, m) = reader.read_solution();
        let mut metric = Vec::with_capacity(mesh.n_verts() as usize);
        if m == 1 {
            let (mini, maxi) = min_max_iter(sol.iter().copied());
            info!("solution: mini = {}, maxi = {}", mini, maxi);

            // Compute the hessian
            mesh.compute_vertex_to_vertices();
            let hessian = mesh.hessian(&sol, 2)?;
            let (mini, maxi) = min_max_iter(hessian.iter().copied());
            info!("hessian: mini = {}, maxi = {}", mini, maxi);

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
                mesh.scale_metric(&mut metric, h_min, h_max, n_elems, 5);
                mesh.apply_metric_gradation(&mut metric, beta, 5)?;
            }

            info!(
                "Metric complexity: {} / target = {}",
                mesh.complexity(metric.iter().cloned(), f64::MIN_POSITIVE, f64::MAX),
                n_elems
            );
        } else if m == 6 {
            for i_vert in 0..mesh.n_verts() {
                let m_v = AnisoMetric3d::from_slice(&sol, i_vert);
                metric.push(m_v);
            }
        }

        let mut gmesh = if let Some(bdy_fname) = bdy_fname {
            let reader = GmfReader::new(&bdy_fname);
            if reader.is_invalid() {
                return Err(Error::from("Cannot open the mesh file"));
            }
            assert_eq!(reader.dim(), 3);
            let coords = reader.read_vertices();
            let (elems, etags) = reader.read_elements(GmfElementTypes::Triangle);

            SimplexMesh::<3, Triangle>::new(coords, elems, etags, vec![0; 0], vec![0; 0])
        } else {
            mesh.boundary()
        };

        mesh.boundary()
            .write_xdmf("boundary_in.xdmf", None, None, None)?;

        orient_stl(&mesh, &mut gmesh);
        gmesh.compute_octree();
        let geom = LinearGeometry::new(gmesh).unwrap();
        let mut remesher = Remesher::new(&mesh, &metric, geom)?;

        remesher.remesh(RemesherParams::default());
        remesher.check()?;

        let mesh = remesher.to_mesh(true);

        let mut writer = GmfWriter::new("remeshed.meshb", 3);

        if writer.is_invalid() {
            return Err(Error::from("Cannot open the result file"));
        }

        writer.write_mesh(&mesh);
        mesh.boundary()
            .write_xdmf("boundary_out.xdmf", None, None, None)?;

        Ok(())
    }
}
