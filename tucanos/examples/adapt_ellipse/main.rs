mod ellipse;

use ellipse::EllipseProjection;
use env_logger::Env;
use nalgebra::SMatrix;
use rustc_hash::FxHashMap;
use std::{path::Path, process::Command, time::Instant};
use tmesh::mesh::{Mesh, partition::PartitionType};
use tucanos::{
    Result, Tag, TopoTag,
    geometry::Geometry,
    mesh::{GElem, HasTmeshImpl, Vertex, SimplexMesh, Tetrahedron, Topology},
    metric::{AnisoMetric3d, Metric},
    remesher::{ParallelRemesher, ParallelRemesherParams, Remesher, RemesherParams},
};

pub fn init_log(level: &str) {
    env_logger::Builder::from_env(Env::default().default_filter_or(level))
        .format_timestamp(None)
        .init();
}

/// .geo file to generate the input mesh with gmsh:
const GEO_FILE: &str = r#"
SetFactory("OpenCASCADE");
Ellipse(1) = {0, 0, 0, 0.5, 0.1, 0, 2*Pi};
Curve Loop(1) = {1};
Plane Surface(1) = {1};
Extrude {0, 0, 1} {
  Surface{1}; 
}
Sphere(2) = {0, 0, 0, 10, 0, Pi/2, 2*Pi};
BooleanDifference{ Volume{2}; Delete; }{ Volume{1}; Delete; }
Physical Surface("symmetry", 7) = {5};
Physical Surface("farfield", 8) = {4};
Physical Surface("wing", 9) = {2};
Physical Surface("wingtip", 10) = {3};
Physical Volume("fluid", 11) = {2};
MeshSize {2, 1} = 0.05;
"#;

const WINGTIP_TAG: Tag = 3;
const SYMMETRY_TAG: Tag = 5;
const FARFIELD_TAG: Tag = 4;
const WING_TAG: Tag = 2;
const FACE_TAGS: [Tag; 4] = [SYMMETRY_TAG, FARFIELD_TAG, WING_TAG, WINGTIP_TAG];
const FACE_NAMES: [&str; 4] = ["Symmetry", "Farfield", "Wing", "Wingtip"];

struct Simple3dGeometry {
    ellipse: EllipseProjection,
    r: f64,
    topo: Topology,
}

impl Simple3dGeometry {
    const fn new(topo: Topology) -> Self {
        Self {
            ellipse: EllipseProjection::new(0.5, 0.1),
            r: 10.0,
            topo,
        }
    }

    fn normal(&self, pt: &Vertex<3>, tag: TopoTag) -> Vertex<3> {
        let n = match tag.0 {
            2 => match tag.1 {
                SYMMETRY_TAG => Vertex::<3>::new(0.0, 0.0, -1.0),
                FARFIELD_TAG => {
                    let r = pt.norm();
                    1. / r * *pt
                }
                WING_TAG => {
                    let (nx, ny) = self.ellipse.normal(pt[0], pt[1]);
                    -Vertex::<3>::new(nx, ny, 0.0)
                }
                WINGTIP_TAG => Vertex::<3>::new(0.0, 0.0, -1.0),
                _ => panic!("Invalid tag {tag:?}"),
            },
            _ => panic!("Invalid tag {tag:?}"),
        };
        n.normalize()
    }
}

impl Geometry<3> for Simple3dGeometry {
    fn check(&self, _topo: &Topology) -> Result<()> {
        Ok(())
    }

    fn project(&self, pt: &mut Vertex<3>, tag: &TopoTag) -> f64 {
        let old = *pt;

        if tag.1 < 0 {
            return 0.0;
        }

        *pt = match tag.0 {
            2 => match tag.1 {
                SYMMETRY_TAG => Vertex::<3>::new(old[0], old[1], 0.0),
                FARFIELD_TAG => {
                    let r = old.norm();
                    (self.r / r) * old
                }
                WING_TAG => {
                    let (x, y) = self.ellipse.project(old[0], old[1]);
                    Vertex::<3>::new(x, y, old[2].clamp(0.0, 1.0))
                }
                WINGTIP_TAG => {
                    let x = old[0];
                    let y = old[1];
                    if self.ellipse.is_in(x, y) {
                        Vertex::<3>::new(x, y, 1.0)
                    } else {
                        let (x, y) = self.ellipse.project(x, y);
                        Vertex::<3>::new(x, y, 1.0)
                    }
                }

                _ => unreachable!(),
            },
            1 => {
                let node = self
                    .topo
                    .get(*tag)
                    .unwrap_or_else(|| panic!("tag = {tag:?}"));
                assert_eq!(node.parents.len(), 2);
                let mut parents = node.parents.iter().copied().collect::<Vec<_>>();
                parents.sort_unstable();
                let x = old[0];
                let y = old[1];
                if parents[0] == WING_TAG {
                    if parents[1] == WINGTIP_TAG {
                        let (x, y) = self.ellipse.project(x, y);
                        Vertex::<3>::new(x, y, 1.0)
                    } else {
                        assert_eq!(parents[1], SYMMETRY_TAG);
                        let (x, y) = self.ellipse.project(x, y);
                        Vertex::<3>::new(x, y, 0.0)
                    }
                } else if parents[0] == FARFIELD_TAG {
                    let r = x.hypot(y);
                    self.r / r * Vertex::<3>::new(x, y, 0.0)
                } else {
                    unreachable!("{parents:?}");
                }
            }
            0 => old,
            _ => unreachable!("{}", tag.0),
        };
        (*pt - old).norm()
    }

    fn angle(&self, pt: &Vertex<3>, n: &Vertex<3>, tag: &TopoTag) -> f64 {
        let n_ref = self.normal(pt, *tag);
        let cos_a = n.dot(&n_ref).clamp(-1.0, 1.0);
        f64::acos(cos_a).to_degrees()
    }
}

fn check_geom(mesh: &SimplexMesh<3, Tetrahedron>, geom: &Simple3dGeometry) {
    let mut max_dist = FxHashMap::<Tag, f64>::default();
    let mut max_angle = FxHashMap::<Tag, f64>::default();

    for ((face, tag), gface) in mesh.faces().zip(mesh.ftags()).zip(mesh.gfaces()) {
        for i in 0..3 {
            let mut pt = mesh.vert(face[i]);
            let dist = geom.project(&mut pt, &(2, tag));
            if let Some(v) = max_dist.get_mut(&tag) {
                *v = v.max(dist);
            } else {
                max_dist.insert(tag, dist);
            }
        }
        let angle = geom.angle(&gface.center(), &gface.normal(), &(2, tag));
        if let Some(v) = max_angle.get_mut(&tag) {
            *v = v.max(angle);
        } else {
            max_angle.insert(tag, angle);
        }
    }
    for (&name, &tag) in FACE_NAMES.iter().zip(FACE_TAGS.iter()) {
        if max_dist.contains_key(&tag) {
            println!("{name} (tag = {tag})");
            println!("  max. distance: {:.1e}", max_dist.get(&tag).unwrap());
            println!("  max. angle: {:.1}", max_angle.get(&tag).unwrap());
        }
    }
}

fn get_bl_metric(
    mesh: &SimplexMesh<3, Tetrahedron>,
    geom: &Simple3dGeometry,
    h_min: f64,
    h_max: f64,
    eps: f64,
) -> Vec<AnisoMetric3d> {
    let h = |d: f64| (h_min + d * eps).min(h_max);

    mesh.verts()
        .map(|pt| {
            let mut tmp = pt;
            let mut d = geom.project(&mut tmp, &(2, WING_TAG));
            let mut tag = WING_TAG;

            let mut tmp = pt;
            let d1 = geom.project(&mut tmp, &(2, WINGTIP_TAG));
            if d1 < d {
                d = d1;
                tag = WINGTIP_TAG;
            }

            let n = geom.normal(&pt, (2, tag));
            let mat = SMatrix::<f64, 3, 3>::new(n[0], 0., 0., n[1], 0., 0., n[2], 0., 0.);
            let u = mat.svd(true, false).u.unwrap();
            assert!((n - u.column(0)).norm() < 1e-8);

            let h0 = h(d) * u.column(0);
            let h1 = h_max * u.column(1);
            let h2 = h_max * u.column(2);

            AnisoMetric3d::from_sizes(&h0, &h1, &h2)
        })
        .collect::<Vec<_>>()
}
fn main() -> Result<()> {
    init_log("info");

    let fname = "ellipse.mesh";
    let fname = Path::new(fname);

    if !fname.exists() {
        std::fs::write("ellipse.geo", GEO_FILE)?;

        let output = Command::new("gmsh")
            .arg("ellipse.geo")
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

    // Load the mesh
    let mut mesh = SimplexMesh::<3, Tetrahedron>::from_meshb(fname.to_str().unwrap())?;

    // Fix face orientation
    mesh.fix()?;

    // // Check the mesh
    // mesh.check()?;

    // Save the input mesh in .vtu format
    let mut writer = mesh.vtu_writer();
    writer.add_cell_data("edge_ratio", 1, mesh.edge_length_ratios());
    writer.add_cell_data("gamma", 1, mesh.elem_gammas());
    let mut skewness = vec![0.0_f64; mesh.n_elems() as usize];
    for (i0, i1, s) in mesh.face_skewnesses(&mesh.all_faces()) {
        skewness[i0] = skewness[i0].max(s);
        skewness[i1] = skewness[i1].max(s);
    }
    writer.add_cell_data("skewness", 1, skewness.iter().copied());

    writer.export("ellipse.vtu")?;
    mesh.boundary().0.vtu_writer().export("ellipse_bdy.vtu")?;

    // Analytical geometry
    mesh.compute_topology();
    let topo = mesh.get_topology()?;
    let geom = Simple3dGeometry::new(topo.clone());
    geom.check(topo)?;

    // Check the geometry
    check_geom(&mesh, &geom);

    geom.project_vertices(&mut mesh);

    // Compute the implied metric
    mesh.compute_vertex_to_elems();
    mesh.compute_face_to_elems();
    mesh.compute_volumes();
    let implied_metric = mesh.implied_metric()?;

    // Compute the boundary layer metric
    let h_min = 5e-3;
    let h_max = 1.0;
    let eps = 0.2;

    let bl_metric = get_bl_metric(&mesh, &geom, h_min, h_max, eps);

    // Intersect the two metrics
    let metric = implied_metric
        .iter()
        .zip(bl_metric.iter())
        .map(|(m0, m1)| m0.intersect(m1))
        .collect::<Vec<_>>();

    let debug = false;
    let mut params = RemesherParams {
        debug,
        ..RemesherParams::default()
    };
    params.set_max_angle(30.0);

    let n_part = 1;
    let now = Instant::now();
    let mut mesh = if n_part == 1 {
        let mut remesher = Remesher::new(&mesh, &metric, &geom)?;
        remesher.remesh(&params, &geom)?;
        remesher.check()?;
        remesher.to_mesh(true)
    } else {
        #[cfg(feature = "metis")]
        let part = PartitionType::MetisRecursive(n_part);
        #[cfg(not(feature = "metis"))]
        let part = PartitionType::Hilbert(n_part);
        let mut dd = ParallelRemesher::new(mesh, part)?;
        dd.set_debug(debug);
        let dd_params = ParallelRemesherParams::new(2, 2, 10000);
        let (mesh, stats, _) = dd.remesh(&metric, &geom, params, &dd_params)?;
        stats.print_summary();
        mesh
    };

    mesh.compute_face_to_elems();
    mesh.check_simple()?;

    println!(
        "Remeshing done in {}s with {n_part} partitions",
        now.elapsed().as_secs_f32()
    );
    mesh.vtu_writer().export("ellipse_remeshed.vtu")?;
    let bdy = mesh.boundary().0;
    bdy.vtu_writer().export("ellipse_remeshed_bdy.vtu")?;

    let max_angle = geom.max_normal_angle(&mesh);
    println!("max angle : {max_angle:.1}");

    Ok(())
}
