use env_logger::Env;
use nalgebra::SMatrix;
use rustc_hash::FxHashMap;
use std::{f64::consts::PI, time::Instant};
use tucanos::{
    geom_elems::GElem,
    geometry::Geometry,
    mesh::{Point, SimplexMesh},
    mesh_partition::PartitionType,
    metric::{AnisoMetric3d, Metric},
    parallel::{ParallelRemesher, ParallelRemeshingParams},
    remesher::{Remesher, RemesherParams},
    topo_elems::Tetrahedron,
    topology::Topology,
    Result, Tag, TopoTag,
};

pub fn init_log(level: &str) {
    env_logger::Builder::from_env(Env::default().default_filter_or(level))
        .format_timestamp(None)
        .init();
}

const WINGTIP_TAG: Tag = 1;
const SYMMETRY_TAG: Tag = 2;
const FARFIELD_TAG: Tag = 3;
const WING_1_TAG: Tag = 4;
const WING_2_TAG: Tag = 5;
const FACE_TAGS: [Tag; 5] = [
    SYMMETRY_TAG,
    FARFIELD_TAG,
    WING_1_TAG,
    WING_2_TAG,
    WINGTIP_TAG,
];
const FACE_NAMES: [&str; 5] = ["Symmetry", "Farfield", "Wing_1", "Wing_2", "Wingtip"];

struct EllipseProjection {
    a: f64,
    b: f64,
}

impl EllipseProjection {
    const TOL: f64 = 1e-12;

    const fn new(a: f64, b: f64) -> Self {
        Self { a, b }
    }

    fn f(&self, p: f64, x: f64, y: f64) -> f64 {
        2.0 * self.a * (x - self.a * p.cos()) * p.sin()
            - 2.0 * self.b * (y - self.b * p.sin()) * p.cos()
    }

    fn solve(&self, x: f64, y: f64) -> f64 {
        let (mut start, mut end) = (0.0, 0.5 * PI);
        let (mut f_start, mut f_end) = (self.f(start, x, y), self.f(end, x, y));

        while (end - start).abs() > Self::TOL {
            let mid = 0.5 * (start + end);
            let f_mid = self.f(mid, x, y);
            if f_mid * f_start > 0.0 {
                start = mid;
                f_start = f_mid;
            } else if f_mid * f_end > 0.0 {
                end = mid;
                f_end = f_mid;
            } else if f_mid.abs() < f64::EPSILON {
                break;
            } else {
                unreachable!("{x} {y} {start} {f_start} {end} {f_end} {f_mid}");
            }
        }
        0.5 * (start + end)
    }

    pub fn project(&self, x: f64, y: f64) -> (f64, f64) {
        let sgn_x = x.signum();
        let x = x.abs();
        let sgn_y = y.signum();
        let y = y.abs();

        let t = self.solve(x, y);
        (self.a * t.cos() * sgn_x, self.b * t.sin() * sgn_y)
    }

    pub fn normal(&self, x: f64, y: f64) -> (f64, f64) {
        let sgn_x = x.signum();
        let x = x.abs();
        let sgn_y = y.signum();
        let y = y.abs();

        let t = self.solve(x, y);
        let x1 = self.a * t.cos() * sgn_x;
        let y1 = self.b * t.sin() * sgn_y;

        let nx = 1. / self.a.powi(2) * x1;
        let ny = 1. / self.b.powi(2) * y1;
        let n = nx.hypot(ny);
        (nx / n, ny / n)
    }
}

struct Simple3dGeometry(Topology);

impl Simple3dGeometry {
    fn normal(pt: &Point<3>, tag: TopoTag) -> Point<3> {
        let n = match tag.0 {
            2 => match tag.1 {
                SYMMETRY_TAG => Point::<3>::new(0.0, 0.0, -1.0),
                FARFIELD_TAG => {
                    let r = pt.norm();
                    1. / r * *pt
                }
                WING_1_TAG => {
                    let ellipse = EllipseProjection::new(0.5, 0.05);
                    let (nx, ny) = ellipse.normal(pt[0], pt[1]);
                    -Point::<3>::new(nx, ny, 0.0)
                }
                WING_2_TAG => unreachable!("Tag has been removed"),
                WINGTIP_TAG => {
                    let x = pt[0];
                    let y = pt[1];
                    let z = pt[2] - 2.0;
                    if z > 0.0 {
                        let r = y.hypot(z);
                        let t = f64::atan2(z, y);
                        let ellipse = EllipseProjection::new(0.5, 0.05);
                        let (nx, nr) = ellipse.normal(x, r);
                        -Point::<3>::new(nx, nr * t.cos(), nr * t.sin())
                    } else {
                        let ellipse = EllipseProjection::new(0.5, 0.05);
                        let (nx, ny) = ellipse.normal(x, y);
                        -Point::<3>::new(nx, ny, 0.0)
                    }
                }
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

    fn project(&self, pt: &mut Point<3>, tag: &TopoTag) -> f64 {
        let old = *pt;

        if tag.1 < 0 {
            return 0.0;
        }

        *pt = match tag.0 {
            2 => match tag.1 {
                SYMMETRY_TAG => Point::<3>::new(old[0], old[1], 0.0),
                FARFIELD_TAG => {
                    let r = old.norm();
                    (50.0 / r) * old
                }
                WING_1_TAG => {
                    let ellipse = EllipseProjection::new(0.5, 0.05);
                    let (x, y) = ellipse.project(old[0], old[1]);
                    Point::<3>::new(x, y, old[2].min(2.0))
                }
                WING_2_TAG => unreachable!(),
                WINGTIP_TAG => {
                    let x = old[0];
                    let y = old[1];
                    let z = old[2] - 2.0;
                    if z > 0.0 {
                        let r = y.hypot(z);
                        let t = f64::atan2(z, y);
                        let ellipse = EllipseProjection::new(0.5, 0.05);
                        let (x, r) = ellipse.project(x, r);
                        Point::<3>::new(x, r * t.cos(), 2.0 + r * t.sin())
                    } else {
                        let ellipse = EllipseProjection::new(0.5, 0.05);
                        let (x, y) = ellipse.project(x, old[1]);
                        Point::<3>::new(x, y, 2.0)
                    }
                }

                _ => unreachable!(),
            },
            1 => {
                let node = self.0.get(*tag).unwrap_or_else(|| panic!("tag = {tag:?}"));
                assert_eq!(node.parents.len(), 2);
                let mut parents = node.parents.iter().copied().collect::<Vec<_>>();
                parents.sort_unstable();
                let x = old[0];
                let y = old[1];
                let z = old[2];
                if parents[0] == WINGTIP_TAG {
                    assert!(parents[1] == WING_1_TAG || parents[1] == WING_2_TAG);
                    let ellipse = EllipseProjection::new(0.5, 0.05);
                    let (x, y) = ellipse.project(x, y);
                    Point::<3>::new(x, y, 2.0)
                } else if parents[0] == SYMMETRY_TAG {
                    if parents[1] == FARFIELD_TAG {
                        let r = x.hypot(y);
                        50. / r * Point::<3>::new(x, y, 0.0)
                    } else {
                        assert!(parents[1] == WING_1_TAG || parents[1] == WING_2_TAG);
                        let ellipse = EllipseProjection::new(0.5, 0.05);
                        let (x, y) = ellipse.project(x, y);
                        Point::<3>::new(x, y, 0.0)
                    }
                } else if parents[0] == WING_1_TAG {
                    assert_eq!(parents[1], WING_2_TAG);
                    if x > 0.0 {
                        Point::<3>::new(0.5, 0.0, z.min(2.0))
                    } else {
                        Point::<3>::new(-0.5, 0.0, z.min(2.0))
                    }
                } else {
                    unreachable!("{parents:?}");
                }
            }
            0 => old,
            _ => unreachable!("{}", tag.0),
        };
        (*pt - old).norm()
    }

    fn angle(&self, pt: &Point<3>, n: &Point<3>, tag: &TopoTag) -> f64 {
        let n_ref = Self::normal(pt, *tag);
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
            println!("  max. distance: {:.2e}", max_dist.get(&tag).unwrap());
            println!("  max. angle: {:.2e}", max_angle.get(&tag).unwrap());
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
            let mut d = geom.project(&mut tmp, &(2, WING_1_TAG));
            let mut tag = WING_1_TAG;

            let mut tmp = pt;
            let d1 = geom.project(&mut tmp, &(2, WINGTIP_TAG));
            if d1 < d {
                d = d1;
                tag = WINGTIP_TAG;
            }

            let n = Simple3dGeometry::normal(&pt, (2, tag));
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
    init_log("warn");

    // Load the mesh
    let mut mesh = SimplexMesh::<3, Tetrahedron>::read_meshb("data/simple3d.meshb")?;

    // Merge tags WING_1_TAG and WING_2_TAG to make analytical projection easier
    mesh.mut_ftags().for_each(|t| {
        if *t == WING_2_TAG {
            *t = WING_1_TAG;
        }
    });

    // Fix face orientation
    mesh.add_boundary_faces();

    // Check the mesh
    mesh.check()?;

    // Save the input mesh in .vtu format
    mesh.write_vtk("simple3d.vtu", None, None)?;
    mesh.boundary()
        .0
        .write_vtk("simple3d_bdy.vtu", None, None)?;

    // Analytical geometry
    mesh.compute_topology();
    let topo = mesh.get_topology()?;
    let geom = Simple3dGeometry(topo.clone());
    geom.check(topo)?;

    // Check the geometry
    if false {
        check_geom(&mesh, &geom);
    }

    geom.project_vertices(&mut mesh);

    // Compute the implied metric
    mesh.compute_vertex_to_elems();
    mesh.compute_volumes();
    let implied_metric = mesh.implied_metric()?;

    // Compute the boundary layer metric
    let h_min = 1e-3;
    let h_max = 10.0;
    let eps = 0.2;

    let bl_metric = get_bl_metric(&mesh, &geom, h_min, h_max, eps);

    // Intersect the two metrics
    let metric = implied_metric
        .iter()
        .zip(bl_metric.iter())
        .map(|(m0, m1)| m0.intersect(m1))
        .collect::<Vec<_>>();

    let debug = false;
    let params = RemesherParams {
        two_steps: false,
        num_iter: 2,
        split_max_iter: 1,
        collapse_max_iter: 1,
        max_angle: 25.0,
        debug,
        ..RemesherParams::default()
    };

    let n_part = 8;
    let now = Instant::now();
    let mut mesh = if n_part == 1 {
        let mut remesher = Remesher::new(&mesh, &metric, &geom)?;
        remesher.remesh(params, &geom)?;
        remesher.check()?;
        remesher.to_mesh(true)
    } else {
        let mut dd = ParallelRemesher::new(mesh, PartitionType::Scotch(n_part))?;
        dd.set_debug(debug);
        let dd_params = ParallelRemeshingParams::new(2, 2, 10000);
        let (mesh, stats, _) = dd.remesh(&metric, &geom, params, dd_params)?;
        stats.print_summary();
        mesh
    };
    println!(
        "Remeshing done in {}s with {n_part} partitions",
        now.elapsed().as_secs_f32()
    );
    mesh.write_vtk("simple3d_remeshed.vtu", None, None)?;
    let bdy = mesh.boundary().0;
    bdy.write_vtk("simple3d_remeshed_bdy.vtu", None, None)?;

    let max_angle = geom.max_normal_angle(&mesh);
    println!("max angle : {max_angle}");
    assert!(max_angle < 25.0);

    mesh.compute_face_to_elems();
    mesh.check()?;

    Ok(())
}
