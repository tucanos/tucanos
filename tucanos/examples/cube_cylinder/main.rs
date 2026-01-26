use nalgebra::Matrix3;
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
    process::Command,
    time::Instant,
};
use tmesh::{
    Result, Tag, Vert3d, init_log,
    mesh::{Mesh, Mesh3d},
};
use tucanos::{
    TopoTag,
    geometry::Geometry,
    mesh::{MeshTopology, Topology},
    metric::{AnisoMetric, AnisoMetric3d, Metric, MetricField},
    remesher::{Remesher, RemesherParams, Stats},
};

/// .geo file to generate the input mesh with gmsh:
const GEO_FILE: &str = r#"
SetFactory("OpenCASCADE");
Box(1) = {0, 0, 0, 1, 1, 1};
Cylinder(2) = {0., 0, -1, 0, 0, 3, 0.5, 2*Pi};
BooleanDifference{ Volume{1}; Delete; }{ Volume{2}; Delete; }

Physical Surface("xmin", 16) = {1};
Physical Surface("xmax", 17) = {6};
Physical Surface("ymin", 18) = {7};
Physical Surface("ymax", 19) = {3};
Physical Surface("zmin", 20) = {4};
Physical Surface("zmax", 21) = {2};
Physical Surface("cylinder", 22) = {5};

Physical Volume("fluid", 23) = {1};

Characteristic Length { 1,4, 7, 10 } = 0.1;
Characteristic Length { 2, 3, 5, 6, 8, 9 } = 0.5;
"#;

const XMIN_TAG: Tag = 1;
const XMAX_TAG: Tag = 6;
const YMIN_TAG: Tag = 7;
const YMAX_TAG: Tag = 3;
const ZMIN_TAG: Tag = 4;
const ZMAX_TAG: Tag = 2;
const CYLINDER_TAG: Tag = 5;

struct CubeCylinderGeometry {
    topo: Topology,
}

impl CubeCylinderGeometry {
    const fn new(topo: Topology) -> Self {
        Self { topo }
    }
}

impl Geometry<3> for CubeCylinderGeometry {
    fn check(&self, _topo: &Topology) -> Result<()> {
        Ok(())
    }

    fn project(&self, pt: &mut Vert3d, tag: &TopoTag) -> f64 {
        let old = *pt;

        if tag.1 < 0 {
            return 0.0;
        }

        *pt = match tag.0 {
            2 => match tag.1 {
                XMIN_TAG => Vert3d::new(0.0, old[1].clamp(0.5, 1.0), old[2].clamp(0.0, 1.0)),
                XMAX_TAG => Vert3d::new(1.0, old[1].clamp(0.0, 1.0), old[2].clamp(0.0, 1.0)),
                YMIN_TAG => Vert3d::new(old[0].clamp(0.5, 1.0), 0.0, old[2].clamp(0.0, 1.0)),
                YMAX_TAG => Vert3d::new(old[0].clamp(0.0, 1.0), 1.0, old[2].clamp(0.0, 1.0)),
                ZMIN_TAG => {
                    let r = old[0].hypot(old[1]);
                    let f = r.max(0.5) / r;
                    let x = f * old[0];
                    let y = f * old[1];
                    Vert3d::new(x.clamp(0.0, 1.0), y.clamp(0.0, 1.0), 0.0)
                }
                ZMAX_TAG => {
                    let r = old[0].hypot(old[1]);
                    let f = r.max(0.5) / r;
                    let x = f * old[0];
                    let y = f * old[1];
                    Vert3d::new(x.clamp(0.0, 1.0), y.clamp(0.0, 1.0), 1.0)
                }
                CYLINDER_TAG => {
                    let r = old[0].hypot(old[1]);
                    let f = 0.5 / r;
                    let x = f * old[0];
                    let y = f * old[1];
                    Vert3d::new(x.clamp(0.0, 1.0), y.clamp(0.0, 1.0), old[2].clamp(0.0, 1.0))
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
                // TODO: improve
                self.project(pt, &(2, parents[0]));
                self.project(pt, &(2, parents[1]));
                *pt
            }
            0 => old,
            _ => unreachable!("{}", tag.0),
        };
        (*pt - old).norm()
    }

    fn angle(&self, pt: &Vert3d, n: &Vert3d, tag: &TopoTag) -> f64 {
        let n_ref = match tag.1 {
            XMIN_TAG => Vert3d::new(-1.0, 0.0, 0.0),
            XMAX_TAG => Vert3d::new(1.0, 0.0, 0.0),
            YMIN_TAG => Vert3d::new(0.0, -1.0, 0.0),
            YMAX_TAG => Vert3d::new(0.0, 1.0, 0.0),
            ZMIN_TAG => Vert3d::new(0.0, 0.0, -1.0),
            ZMAX_TAG => Vert3d::new(0.0, 0.0, 1.0),
            CYLINDER_TAG => {
                let r = pt[0].hypot(pt[1]);
                Vert3d::new(-pt[0] / r, -pt[1] / r, 0.0)
            }
            _ => unreachable!(),
        };
        let cos_a = n.dot(&n_ref).clamp(-1.0, 1.0);
        f64::acos(cos_a).to_degrees()
    }
}

#[allow(dead_code)]
enum BenchmarkType {
    Linear,
    Polar1,
    Polar2,
}

fn get_metric(mesh: &Mesh3d, btype: &BenchmarkType, step: Option<f64>) -> Vec<AnisoMetric3d> {
    let mut m = mesh
        .verts()
        .map(|v| match btype {
            BenchmarkType::Linear => {
                let h_x = 0.1;
                let h_y = 0.1;
                let h0 = 0.001;
                let h_z = h0 + 2.0 * (0.1 - h0) * (v[2] - 0.5).abs();
                AnisoMetric3d::from_sizes(
                    &Vert3d::new(h_x, 0.0, 0.0),
                    &Vert3d::new(0.0, h_y, 0.0),
                    &Vert3d::new(0.0, 0.0, h_z),
                )
            }
            BenchmarkType::Polar1 => {
                let r = v[0].hypot(v[1]);
                let theta = v[1].atan2(v[0]);
                let h_z = 0.1;
                let h_t = 0.1;
                let h0 = 0.001;
                let h_r = h0 + 2.0 * (0.1 - h0) * (r - 0.5).abs();
                let a = Matrix3::new(
                    theta.cos(),
                    theta.sin(),
                    0.0,
                    -theta.sin(),
                    theta.cos(),
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                );
                let b = Matrix3::new(
                    1.0 / h_r / h_r,
                    0.0,
                    0.0,
                    0.0,
                    1.0 / h_t / h_t,
                    0.0,
                    0.0,
                    0.0,
                    1.0 / h_z / h_z,
                );
                AnisoMetric3d::from_mat(a.transpose() * b * a)
            }
            BenchmarkType::Polar2 => todo!(),
        })
        .collect::<Vec<_>>();

    if let Some(step) = step {
        let implied = MetricField::implied_metric(mesh);
        for (m_i, m_other_i) in m.iter_mut().zip(implied.metric()) {
            m_i.control_step(m_other_i, step);
        }
    }
    m
}

fn write_dat_file(arr: &[f64], fname: &str) -> Result<()> {
    let file = File::create(fname)?;
    let mut writer = BufWriter::new(file);

    for val in arr {
        writeln!(writer, "{val}")?;
    }

    writer.flush()?;
    Ok(())
}

fn generate_mesh(fname: &str) -> Result<Mesh3d> {
    std::fs::write("cube-cylinder.geo", GEO_FILE)?;

    let output = Command::new("gmsh")
        .arg("cube-cylinder.geo")
        .arg("-3")
        .arg("-o")
        .arg(fname)
        .output()?;

    assert!(
        output.status.success(),
        "gmsh error: {}",
        String::from_utf8(output.stderr).unwrap()
    );

    Mesh3d::from_meshb(fname)
}

fn main() -> Result<()> {
    init_log("info");

    let btype = BenchmarkType::Polar1;
    let n_steps = 5;
    let step = Some(4.0);

    // let label = "default";
    // let params = RemesherParams::default();

    // let label = "old_4";
    // let params = RemesherParams::new(25.0, 4);

    let label = "ordered_4";
    let params = RemesherParams::new_ordered(25.0, 4);

    let fname = "cube-cylinder.mesh";
    // let fname = "cube-cylinder_old_4_1.meshb";
    let fname = Path::new(fname);

    let mut mesh = if fname.exists() {
        Mesh3d::from_meshb(fname.to_str().unwrap())?
    } else {
        generate_mesh(fname.to_str().unwrap())?
    };

    println!(
        "Initial mesh: {} verts, {} elems, {} faces",
        mesh.n_verts(),
        mesh.n_elems(),
        mesh.n_faces()
    );

    // Fix face orientation
    mesh.fix()?;

    // Check the mesh
    mesh.check(&mesh.all_faces())?;

    let fname = format!("cube-cylinder_{label}_0.vtu");
    mesh.write_vtk(fname)?;

    let start = Instant::now();

    for i_step in 0..n_steps {
        let topo = MeshTopology::new(&mesh);
        let geom = CubeCylinderGeometry::new(topo.topo().clone());
        let max_angle = geom.max_normal_angle(&mesh);
        let d = geom.project_vertices(&mut mesh, &topo);
        assert!(d < 1e-12);
        println!("geometry: d = {d:.2e}, max_angle = {max_angle:.2}");

        println!("Step {i_step}");
        let metric = get_metric(&mesh, &btype, step);
        let now = Instant::now();
        let fname = format!("cube-cylinder_{label}_{i_step}.meshb");
        println!("  write {fname}");
        mesh.write_meshb(&fname)?;

        let mut remesher = Remesher::new(&mesh, &topo, &metric, &geom)?;
        remesher.remesh(&params, &geom)?;
        remesher.check()?;
        mesh = remesher.to_mesh(true);

        println!("  remeshing done in {}s ", now.elapsed().as_secs_f32());
        println!(
            "  {} verts, {} elems, {} faces",
            mesh.n_verts(),
            mesh.n_elems(),
            mesh.n_faces()
        );
        mesh.check(&mesh.all_faces())?;

        let fname = format!("cube-cylinder_{label}_{}.vtu", i_step + 1);
        println!("  write {fname}");
        mesh.write_vtk(fname)?;

        let bins = [0.0, 0.25, 0.5, 0.75, 1.0];
        let stats_q = Stats::new(remesher.qualities_iter(), &bins);
        println!("  q: {stats_q}");
        let stats_l = Stats::new(remesher.lengths_iter(), &[0.5_f64.sqrt(), 2.0_f64.sqrt()]);
        println!("  l: {stats_l}");
        if i_step == n_steps - 1 {
            let fname = format!("cube-cylinder_{label}_q.dat");
            println!("  write {fname}");
            write_dat_file(&remesher.qualities(), &fname)?;
            let fname = format!("cube-cylinder_{label}_l.dat");
            println!("  write {fname}");
            write_dat_file(&remesher.lengths(), &fname)?;
        }
    }

    println!("Done in {}s ", start.elapsed().as_secs_f32());

    Ok(())
}
