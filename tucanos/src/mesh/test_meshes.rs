use tmesh::mesh::{Edge, Mesh, MutMesh, Prism, Simplex, Tetrahedron, Triangle, pri2tets};
use tmesh::{Vert2d, Vert3d};

use crate::geometry::Geometry;
use crate::mesh::SimplexMesh;
use crate::{Dim, Error, Result, TopoTag};
use std::fs::File;
use std::io::Write;
use std::iter::{once, repeat_n};

/// Build a 2d mesh of a square with 2 triangles tagged differently
/// WARNING: the mesh tags are not valid as the diagonal (0, 2) is between
/// two different element tags not is not tagged
#[must_use]
pub fn test_mesh_2d() -> SimplexMesh<u32, 2, Triangle<u32>> {
    let coords = vec![
        Vert2d::new(0., 0.),
        Vert2d::new(1., 0.),
        Vert2d::new(1., 1.),
        Vert2d::new(0., 1.),
    ];
    let elems = vec![Triangle::from([0, 1, 2]), Triangle::from([0, 2, 3])];
    let etags = vec![1, 2];
    let faces = vec![
        Edge::from([0, 1]),
        Edge::from([1, 2]),
        Edge::from([2, 3]),
        Edge::from([3, 0]),
    ];
    let ftags = vec![1, 2, 3, 4];

    SimplexMesh::new_with_vec(coords, elems, etags, faces, ftags)
}

/// Build a 2d mesh of a square with 2 triangles tagged differently
/// Boundaries are not defined
#[must_use]
pub fn test_mesh_2d_nobdy() -> SimplexMesh<u32, 2, Triangle<u32>> {
    let coords = vec![
        Vert2d::new(0., 0.),
        Vert2d::new(1., 0.),
        Vert2d::new(1., 1.),
        Vert2d::new(0., 1.),
    ];
    let elems = vec![Triangle::from([0, 1, 2]), Triangle::from([0, 2, 3])];
    let etags = vec![1, 2];
    let faces = vec![];
    let ftags = vec![];

    SimplexMesh::new_with_vec(coords, elems, etags, faces, ftags)
}

/// Build a 2d mesh with 2 flat triangles with the same tag
#[must_use]
pub fn test_mesh_2d_two_tris() -> SimplexMesh<u32, 2, Triangle<u32>> {
    let coords = vec![
        Vert2d::new(0., 0.),
        Vert2d::new(1., 0.5),
        Vert2d::new(0., 1.),
        Vert2d::new(-1., 0.5),
    ];
    let elems = vec![Triangle::from([0, 1, 3]), Triangle::from([3, 1, 2])];
    let etags = vec![1, 1];
    let faces = vec![
        Edge::from([0, 1]),
        Edge::from([1, 2]),
        Edge::from([2, 3]),
        Edge::from([3, 0]),
    ];
    let ftags = vec![1, 1, 2, 2];

    SimplexMesh::new_with_vec(coords, elems, etags, faces, ftags)
}

/// Gaussian size field in 2d
#[must_use]
pub fn h_2d(p: &Vert2d) -> f64 {
    let x = p[0];
    let y = p[1];
    let hmin = 0.001;
    let hmax = 0.3;
    let sigma: f64 = 0.25;
    hmin + (hmax - hmin)
        * (1.0 - f64::exp(-((x - 0.5).powi(2) + (y - 0.35).powi(2)) / sigma.powi(2)))
}

/// Build a 2d mesh with 2 triangles corresponding to the geometry `GeomHalfCircle2d`
#[must_use]
pub fn test_mesh_moon_2d() -> SimplexMesh<u32, 2, Triangle<u32>> {
    let coords = vec![
        Vert2d::new(-1., 0.),
        Vert2d::new(0., 0.5),
        Vert2d::new(1., 0.),
        Vert2d::new(0., 1.),
    ];
    let elems = vec![Triangle::from([0, 1, 3]), Triangle::from([1, 2, 3])];
    let etags = vec![1, 1];
    let faces = vec![
        Edge::from([0, 1]),
        Edge::from([1, 2]),
        Edge::from([2, 3]),
        Edge::from([3, 0]),
    ];
    let ftags = vec![1, 1, 2, 2];

    SimplexMesh::new_with_vec(coords, elems, etags, faces, ftags)
}

/// Analytical geometry of a 2d domain bounded by two circle arcs
pub struct GeomHalfCircle2d();

impl Geometry<2> for GeomHalfCircle2d {
    fn check(&self, topo: &crate::mesh::Topology) -> Result<()> {
        let ntags = [1, 2, 1];
        let tags: [TopoTag; 4] = [(2, 1), (1, 1), (1, 2), (0, 1)];

        for (dim, n) in ntags.iter().enumerate() {
            if *n != topo.ntags(dim as Dim) {
                return Err(Error::from(&format!("Invalid # of tags for dim {dim}")));
            }
        }
        for tag in tags {
            if topo.get(tag).is_none() {
                return Err(Error::from(&format!("Tag {tag:?} not found in topo")));
            }
        }
        Ok(())
    }

    fn project(&self, pt: &mut Vert2d, tag: &TopoTag) -> f64 {
        assert!(tag.0 < 2);
        let p: Vert2d = *pt;
        match *tag {
            (1, 1) => {
                let p = Vert2d::new(0.0, -0.75);
                let r = (*pt - p).norm();
                (*pt) = p + 1.25 * (*pt - p) / r;
            }
            (1, 2) => {
                let r = pt.norm();
                (*pt) *= 1. / r;
            }
            (0, 1) => {
                pt[0] = if pt[0] < 0. { -1. } else { 1. };
                pt[1] = 0.;
            }
            _ => unreachable!(),
        }
        (*pt - p).norm()
    }

    fn angle(&self, _pt: &Vert2d, _n: &Vert2d, _tag: &TopoTag) -> f64 {
        0.0
    }
}

/// Build a 3d mesh with 1 single tetrahedron
#[must_use]
pub fn test_mesh_3d_single_tet() -> SimplexMesh<u32, 3, Tetrahedron<u32>> {
    let coords = vec![
        Vert3d::new(0., 0., 0.),
        Vert3d::new(1., 0., 0.),
        Vert3d::new(0., 1., 0.),
        Vert3d::new(0., 0., 1.),
    ];
    let elems = vec![Tetrahedron::from([0, 1, 2, 3])];
    let etags = vec![1];
    let faces = vec![
        Triangle::from([0, 1, 2]),
        Triangle::from([0, 1, 3]),
        Triangle::from([1, 2, 3]),
        Triangle::from([2, 0, 3]),
    ];
    let ftags = vec![1, 2, 3, 4];

    SimplexMesh::new_with_vec(coords, elems, etags, faces, ftags)
}

/// Build a 3d mesh with two flat tetrahedra
#[must_use]
pub fn test_mesh_3d_two_tets() -> SimplexMesh<u32, 3, Tetrahedron<u32>> {
    let coords = vec![
        Vert3d::new(0., 0., 0.),
        Vert3d::new(1., 0., 0.),
        Vert3d::new(0.5, 0.1, 0.),
        Vert3d::new(0., 0., 1.),
        Vert3d::new(0.5, -0.1, 0.),
    ];
    let elems = vec![
        Tetrahedron::from([0, 1, 2, 3]),
        Tetrahedron::from([0, 4, 1, 3]),
    ];
    let etags = vec![1, 1];
    let faces = vec![
        Triangle::from([0, 1, 2]),
        Triangle::from([1, 2, 3]),
        Triangle::from([2, 0, 3]),
        Triangle::from([0, 4, 1]),
        Triangle::from([4, 1, 3]),
        Triangle::from([0, 4, 3]),
    ];
    let ftags = vec![1, 2, 3, 1, 2, 3];

    SimplexMesh::new_with_vec(coords, elems, etags, faces, ftags)
}

/// Build a 3d mesh of a cube with 5 tetrahedra
#[must_use]
pub fn test_mesh_3d() -> SimplexMesh<u32, 3, Tetrahedron<u32>> {
    let coords = vec![
        Vert3d::new(0., 0., 0.),
        Vert3d::new(1., 0., 0.),
        Vert3d::new(1., 1., 0.),
        Vert3d::new(0., 1., 0.),
        Vert3d::new(0., 0., 1.),
        Vert3d::new(1., 0., 1.),
        Vert3d::new(1., 1., 1.),
        Vert3d::new(0., 1., 1.),
    ];
    let elems = vec![
        Tetrahedron::from([0, 1, 2, 5]),
        Tetrahedron::from([0, 2, 7, 5]),
        Tetrahedron::from([0, 2, 3, 7]),
        Tetrahedron::from([0, 5, 7, 4]),
        Tetrahedron::from([2, 7, 5, 6]),
    ];
    let etags = vec![1, 1, 1, 1, 1];
    let faces = vec![
        Triangle::from([0, 2, 1]),
        Triangle::from([0, 3, 2]),
        Triangle::from([5, 6, 7]),
        Triangle::from([5, 7, 4]),
        Triangle::from([0, 1, 5]),
        Triangle::from([0, 5, 4]),
        Triangle::from([2, 7, 6]),
        Triangle::from([2, 3, 7]),
        Triangle::from([1, 2, 5]),
        Triangle::from([2, 6, 5]),
        Triangle::from([0, 7, 3]),
        Triangle::from([0, 4, 7]),
    ];
    let ftags = vec![1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6];

    SimplexMesh::new_with_vec(coords, elems, etags, faces, ftags)
}

/// Gaussian size field in 3d
#[must_use]
pub fn h_3d(p: &Vert3d) -> f64 {
    let x = p[0];
    let y = p[1];
    let z = p[2];
    let hmin = 0.03;
    let hmax = 0.3;
    let sigma: f64 = 0.25;
    let var = -((x - 0.5).powi(2) + (y - 0.35).powi(2) + (z - 0.35).powi(2)) / sigma.powi(2);
    hmin + (hmax - hmin) * (1.0 - f64::exp(var))
}

/// Write a sample .stl file for a cube
pub fn write_stl_file(fname: &str) -> Result<()> {
    let cube_stl = "solid 
facet normal -1 0 0
  outer loop
    vertex 0 0 1
    vertex 0 1 0
    vertex 0 0 0
  endloop
endfacet
facet normal -1 0 0
  outer loop
    vertex 0 0 1
    vertex 0 1 1
    vertex 0 1 0
  endloop
endfacet
facet normal 1 0 0
  outer loop
    vertex 1 0 1
    vertex 1 0 0
    vertex 1 1 0
  endloop
endfacet
facet normal 1 -0 0
  outer loop
    vertex 1 0 1
    vertex 1 1 0
    vertex 1 1 1
  endloop
endfacet
facet normal 0 -1 -0
  outer loop
    vertex 1 0 1
    vertex 0 0 0
    vertex 1 0 0
  endloop
endfacet
facet normal -0 -1 0
  outer loop
    vertex 1 0 1
    vertex 0 0 1
    vertex 0 0 0
  endloop
endfacet
facet normal 0 1 0
  outer loop
    vertex 1 1 1
    vertex 1 1 0
    vertex 0 1 0
  endloop
endfacet
facet normal 0 1 0
  outer loop
    vertex 1 1 1
    vertex 0 1 0
    vertex 0 1 1
  endloop
endfacet
facet normal -0 0 -1
  outer loop
    vertex 1 1 0
    vertex 0 0 0
    vertex 0 1 0
  endloop
endfacet
facet normal 0 -0 -1
  outer loop
    vertex 1 1 0
    vertex 1 0 0
    vertex 0 0 0
  endloop
endfacet
facet normal 0 0 1
  outer loop
    vertex 1 1 1
    vertex 0 1 1
    vertex 0 0 1
  endloop
endfacet
facet normal 0 0 1
  outer loop
    vertex 1 1 1
    vertex 0 0 1
    vertex 1 0 1
  endloop
endfacet
endsolid Created by Gmsh";

    let mut file = File::create(fname)?;
    file.write_all(cube_stl.as_bytes())?;

    Ok(())
}

pub struct SphereGeometry;

impl Geometry<3> for SphereGeometry {
    fn check(&self, _topo: &super::Topology) -> Result<()> {
        Ok(())
    }

    fn project(&self, pt: &mut Vert3d, _tag: &TopoTag) -> f64 {
        let nrm = pt.norm();
        *pt /= nrm;
        nrm - 1.0
    }

    fn angle(&self, pt: &Vert3d, n: &Vert3d, _tag: &TopoTag) -> f64 {
        let n_ref = pt.normalize();
        let cos_a = n.dot(&n_ref).clamp(-1.0, 1.0);
        f64::acos(cos_a).to_degrees()
    }
}

#[must_use]
pub fn sphere_mesh_surf(level: usize) -> SimplexMesh<u32, 3, Triangle<u32>> {
    let verts = vec![
        Vert3d::new(0., 0., 1.),
        Vert3d::new(1., 0., 0.),
        Vert3d::new(0., 1., 0.),
        Vert3d::new(-1., 0., 0.),
        Vert3d::new(0., -1., 0.),
        Vert3d::new(0., 0., -1.),
    ];

    let elems = vec![
        Triangle::from([0, 1, 2]),
        Triangle::from([0, 2, 3]),
        Triangle::from([0, 3, 4]),
        Triangle::from([0, 4, 1]),
        Triangle::from([5, 2, 1]),
        Triangle::from([5, 3, 2]),
        Triangle::from([5, 4, 3]),
        Triangle::from([5, 1, 4]),
    ];

    let etags = vec![1; elems.len()];
    let faces = Vec::new();
    let ftags = Vec::new();

    let mut grid = SimplexMesh::new_with_vec(verts, elems, etags, faces, ftags);

    let geom = SphereGeometry;
    for _ in 0..level {
        grid = grid.split();
        grid.verts_mut().for_each(|v| {
            geom.project(v, &(2, 1));
        });
    }
    grid.compute_topology();
    grid
}

#[must_use]
pub fn sphere_mesh(level: usize) -> SimplexMesh<u32, 3, Tetrahedron<u32>> {
    let verts = vec![
        Vert3d::new(0., 0., 1.),
        Vert3d::new(1., 0., 0.),
        Vert3d::new(0., 1., 0.),
        Vert3d::new(-1., 0., 0.),
        Vert3d::new(0., -1., 0.),
        Vert3d::new(0., 0., -1.),
        Vert3d::new(0., 0., 0.),
    ];

    let elems = vec![
        Tetrahedron::from([6, 0, 1, 2]),
        Tetrahedron::from([6, 0, 2, 3]),
        Tetrahedron::from([6, 0, 3, 4]),
        Tetrahedron::from([6, 0, 4, 1]),
        Tetrahedron::from([6, 5, 2, 1]),
        Tetrahedron::from([6, 5, 3, 2]),
        Tetrahedron::from([6, 5, 4, 3]),
        Tetrahedron::from([6, 5, 1, 4]),
    ];

    let etags = vec![1; elems.len()];

    let faces = vec![
        Triangle::from([0, 1, 2]),
        Triangle::from([0, 2, 3]),
        Triangle::from([0, 3, 4]),
        Triangle::from([0, 4, 1]),
        Triangle::from([5, 2, 1]),
        Triangle::from([5, 3, 2]),
        Triangle::from([5, 4, 3]),
        Triangle::from([5, 1, 4]),
    ];

    let ftags = vec![1; faces.len()];

    let mut grid = SimplexMesh::new_with_vec(verts, elems, etags, faces, ftags);

    let geom = SphereGeometry;
    for _ in 0..level {
        grid = grid.split();
        grid.compute_topology();
        geom.project_vertices(&mut grid);
    }
    grid
}

pub struct ConcentricCircles;

impl Geometry<2> for ConcentricCircles {
    fn check(&self, _topo: &super::Topology) -> Result<()> {
        Ok(())
    }

    fn project(&self, pt: &mut Vert2d, tag: &TopoTag) -> f64 {
        assert_eq!(tag.0, 1);
        let nrm = pt.norm();
        *pt *= 1.0 / nrm;
        match tag.1 {
            1 => {
                *pt *= 0.5;
                (nrm - 0.5).abs()
            }
            2 => (nrm - 1.0).abs(),
            _ => unreachable!(),
        }
    }

    fn angle(&self, pt: &Vert2d, n: &Vert2d, tag: &TopoTag) -> f64 {
        assert_eq!(tag.0, 1);
        let mut n_ref = pt.normalize();
        match tag.1 {
            1 => {
                n_ref *= -1.0;
            }
            2 => {}
            _ => unreachable!(),
        }
        let cos_a = n.dot(&n_ref).clamp(-1.0, 1.0);
        f64::acos(cos_a).to_degrees()
    }
}

#[must_use]
pub fn concentric_circles_mesh(nr: usize) -> SimplexMesh<u32, 2, Triangle<u32>> {
    assert!(nr >= 2);
    let dr = 0.5 / (nr as f64 - 1.0);
    let mut verts = Vec::new();
    for i in 0..nr {
        let r = 0.5 + (i as f64) * dr;
        verts.push(Vert2d::new(r, 0.0));
        verts.push(Vert2d::new(0.0, r));
        verts.push(Vert2d::new(-r, 0.0));
        verts.push(Vert2d::new(0.0, -r));
    }

    let idx = |i: usize, j: usize| (4 * i + j).try_into().unwrap();
    let mut elems = Vec::new();
    let mut etags = Vec::new();
    for i in 0..nr - 1 {
        for (j0, j1) in [(0, 1), (1, 2), (2, 3), (3, 0)] {
            elems.push(Triangle::from([idx(i, j0), idx(i + 1, j0), idx(i, j1)]));
            etags.push(1);
            elems.push(Triangle::from([idx(i, j1), idx(i + 1, j0), idx(i + 1, j1)]));
            etags.push(1);
        }
    }

    let mut faces = Vec::new();
    let mut ftags = Vec::new();
    for (j0, j1) in [(0, 1), (1, 2), (2, 3), (3, 0)] {
        faces.push(Edge::from([idx(0, j0), idx(0, j1)]));
        ftags.push(1);
        faces.push(Edge::from([idx(nr - 1, j0), idx(nr - 1, j1)]));
        ftags.push(2);
    }

    SimplexMesh::new_with_vec(verts, elems, etags, faces, ftags)
}

pub struct ConcentricSpheres;

impl Geometry<3> for ConcentricSpheres {
    fn check(&self, _topo: &super::Topology) -> Result<()> {
        Ok(())
    }

    fn project(&self, pt: &mut Vert3d, tag: &TopoTag) -> f64 {
        assert_eq!(tag.0, 2);
        let nrm = pt.norm();
        *pt *= 1.0 / nrm;
        match tag.1 {
            1 => {
                *pt *= 0.5;
                (nrm - 0.5).abs()
            }
            2 => (nrm - 1.0).abs(),
            _ => unreachable!(),
        }
    }

    fn angle(&self, pt: &Vert3d, n: &Vert3d, tag: &TopoTag) -> f64 {
        assert_eq!(tag.0, 1);
        let mut n_ref = pt.normalize();
        match tag.1 {
            1 => {
                n_ref *= -1.0;
            }
            2 => {}
            _ => unreachable!(),
        }
        let cos_a = n.dot(&n_ref).clamp(-1.0, 1.0);
        f64::acos(cos_a).to_degrees()
    }
}

#[must_use]
pub fn concentric_spheres_mesh(nr: u32) -> SimplexMesh<u32, 3, Tetrahedron<u32>> {
    assert!(nr >= 2);
    let dr = 0.5 / (nr as f64 - 1.0);

    let surf = sphere_mesh_surf(0);
    let n = surf.n_verts();

    let mut res = SimplexMesh::empty();

    for v in surf.verts() {
        let nrm = v.norm();
        res.add_verts(once(0.5 / nrm * v));
    }

    for i in 0..nr - 1 {
        let offset = i * n;
        for v in surf.verts() {
            let nrm = v.norm();
            res.add_verts(once((0.5 + (i as f64 + 1.0) * dr / nrm) * v));
        }
        for f in surf.elems() {
            let pri = Prism::from([
                f[0] + offset,
                f[1] + offset,
                f[2] + offset,
                f[0] + offset + n,
                f[1] + offset + n,
                f[2] + offset + n,
            ]);
            let tets = pri2tets(&pri);
            res.add_elems(tets.iter().copied(), repeat_n(1, 3));
        }
    }

    for mut f in surf.elems() {
        f.invert();
        res.add_faces(once(Triangle::from([f[0], f[1], f[2]])), once(1));
        f.invert();
        let offset = (nr - 1) * n;
        res.add_faces(
            once(Triangle::from([
                f[0] + offset,
                f[1] + offset,
                f[2] + offset,
            ])),
            once(2),
        );
    }

    res
}
