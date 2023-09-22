use crate::geometry::Geometry;
use crate::mesh::{Point, SimplexMesh};
use crate::topo_elems::{Edge, Tetrahedron, Triangle};
use crate::{Dim, Error, Result, TopoTag};
use std::fs::File;
use std::io::Write;

/// Build a 2d mesh of a square with 2 triangles tagged differently
#[must_use]
pub fn test_mesh_2d() -> SimplexMesh<2, Triangle> {
    let coords = vec![
        Point::<2>::new(0., 0.),
        Point::<2>::new(1., 0.),
        Point::<2>::new(1., 1.),
        Point::<2>::new(0., 1.),
    ];
    let elems = vec![Triangle::new(0, 1, 2), Triangle::new(0, 2, 3)];
    let etags = vec![1, 2];
    let faces = vec![
        Edge::new(0, 1),
        Edge::new(1, 2),
        Edge::new(2, 3),
        Edge::new(3, 0),
    ];
    let ftags = vec![1, 2, 3, 4];

    SimplexMesh::new(coords, elems, etags, faces, ftags)
}

/// Build a 2d mesh of a square with 2 triangles tagged differently
/// Boundaries are not defined
#[must_use]
pub fn test_mesh_2d_nobdy() -> SimplexMesh<2, Triangle> {
    let coords = vec![
        Point::<2>::new(0., 0.),
        Point::<2>::new(1., 0.),
        Point::<2>::new(1., 1.),
        Point::<2>::new(0., 1.),
    ];
    let elems = vec![Triangle::new(0, 1, 2), Triangle::new(0, 2, 3)];
    let etags = vec![1, 2];
    let faces = vec![];
    let ftags = vec![];

    SimplexMesh::new(coords, elems, etags, faces, ftags)
}

/// Build a 2d mesh with 2 flat triangles with the same tag
#[must_use]
pub fn test_mesh_2d_two_tris() -> SimplexMesh<2, Triangle> {
    let coords = vec![
        Point::<2>::new(0., 0.),
        Point::<2>::new(1., 0.5),
        Point::<2>::new(0., 1.),
        Point::<2>::new(-1., 0.5),
    ];
    let elems = vec![Triangle::new(0, 1, 3), Triangle::new(3, 1, 2)];
    let etags = vec![1, 1];
    let faces = vec![
        Edge::new(0, 1),
        Edge::new(1, 2),
        Edge::new(2, 3),
        Edge::new(3, 0),
    ];
    let ftags = vec![1, 1, 2, 2];

    SimplexMesh::new(coords, elems, etags, faces, ftags)
}

/// Gaussian size field in 2d
#[must_use]
pub fn h_2d(p: &Point<2>) -> f64 {
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
pub fn test_mesh_moon_2d() -> SimplexMesh<2, Triangle> {
    let coords = vec![
        Point::<2>::new(-1., 0.),
        Point::<2>::new(0., 0.5),
        Point::<2>::new(1., 0.),
        Point::<2>::new(0., 1.),
    ];
    let elems = vec![Triangle::new(0, 1, 3), Triangle::new(1, 2, 3)];
    let etags = vec![1, 1];
    let faces = vec![
        Edge::new(0, 1),
        Edge::new(1, 2),
        Edge::new(2, 3),
        Edge::new(3, 0),
    ];
    let ftags = vec![1, 1, 2, 2];

    SimplexMesh::new(coords, elems, etags, faces, ftags)
}

/// Analytical geometry of a 2d domain bounded by two circle arcs
pub struct GeomHalfCircle2d();

impl Geometry<2> for GeomHalfCircle2d {
    fn check(&self, topo: &crate::topology::Topology) -> Result<()> {
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

    fn project(&self, pt: &mut Point<2>, tag: &crate::TopoTag) -> f64 {
        assert!(tag.0 < 2);
        let p: Point<2> = *pt;
        match *tag {
            (1, 1) => {
                let p = Point::<2>::new(0.0, -0.75);
                let r = (*pt - p).norm();
                (*pt) = p + 1.25 * (*pt - p) / r;
            }
            (1, 2) => {
                let r = pt.norm();
                (*pt) *= 1. / r;
            }
            (0, 1) => {
                if pt[0] < 0. {
                    pt[0] = -1.;
                    pt[1] = 0.;
                } else {
                    pt[0] = 1.;
                    pt[1] = 0.;
                }
            }
            _ => unreachable!(),
        }
        (*pt - p).norm()
    }

    fn angle(&self, _pt: &mut Point<2>, _n: &Point<2>, _tag: &TopoTag) -> f64 {
        0.0
    }
}

/// Build a 3d mesh with 1 single tetrahedron
#[must_use]
pub fn test_mesh_3d_single_tet() -> SimplexMesh<3, Tetrahedron> {
    let coords = vec![
        Point::<3>::new(0., 0., 0.),
        Point::<3>::new(1., 0., 0.),
        Point::<3>::new(0., 1., 0.),
        Point::<3>::new(0., 0., 1.),
    ];
    let elems = vec![Tetrahedron::new(0, 1, 2, 3)];
    let etags = vec![1];
    let faces = vec![
        Triangle::new(0, 1, 2),
        Triangle::new(0, 1, 3),
        Triangle::new(1, 2, 3),
        Triangle::new(2, 0, 3),
    ];
    let ftags = vec![1, 2, 3, 4];

    SimplexMesh::new(coords, elems, etags, faces, ftags)
}

/// Build a 3d mesh with two flat tetrahedra
#[must_use]
pub fn test_mesh_3d_two_tets() -> SimplexMesh<3, Tetrahedron> {
    let coords = vec![
        Point::<3>::new(0., 0., 0.),
        Point::<3>::new(1., 0., 0.),
        Point::<3>::new(0.5, 0.1, 0.),
        Point::<3>::new(0., 0., 1.),
        Point::<3>::new(0.5, -0.1, 0.),
    ];
    let elems = vec![Tetrahedron::new(0, 1, 2, 3), Tetrahedron::new(0, 4, 1, 3)];
    let etags = vec![1, 1];
    let faces = vec![
        Triangle::new(0, 1, 2),
        Triangle::new(1, 2, 3),
        Triangle::new(2, 0, 3),
        Triangle::new(0, 4, 1),
        Triangle::new(4, 1, 3),
        Triangle::new(0, 4, 3),
    ];
    let ftags = vec![1, 2, 3, 1, 2, 3];

    SimplexMesh::new(coords, elems, etags, faces, ftags)
}

/// Build a 3d mesh of a cube with 5 tetrahedra
#[must_use]
pub fn test_mesh_3d() -> SimplexMesh<3, Tetrahedron> {
    let coords = vec![
        Point::<3>::new(0., 0., 0.),
        Point::<3>::new(1., 0., 0.),
        Point::<3>::new(1., 1., 0.),
        Point::<3>::new(0., 1., 0.),
        Point::<3>::new(0., 0., 1.),
        Point::<3>::new(1., 0., 1.),
        Point::<3>::new(1., 1., 1.),
        Point::<3>::new(0., 1., 1.),
    ];
    let elems = vec![
        Tetrahedron::new(0, 1, 2, 5),
        Tetrahedron::new(0, 2, 7, 5),
        Tetrahedron::new(0, 2, 3, 7),
        Tetrahedron::new(0, 5, 7, 4),
        Tetrahedron::new(2, 7, 5, 6),
    ];
    let etags = vec![1, 1, 1, 1, 1];
    let faces = vec![
        Triangle::new(0, 1, 2),
        Triangle::new(0, 2, 3),
        Triangle::new(5, 6, 7),
        Triangle::new(5, 7, 4),
        Triangle::new(0, 1, 5),
        Triangle::new(0, 5, 4),
        Triangle::new(2, 6, 7),
        Triangle::new(2, 7, 3),
        Triangle::new(1, 2, 5),
        Triangle::new(2, 6, 5),
        Triangle::new(0, 3, 7),
        Triangle::new(0, 7, 4),
    ];
    let ftags = vec![1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6];

    SimplexMesh::new(coords, elems, etags, faces, ftags)
}

/// Gaussian size field in 3d
#[must_use]
pub fn h_3d(p: &Point<3>) -> f64 {
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
