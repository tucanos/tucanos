use crate::{
    curvature::{compute_curvature_tensor, fix_curvature},
    geom_elems::GElem,
    mesh::{Point, SimplexMesh},
    topo_elems::{Elem, Triangle},
    topology::Topology,
    Dim, Error, Result, Tag, TopoTag,
};
use rustc_hash::FxHashSet;
use std::{collections::HashMap, f64::consts::PI, hash::BuildHasherDefault};

/// Representation of a D-dimensional geometry
pub trait Geometry<const D: usize> {
    /// Check that the geometry is consistent with a Topology
    fn check(&self, topo: &Topology) -> Result<()>;

    /// Project a vertex, associated with a given `TopoTag`, onto the geometry
    /// The distance between the original vertex and its projection if returned
    fn project(&self, pt: &mut Point<D>, tag: &TopoTag) -> f64;

    /// Compute the angle between a vector n and the normal at the projection of pt onto the geometry
    fn angle(&self, pt: &mut Point<D>, n: &Point<D>, tag: &TopoTag) -> f64;
}

/// No geometric model
pub struct NoGeometry<const D: usize>();

impl<const D: usize> Geometry<D> for NoGeometry<D> {
    fn check(&self, _topo: &Topology) -> Result<()> {
        Ok(())
    }

    fn project(&self, _pt: &mut Point<D>, tag: &TopoTag) -> f64 {
        assert!(tag.0 < D as Dim);
        0.
    }

    fn angle(&self, _pt: &mut Point<D>, _n: &Point<D>, tag: &TopoTag) -> f64 {
        assert_eq!(tag.0, D as Dim - 1);
        0.
    }
}

/// Piecewise linear (stl-like) representation of a geometry
pub struct LinearGeometry<const D: usize, E: Elem> {
    mesh: SimplexMesh<D, E>,
    u: Option<Vec<Point<D>>>,
    v: Option<Vec<Point<D>>>,
}

impl<const D: usize, E: Elem> LinearGeometry<D, E> {
    /// Create a `LinearGeometry` from a `SimplexMesh`
    /// compute_octree should be
    pub fn new(mut mesh: SimplexMesh<D, E>) -> Result<Self> {
        if mesh.tree.is_none() {
            mesh.compute_octree();
        }

        Ok(Self {
            mesh,
            u: None,
            v: None,
        })
    }
}

impl LinearGeometry<3, Triangle> {
    pub fn compute_curvature(&mut self) {
        self.mesh.add_boundary_faces();
        if self.mesh.elem_to_elems.is_none() {
            self.mesh.compute_elem_to_elems();
        }

        let (mut u, mut v) = compute_curvature_tensor(&self.mesh);
        fix_curvature(&self.mesh, &mut u, &mut v).unwrap();

        self.u = Some(u);
        self.v = Some(v);
    }

    pub fn curvature(&self, pt: &Point<3>) -> Result<(Point<3>, Point<3>)> {
        if self.u.is_none() {
            return Err(Error::from(
                "LinearGeometry<3, Triangle>: compute_curvature not called",
            ));
        }

        let tree = self.mesh.tree.as_ref().unwrap();
        let i_elem = tree.nearest(pt) as usize;
        let u = self.u.as_ref().unwrap();
        let v = self.v.as_ref().unwrap();

        Ok((u[i_elem], v[i_elem]))
    }

    pub fn write_curvature(&self, fname: &str) -> Result<()> {
        // Export the curvature
        let u1 = self
            .u
            .as_ref()
            .unwrap()
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        let v1 = self
            .v
            .as_ref()
            .unwrap()
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        let mut data = HashMap::new();
        data.insert(String::from("u"), u1.as_slice());
        data.insert(String::from("v"), v1.as_slice());

        self.mesh.write_vtk(fname, None, Some(data))?;

        Ok(())
    }
}

impl<const D: usize, E: Elem> Geometry<D> for LinearGeometry<D, E> {
    fn check(&self, topo: &Topology) -> Result<()> {
        // Check that the geometry and topo have the same D-1 dimensional tags

        let mut tags = FxHashSet::with_hasher(BuildHasherDefault::default());
        tags.extend(self.mesh.etags.iter().copied());
        let tags = tags.iter().copied().collect::<Vec<Tag>>();
        let topo_tags = topo.tags(E::DIM as Dim);

        if tags.len() != topo.ntags(E::DIM as Dim) {
            return Err(Error::from(&format!(
                "LinearGeometry: invalid # of tags (mesh: {tags:?}, topo: {topo_tags:?})"
            )));
        }

        for tag in tags.iter().copied() {
            if topo.get((E::DIM as Dim, tag)).is_none() {
                return Err(Error::from(&format!("LinearGeometry: tag {tag:?} not found in topo (mesh: {tags:?}, topo: {topo_tags:?})")));
            }
        }

        Ok(())
    }

    fn project(&self, pt: &mut Point<D>, tag: &TopoTag) -> f64 {
        assert!(tag.0 < D as Dim);
        // TODO: check that the tag is consistent
        let tree = self.mesh.tree.as_ref().unwrap();
        let (dist, p) = tree.project(pt);
        *pt = p;
        dist
    }

    fn angle(&self, pt: &mut Point<D>, n: &Point<D>, tag: &TopoTag) -> f64 {
        assert_eq!(tag.0, D as Dim - 1);
        // TODO: check that the tag is consistent
        let tree = self.mesh.tree.as_ref().unwrap();
        let idx = tree.nearest(pt);
        let n_ref = self.mesh.gelem(idx).normal();
        f64::acos(n.dot(&n_ref)) * 180. / PI
    }
}

#[cfg(test)]
mod tests {
    use super::{Geometry, LinearGeometry};
    use crate::mesh::Point;
    use crate::mesh_stl::read_stl;
    use crate::test_meshes::write_stl_file;
    use crate::Result;
    use std::fs::remove_file;

    #[test]
    fn test_stl() -> Result<()> {
        write_stl_file("cube2.stl")?;
        let mut geom = read_stl("cube2.stl");
        geom.compute_octree();
        remove_file("cube2.stl")?;
        let geom = LinearGeometry::new(geom)?;

        let mut p = Point::<3>::new(2., 0.5, 0.5);
        let d = geom.project(&mut p, &(2, 1));
        assert!(f64::abs(d - 1.) < 1e-12);
        assert!((p - Point::<3>::new(1., 0.5, 0.5)).norm() < 1e-12);

        let mut p = Point::<3>::new(0.5, 0.75, 0.5);
        let d = geom.project(&mut p, &(2, 1));
        assert!(f64::abs(d - 0.25) < 1e-12);
        assert!((p - Point::<3>::new(0.5, 1., 0.5)).norm() < 1e-12);

        Ok(())
    }
}
