use crate::{
    mesh::{Point, SimplexMesh},
    topo_elems::Elem,
    Idx,
};

#[path = "libol.rs"]
mod libol;
pub type DefaultPointIndex = libol::Octree;
pub type DefaultObjectIndex = libol::Octree;

pub trait PointIndex<const D: usize> {
    fn new<E: Elem>(mesh: &SimplexMesh<D, E>) -> Self;
    fn nearest_vertex(&self, pt: &Point<D>) -> Idx;
}

pub trait ObjectIndex<const D: usize> {
    fn new<E: Elem>(mesh: &SimplexMesh<D, E>) -> Self;
    fn nearest(&self, pt: &Point<D>) -> Idx;
    fn project(&self, pt: &Point<D>) -> (f64, Point<D>);
}

#[cfg(test)]
mod tests {
    use nalgebra::SVector;
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use std::f64::consts::PI;

    use crate::{
        mesh::{Point, SimplexMesh},
        spatialindex::{DefaultObjectIndex, ObjectIndex},
        topo_elems::{Edge, Elem, Triangle},
        Tag,
    };

    #[test]
    fn test_edgs() {
        let coords = vec![
            Point::<2>::new(0., 0.),
            Point::<2>::new(1., 0.),
            Point::<2>::new(1., 1.),
            Point::<2>::new(0., 1.),
        ];
        let elems: Vec<_> = vec![
            Edge::from_slice(&[0, 1]),
            Edge::from_slice(&[1, 2]),
            Edge::from_slice(&[2, 3]),
            Edge::from_slice(&[3, 0]),
        ];
        let etags: Vec<Tag> = vec![1, 2, 3, 4];
        let faces: Vec<_> = Vec::new();
        let ftags: Vec<Tag> = Vec::new();

        let mesh = SimplexMesh::new(coords, elems, etags, faces, ftags);

        let tree = DefaultObjectIndex::new(&mesh);

        let pt = Point::<2>::new(1.5, 0.25);
        let (dist, p) = tree.project(&pt);
        assert!(f64::abs(dist - 0.5) < 1e-12);
        assert!((p - Point::<2>::new(1., 0.25)).norm() < 1e-12);
    }

    #[test]
    fn test_proj() {
        let r_in = 1.;
        let r_out = 1000.;
        let n = 100;
        let dim = 3;

        let mut coords = Vec::with_capacity(2 * (n as usize) * dim);
        for i in 0..n {
            let theta = 2.0 * PI * f64::from(i) / f64::from(n);
            coords.push(Point::<3>::new(
                r_in * f64::cos(theta),
                r_in * f64::sin(theta),
                0.0,
            ));
            coords.push(Point::<3>::new(
                r_out * f64::cos(theta),
                r_out * f64::sin(theta),
                0.0,
            ));
        }

        let mut tris = Vec::with_capacity(2 * (n as usize));
        for i in 0..n - 1 {
            tris.push(Triangle::from_slice(&[2 * i, 2 * i + 1, 2 * i + 2]));
            tris.push(Triangle::from_slice(&[2 * i + 2, 2 * i + 1, 2 * i + 3]));
        }
        tris.push(Triangle::from_slice(&[2 * n - 2, 2 * n - 1, 0]));
        tris.push(Triangle::from_slice(&[0, 2 * n - 1, 1]));

        let tri_tags = vec![1; 2 * (n as usize)];

        let msh = SimplexMesh::<3, Triangle>::new(coords, tris, tri_tags, Vec::new(), Vec::new());
        // msh.write_vtk("dbg.vtu", None, None);

        let tree = DefaultObjectIndex::new(&msh);
        let pt = Point::<3>::new(-360., -105., 0.);
        assert_eq!(tree.nearest(&pt), 109);

        let pt = Point::<3>::new(41.905, -7.933, 0.);
        assert_eq!(tree.nearest(&pt), 194);

        let pt = Point::<3>::new(977.405_622_304_933_2, -193.219_725_123_763_82, 0.);
        assert_eq!(tree.nearest(&pt), 193);
        let (d, _) = tree.project(&pt);
        assert!(f64::abs(d) < 1e-12, "{d} vs 0");

        let pt = Point::<3>::new(732.254_535_699_460_3, 628.314_474_637_604_1, 0.);
        assert_eq!(tree.nearest(&pt), 23);
        let (d, _) = tree.project(&pt);
        assert!(f64::abs(d) < 1e-12, "{d} vs 0");

        let pt = Point::<3>::new(41.905_036_870_164_33, -7.932_967_693_525_678, 0.);
        assert_eq!(tree.nearest(&pt), 194);
        let (d, _) = tree.project(&pt);
        assert!(f64::abs(d) < 1e-12, "{d} vs 0");

        let mut rng = StdRng::seed_from_u64(0);
        for _ in 0..10000 {
            let tmp = SVector::<f64, 3>::from_fn(|_, _| rng.gen());
            let theta = 2.0 * PI * tmp[0];
            let r = r_in + tmp[1] * (r_out * 0.999 - r_in);
            let x = r * f64::cos(theta);
            let y = r * f64::sin(theta);
            let z = r_out * (tmp[2] - 0.5);
            let pt = Point::<3>::new(x, y, z);
            let (d, pt_proj) = tree.project(&pt);
            println!("{pt:?} -> {pt_proj:?}, {d}");
            assert!(f64::abs(d - z.abs()) < 1e-12);
            assert!(f64::abs(pt_proj[0] - x) < 1e-12);
            assert!(f64::abs(pt_proj[1] - y) < 1e-12);
            assert!(f64::abs(pt_proj[2] - 0.0) < 1e-12);
        }
    }
}
