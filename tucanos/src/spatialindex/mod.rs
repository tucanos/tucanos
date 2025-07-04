use crate::{
    Idx,
    mesh::{Elem, Point, SimplexMesh},
};

mod parry;
pub type DefaultPointIndex<const D: usize> = KdTreePointIndex<D>;
pub type DefaultObjectIndex<const D: usize> = parry::ObjectIndex<D>;

pub trait PointIndex<const D: usize> {
    fn new<E: Elem>(mesh: &SimplexMesh<D, E>) -> Self;
    fn nearest_vert(&self, pt: &Point<D>) -> (Idx, f64);
}

pub trait ObjectIndex<const D: usize> {
    fn new<E: Elem>(mesh: &SimplexMesh<D, E>) -> Self;
    fn nearest_elem(&self, pt: &Point<D>) -> Idx;
    fn project(&self, pt: &Point<D>) -> (f64, Point<D>);
}

pub struct KdTreePointIndex<const D: usize> {
    tree: kdtree::KdTree<f64, usize, [f64; D]>,
}

impl<const D: usize> PointIndex<D> for KdTreePointIndex<D> {
    fn new<E: Elem>(mesh: &SimplexMesh<D, E>) -> Self {
        assert!(mesh.n_verts() > 0);
        let mut tree = kdtree::KdTree::new(D);
        for (i, pt) in mesh.verts().enumerate() {
            tree.add(pt.as_slice().try_into().unwrap(), i).unwrap();
        }
        Self { tree }
    }

    fn nearest_vert(&self, pt: &Point<D>) -> (Idx, f64) {
        let r = self
            .tree
            .nearest(pt.as_slice(), 1, &kdtree::distance::squared_euclidean)
            .unwrap()[0];
        (*r.1 as Idx, r.0)
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::SVector;
    use rand::{Rng, SeedableRng, rngs::StdRng};
    use std::{f64::consts::PI, time::Instant};

    use crate::{
        Tag,
        mesh::{Edge, Elem, Point, SimplexMesh, Triangle},
        spatialindex::{DefaultObjectIndex, ObjectIndex},
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

        let pt = Point::<3>::new(-11.134_680_738_954_373, 10.371_256_484_784_858, 0.);
        assert_eq!(tree.nearest_elem(&pt), 76);
        let (d, _) = tree.project(&pt);
        assert!(f64::abs(d) < 2e-11, "{d} vs 0");

        let pt = Point::<3>::new(-360., -105., 0.);
        assert_eq!(tree.nearest_elem(&pt), 109);

        let pt = Point::<3>::new(41.905, -7.933, 0.);
        assert_eq!(tree.nearest_elem(&pt), 194);

        let pt = Point::<3>::new(977.405_622_304_933_2, -193.219_725_123_763_82, 0.);
        assert_eq!(tree.nearest_elem(&pt), 193);
        let (d, _) = tree.project(&pt);
        assert!(f64::abs(d) < 1e-12, "{d} vs 0");

        let pt = Point::<3>::new(732.254_535_699_460_3, 628.314_474_637_604_1, 0.);
        assert_eq!(tree.nearest_elem(&pt), 23);
        let (d, _) = tree.project(&pt);
        assert!(f64::abs(d) < 1e-12, "{d} vs 0");

        let pt = Point::<3>::new(41.905_036_870_164_33, -7.932_967_693_525_678, 0.);
        assert_eq!(tree.nearest_elem(&pt), 194);
        let (d, _) = tree.project(&pt);
        assert!(f64::abs(d) < 4e-12, "{d} vs 0");

        let mut rng = StdRng::seed_from_u64(0);
        let mut worst = (0., None);
        let start = Instant::now();
        let num_proj = 10000;
        for _ in 0..num_proj {
            let tmp = SVector::<f64, 3>::from_fn(|_, _| rng.random());
            let theta = 2.0 * PI * tmp[0];
            let r = r_in + tmp[1] * (r_out * 0.999 - r_in);
            let x = r * f64::cos(theta);
            let y = r * f64::sin(theta);
            let z = r_out * (tmp[2] - 0.5);
            let pt = Point::<3>::new(x, y, z);
            let (d, pt_proj) = tree.project(&pt);
            assert!(f64::abs(d - z.abs()) < 1e-12);
            for (i, v) in [x, y, 0.].iter().enumerate() {
                let d = f64::abs(pt_proj[i] - v);
                if d > worst.0 {
                    worst.0 = d;
                    worst.1 = Some(pt_proj);
                    println!("{pt:?} -> {pt_proj:?}, {d:e}");
                }
                assert!(f64::abs(pt_proj[i] - v) < 3e-10, "{} != {}", pt_proj[i], v);
            }
        }
        println!("Time by projection: {:?}", start.elapsed() / num_proj);
    }
}
