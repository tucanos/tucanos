//! Indices to efficiently locate the nearest vertices or elements
use crate::Vertex;

mod parry;

/// Point index based on `kdtree`
pub struct PointIndex<const D: usize> {
    tree: kdtree::KdTree<f64, usize, [f64; D]>,
}

impl<const D: usize> PointIndex<D> {
    /// Create a PointIndex from vertices
    pub fn new<I: ExactSizeIterator<Item = Vertex<D>>>(verts: I) -> Self {
        assert!(verts.len() > 0);
        let mut tree = kdtree::KdTree::new(D);
        for (i, pt) in verts.enumerate() {
            tree.add(pt.as_slice().try_into().unwrap(), i).unwrap();
        }
        Self { tree }
    }

    /// Get the index of the nearest point & the distance
    #[must_use]
    pub fn nearest_vert(&self, pt: &Vertex<D>) -> (usize, f64) {
        let r = self
            .tree
            .nearest(pt.as_slice(), 1, &kdtree::distance::squared_euclidean)
            .unwrap()[0];
        (*r.1, r.0)
    }
}

pub use parry::ObjectIndex;

#[cfg(test)]
mod tests {
    use nalgebra::SVector;
    use rand::{Rng, SeedableRng, rngs::StdRng};
    use std::f64::consts::PI;

    use crate::{
        Vert2d, Vert3d,
        mesh::{BoundaryMesh2d, BoundaryMesh3d, Mesh},
        spatialindex::ObjectIndex,
    };

    #[test]
    fn test_edgs() {
        let coords = vec![
            Vert2d::new(0., 0.),
            Vert2d::new(1., 0.),
            Vert2d::new(1., 1.),
            Vert2d::new(0., 1.),
        ];
        let elems = vec![[0, 1], [1, 2], [2, 3], [3, 0]];
        let etags = vec![1, 2, 3, 4];
        let faces = Vec::new();
        let ftags = Vec::new();

        let mesh = BoundaryMesh2d::new(&coords, &elems, &etags, &faces, &ftags);

        let tree = ObjectIndex::new(&mesh);

        let pt = Vert2d::new(1.5, 0.25);
        let (dist, p) = tree.project(&pt);
        assert!(f64::abs(dist - 0.5) < 1e-12);
        assert!((p - Vert2d::new(1., 0.25)).norm() < 1e-12);
    }

    #[test]
    fn test_proj() {
        let r_in = 1.;
        let r_out = 1000.;
        let n = 100;
        let dim = 3;

        let mut coords = Vec::with_capacity(2 * n * dim);
        for i in 0..n {
            let theta = 2.0 * PI * i as f64 / n as f64;
            coords.push(Vert3d::new(
                r_in * f64::cos(theta),
                r_in * f64::sin(theta),
                0.0,
            ));
            coords.push(Vert3d::new(
                r_out * f64::cos(theta),
                r_out * f64::sin(theta),
                0.0,
            ));
        }

        let mut tris = Vec::with_capacity(2 * n);
        for i in 0..n - 1 {
            tris.push([2 * i, 2 * i + 1, 2 * i + 2]);
            tris.push([2 * i + 2, 2 * i + 1, 2 * i + 3]);
        }
        tris.push([2 * n - 2, 2 * n - 1, 0]);
        tris.push([0, 2 * n - 1, 1]);

        let tri_tags = vec![1; 2 * n];

        let msh = BoundaryMesh3d::new(&coords, &tris, &tri_tags, &Vec::new(), &Vec::new());
        // msh.write_vtk("dbg.vtu", None, None);

        let tree = ObjectIndex::new(&msh);

        let pt = Vert3d::new(-11.134_680_738_954_373, 10.371_256_484_784_858, 0.);
        assert_eq!(tree.nearest_elem(&pt), 76);
        let (d, _) = tree.project(&pt);
        assert!(f64::abs(d) < 2e-11, "{d} vs 0");

        let pt = Vert3d::new(-360., -105., 0.);
        assert_eq!(tree.nearest_elem(&pt), 109);

        let pt = Vert3d::new(41.905, -7.933, 0.);
        assert_eq!(tree.nearest_elem(&pt), 194);

        let pt = Vert3d::new(977.405_622_304_933_2, -193.219_725_123_763_82, 0.);
        assert_eq!(tree.nearest_elem(&pt), 193);
        let (d, _) = tree.project(&pt);
        assert!(f64::abs(d) < 1e-12, "{d} vs 0");

        let pt = Vert3d::new(732.254_535_699_460_3, 628.314_474_637_604_1, 0.);
        assert_eq!(tree.nearest_elem(&pt), 23);
        let (d, _) = tree.project(&pt);
        assert!(f64::abs(d) < 1e-12, "{d} vs 0");

        let pt = Vert3d::new(41.905_036_870_164_33, -7.932_967_693_525_678, 0.);
        assert_eq!(tree.nearest_elem(&pt), 194);
        let (d, _) = tree.project(&pt);
        assert!(f64::abs(d) < 4e-12, "{d} vs 0");

        let mut rng = StdRng::seed_from_u64(0);
        let mut worst = (0., None);
        // let start = Instant::now();
        let num_proj = 10000;
        for _ in 0..num_proj {
            let tmp = SVector::<f64, 3>::from_fn(|_, _| rng.random());
            let theta = 2.0 * PI * tmp[0];
            let r = r_in + tmp[1] * (r_out * 0.999 - r_in);
            let x = r * f64::cos(theta);
            let y = r * f64::sin(theta);
            let z = r_out * (tmp[2] - 0.5);
            let pt = Vert3d::new(x, y, z);
            let (d, pt_proj) = tree.project(&pt);
            assert!(f64::abs(d - z.abs()) < 1e-12);
            for (i, v) in [x, y, 0.].iter().enumerate() {
                let d = f64::abs(pt_proj[i] - v);
                if d > worst.0 {
                    worst.0 = d;
                    worst.1 = Some(pt_proj);
                    // println!("{pt:?} -> {pt_proj:?}, {d:e}");
                }
                assert!(f64::abs(pt_proj[i] - v) < 3e-10, "{} != {}", pt_proj[i], v);
            }
        }
        // println!("Time by projection: {:?}", start.elapsed() / num_proj);
    }
}
