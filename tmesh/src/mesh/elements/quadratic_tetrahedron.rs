use nalgebra::{Const, LU, SMatrix, SVector};

use crate::{
    Vertex,
    mesh::{
        Edge, GEdge, GSimplex, Idx, Mesh, QuadraticGTriangle, QuadraticTriangle, Simplex,
        elements::ho_simplex::HOType,
    },
};
use std::fmt::Debug;
use std::ops::Index;

/// Edge
#[derive(Default, Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct QuadraticTetrahedron<T: Idx>(pub(crate) [T; 10]);

impl<T: Idx> QuadraticTetrahedron<T> {
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        i0: usize,
        i1: usize,
        i2: usize,
        i3: usize,
        i4: usize,
        i5: usize,
        i6: usize,
        i7: usize,
        i8: usize,
        i9: usize,
    ) -> Self {
        Self([
            i0.try_into().unwrap(),
            i1.try_into().unwrap(),
            i2.try_into().unwrap(),
            i3.try_into().unwrap(),
            i4.try_into().unwrap(),
            i5.try_into().unwrap(),
            i6.try_into().unwrap(),
            i7.try_into().unwrap(),
            i8.try_into().unwrap(),
            i9.try_into().unwrap(),
        ])
    }
}

impl<T: Idx> IntoIterator for QuadraticTetrahedron<T> {
    type Item = usize;
    type IntoIter = std::iter::Map<std::array::IntoIter<T, 10>, fn(T) -> usize>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter().map(|x| x.try_into().unwrap())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct QuadraticGTetrahedron<const D: usize>([Vertex<D>; 10], HOType);

impl<const D: usize> QuadraticGTetrahedron<D> {
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        v0: &Vertex<D>,
        v1: &Vertex<D>,
        v2: &Vertex<D>,
        v3: &Vertex<D>,
        v4: &Vertex<D>,
        v5: &Vertex<D>,
        v6: &Vertex<D>,
        v7: &Vertex<D>,
        v8: &Vertex<D>,
        v9: &Vertex<D>,
        etype: HOType,
    ) -> Self {
        Self([*v0, *v1, *v2, *v3, *v4, *v5, *v6, *v7, *v8, *v9], etype)
    }

    // fn linear(&self) -> GTetrahedron<D> {
    //     GTetrahedron::new(&self[0], &self[1], &self[2], &self[3])
    // }

    fn mapping(&self, bcoords: &[f64; 4]) -> Vertex<D> {
        let [u, v, w, t] = bcoords;
        2.0 * u * (u - 0.5) * self[0]
            + 2.0 * v * (v - 0.5) * self[1]
            + 2.0 * w * (w - 0.5) * self[2]
            + 2.0 * t * (t - 0.5) * self[3]
            + 4.0 * u * v * self[4]
            + 4.0 * v * w * self[5]
            + 4.0 * u * w * self[6]
            + 4.0 * u * t * self[7]
            + 4.0 * v * t * self[8]
            + 4.0 * w * t * self[9]
    }

    fn jac_mapping(&self, bcoords: &[f64; 4]) -> [Vertex<D>; 4] {
        let [u, v, w, t] = bcoords;
        [
            (4.0 * u - 1.0) * self[0] + 4.0 * v * self[4] + 4.0 * w * self[6] + 4.0 * t * self[7],
            (4.0 * v - 1.0) * self[1] + 4.0 * u * self[4] + 4.0 * w * self[5] + 4.0 * t * self[8],
            (4.0 * w - 1.0) * self[2] + 4.0 * v * self[5] + 4.0 * u * self[6] + 4.0 * t * self[9],
            (4.0 * t - 1.0) * self[3] + 4.0 * u * self[7] + 4.0 * v * self[8] + 4.0 * w * self[9],
        ]
    }

    fn bezier(&self) -> Self {
        match self.1 {
            HOType::Lagrange => {
                let p1 = 0.5 * (4.0 * self[4] - self[0] - self[1]);
                let p2 = 0.5 * (4.0 * self[5] - self[1] - self[2]);
                let p3 = 0.5 * (4.0 * self[6] - self[2] - self[0]);
                let p4 = 0.5 * (4.0 * self[7] - self[0] - self[3]);
                let p5 = 0.5 * (4.0 * self[8] - self[1] - self[3]);
                let p6 = 0.5 * (4.0 * self[9] - self[2] - self[3]);

                Self(
                    [self[0], self[1], self[2], self[3], p1, p2, p3, p4, p5, p6],
                    HOType::Bezier,
                )
            }
            HOType::Bezier => *self,
        }
    }
}

impl<const D: usize> Index<usize> for QuadraticGTetrahedron<D> {
    type Output = Vertex<D>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<const D: usize> IntoIterator for QuadraticGTetrahedron<D> {
    type Item = Vertex<D>;
    type IntoIter = std::array::IntoIter<Self::Item, 10>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<const D: usize> Default for QuadraticGTetrahedron<D> {
    fn default() -> Self {
        Self([Vertex::zeros(); 10], HOType::Lagrange)
    }
}
// Edge centers:
// (0, 1) -> 4
// (1, 2) -> 5
// (2, 0) -> 6
// (0, 3) -> 7
// (1, 3) -> 8
// (2, 3) -> 9
const QUADRATICTETRAHEDRON2FACE: [QuadraticTriangle<usize>; 4] = [
    QuadraticTriangle([1, 2, 3, 5, 9, 8]),
    QuadraticTriangle([2, 0, 3, 6, 7, 9]),
    QuadraticTriangle([0, 1, 3, 4, 8, 7]),
    QuadraticTriangle([0, 2, 1, 6, 5, 4]),
];

impl<T: Idx> Simplex for QuadraticTetrahedron<T> {
    type T = T;
    type FACE = QuadraticTriangle<T>;
    type GEOM<const D: usize> = QuadraticGTetrahedron<D>;
    const DIM: usize = 3;
    const N_VERTS: usize = 10;
    const N_EDGES: usize = 6;
    const N_FACES: usize = 4;

    fn get(&self, index: usize) -> usize {
        self.0[index].try_into().unwrap()
    }

    fn edge(&self, _i: usize) -> Edge<usize> {
        unreachable!()
    }

    fn face(&self, i: usize) -> Self::FACE {
        Self::FACE::from_iter(
            QUADRATICTETRAHEDRON2FACE[i]
                .into_iter()
                .map(|j| self.get(j)),
        )
    }

    fn set(&mut self, i: usize, v: usize) {
        self.0[i] = v.try_into().unwrap();
    }

    fn contains(&self, i: usize) -> bool {
        self.0.contains(&i.try_into().unwrap())
    }

    fn sorted(&self) -> Self {
        unreachable!()
    }

    fn is_same(&self, other: &Self) -> bool {
        *self == *other
    }

    fn invert(&mut self) {
        self.0.swap(1, 0);
    }

    fn order() -> u8 {
        2
    }
}

impl<const D: usize> GSimplex<D> for QuadraticGTetrahedron<D> {
    const N_VERTS: usize = 10;
    type ARRAY<T: Debug + Default + Clone + Copy> = [T; 4];
    type BCOORDS = Self::ARRAY<f64>;
    type TOPO = QuadraticTetrahedron<usize>;
    type FACE = QuadraticGTriangle<D>;

    fn ideal_vol() -> f64 {
        1.0
    }

    fn edge(&self, _i: usize) -> GEdge<D> {
        unreachable!()
    }

    fn face(&self, i: usize) -> Self::FACE {
        Self::FACE::from_iter(QUADRATICTETRAHEDRON2FACE[i].into_iter().map(|j| self[j]))
    }

    fn set(&mut self, i: usize, v: Vertex<D>) {
        self.0[i] = v;
    }

    fn has_normal() -> bool {
        D == 3
    }

    fn vol(&self) -> f64 {
        self.integrate(|_| 1.0)
    }

    fn integrate<G: Fn(&Self::BCOORDS) -> f64>(&self, f: G) -> f64 {
        let mut res = 0.0;
        for &(weight, v, w, t) in &super::quadratures::QUADRATURE_TETRAHEDRON_10 {
            let bcoords = [1.0 - v - w - t, v, w, t];
            let [du, mut dv, mut dw, mut dt] = self.jac_mapping(&bcoords);
            dv -= du;
            dw -= du;
            dt -= du;
            res += weight * f(&bcoords) * dt.dot(&dv.cross(&dw));
        }
        1.0 / 6.0 * res
    }

    fn normal(&self, _bcoords: Option<&Self::BCOORDS>) -> Vertex<D> {
        unreachable!()
    }

    fn radius(&self) -> f64 {
        unreachable!()
    }

    fn center_bcoords() -> Self::BCOORDS {
        [0.25; 4]
    }

    fn bcoords(&self, _v: &Vertex<D>) -> Self::BCOORDS {
        unimplemented!()
    }

    /// Vertex from barycentric coordinates
    fn vert(&self, bcoords: &Self::BCOORDS) -> Vertex<D> {
        self.mapping(bcoords)
    }

    fn gamma(&self) -> f64 {
        unreachable!()
    }

    fn bounding_box(&self) -> (Vertex<D>, Vertex<D>) {
        match self.1 {
            HOType::Lagrange => self.bezier().bounding_box(),
            HOType::Bezier => self
                .into_iter()
                .skip(1)
                .fold((self[0], self[0]), |mut a, b| {
                    for i in 0..D {
                        a.0[i] = a.0[i].min(b[i]);
                        a.1[i] = a.1[i].max(b[i]);
                    }
                    a
                }),
        }
    }
}

/// Adaptive computation of the bounds of the determinant of the jacobian
/// for quadratic Tetrahedrons
pub struct AdativeBoundsQuadraticTetrahedron<'a> {
    c0: SVector<f64, 4>,
    c1: SVector<f64, 4>,
    c2: SVector<f64, 4>,
    c3: SVector<f64, 4>,
    tet: &'a QuadraticGTetrahedron<3>,
    b: SVector<f64, 10>,
    children: Option<Box<[Self; 8]>>,
    lu: &'a LU<f64, Const<10>, Const<10>>,
}

impl<'a> AdativeBoundsQuadraticTetrahedron<'a> {
    fn midpoints(
        c0: &SVector<f64, 4>,
        c1: &SVector<f64, 4>,
        c2: &SVector<f64, 4>,
        c3: &SVector<f64, 4>,
    ) -> [SVector<f64, 4>; 6] {
        [
            0.5 * (c0 + c1),
            0.5 * (c1 + c2),
            0.5 * (c2 + c0),
            0.5 * (c0 + c3),
            0.5 * (c1 + c3),
            0.5 * (c2 + c3),
        ]
    }

    #[must_use]
    pub fn lagrange_to_bezier() -> LU<f64, Const<10>, Const<10>> {
        let c0 = SVector::<f64, 4>::new(1.0, 0.0, 0.0, 0.0);
        let c1 = SVector::<f64, 4>::new(0.0, 1.0, 0.0, 0.0);
        let c2 = SVector::<f64, 4>::new(0.0, 0.0, 1.0, 0.0);
        let c3 = SVector::<f64, 4>::new(0.0, 0.0, 0.0, 1.0);

        let [c4, c5, c6, c7, c8, c9] = Self::midpoints(&c0, &c1, &c2, &c3);
        let pts = [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9];

        let bezier = |(i, u, v, w, t)| match i {
            0 => u * u,
            1 => v * v,
            2 => w * w,
            3 => t * t,
            4 => 2.0 * u * v,
            5 => 2.0 * v * w,
            6 => 2.0 * w * u,
            7 => 2.0 * u * t,
            8 => 2.0 * v * t,
            9 => 2.0 * w * t,
            _ => unreachable!(),
        };

        let mat = SMatrix::<f64, 10, 10>::from_fn(|i, j| {
            bezier((j, pts[i][0], pts[i][1], pts[i][2], pts[i][3]))
        });
        mat.lu()
    }

    /// Compute the distortion (ratio of the max to min of the determinant of the jacobian)
    /// for all elements in the mesh
    #[must_use]
    pub fn element_distortion<T: Idx>(msh: &impl Mesh<3, C = QuadraticTetrahedron<T>>) -> Vec<f64> {
        let lu = Self::lagrange_to_bezier();

        msh.gelems()
            .map(|ge| {
                let (_, (min, max)) =
                    AdativeBoundsQuadraticTetrahedron::new(&ge, &lu).compute_bounds(None);
                max / min
            })
            .collect()
    }

    #[must_use]
    pub fn new(tet: &'a QuadraticGTetrahedron<3>, lu: &'a LU<f64, Const<10>, Const<10>>) -> Self {
        Self::new_with_corners(
            tet,
            lu,
            SVector::<f64, 4>::new(1.0, 0.0, 0.0, 0.0),
            SVector::<f64, 4>::new(0.0, 1.0, 0.0, 0.0),
            SVector::<f64, 4>::new(0.0, 0.0, 1.0, 0.0),
            SVector::<f64, 4>::new(0.0, 0.0, 0.0, 1.0),
        )
    }

    #[must_use]
    fn new_with_corners(
        tet: &'a QuadraticGTetrahedron<3>,
        lu: &'a LU<f64, Const<10>, Const<10>>,
        c0: SVector<f64, 4>,
        c1: SVector<f64, 4>,
        c2: SVector<f64, 4>,
        c3: SVector<f64, 4>,
    ) -> Self {
        let [c4, c5, c6, c7, c8, c9] = Self::midpoints(&c0, &c1, &c2, &c3);
        let pts = [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9];

        let det_jac = |u, v, w, t| {
            let [du, mut dv, mut dw, mut dt] = tet.jac_mapping(&[u, v, w, t]);
            dv -= du;
            dw -= du;
            dt -= du;
            dt.dot(&dv.cross(&dw)) / 6.0
        };
        let vals =
            SVector::<f64, 10>::from_fn(|i, _| det_jac(pts[i][0], pts[i][1], pts[i][2], pts[i][3]));
        let b = lu.solve(&vals).unwrap();
        Self {
            c0,
            c1,
            c2,
            c3,
            tet,
            b,
            children: None,
            lu,
        }
    }

    fn subdivide(&mut self) {
        let [c4, c5, c6, c7, c8, c9] = Self::midpoints(&self.c0, &self.c1, &self.c2, &self.c3);
        self.children = Some(Box::new([
            AdativeBoundsQuadraticTetrahedron::new_with_corners(
                self.tet, self.lu, self.c0, c4, c6, c7,
            ),
            AdativeBoundsQuadraticTetrahedron::new_with_corners(
                self.tet, self.lu, c4, self.c1, c5, c8,
            ),
            AdativeBoundsQuadraticTetrahedron::new_with_corners(self.tet, self.lu, c4, c5, c6, c7),
            AdativeBoundsQuadraticTetrahedron::new_with_corners(self.tet, self.lu, c4, c5, c7, c8),
            AdativeBoundsQuadraticTetrahedron::new_with_corners(self.tet, self.lu, c8, c5, c7, c9),
            AdativeBoundsQuadraticTetrahedron::new_with_corners(self.tet, self.lu, c6, c5, c9, c7),
            AdativeBoundsQuadraticTetrahedron::new_with_corners(
                self.tet, self.lu, c7, c8, c9, self.c3,
            ),
            AdativeBoundsQuadraticTetrahedron::new_with_corners(
                self.tet, self.lu, c6, c5, self.c2, c9,
            ),
        ]));
    }

    fn bounds(&self) -> (f64, f64) {
        let mut min = f64::MAX;
        let mut max = f64::MIN;
        if let Some(children) = &self.children {
            for child in children.iter() {
                let (cmin, cmax) = child.bounds();
                min = min.min(cmin);
                max = max.max(cmax);
            }
        } else {
            min = self.b.iter().copied().fold(f64::INFINITY, f64::min);
            max = self.b.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        }
        (min, max)
    }

    fn refine_min(&mut self) -> bool {
        if let Some(children) = &mut self.children {
            let mut children_bounds = [(0.0, 0.0); 8];
            for (i, child) in children.iter_mut().enumerate() {
                children_bounds[i] = child.bounds();
            }
            if children_bounds.iter().any(|(_, x)| *x < 0.0) {
                return true;
            }
            let min = children_bounds
                .iter()
                .map(|(x, _)| *x)
                .fold(f64::INFINITY, f64::min);
            let imin = children_bounds
                .iter()
                .position(|(x, _)| (*x - min).abs() < f64::EPSILON)
                .unwrap();
            children[imin].refine_min()
        } else {
            self.subdivide();
            false
        }
    }

    fn refine_max(&mut self) {
        if let Some(children) = &mut self.children {
            let mut children_bounds = [(0.0, 0.0); 4];
            for (i, child) in children.iter_mut().enumerate() {
                children_bounds[i] = child.bounds();
            }
            let max = children_bounds
                .iter()
                .map(|(x, _)| *x)
                .fold(f64::NEG_INFINITY, f64::max);
            let imax = children_bounds
                .iter()
                .position(|(x, _)| (*x - max).abs() < f64::EPSILON)
                .unwrap();
            children[imax].refine_max();
        } else {
            self.subdivide();
        }
    }

    /// Compute the bounds of the determinant of the jacobian as described in
    /// https://gmsh.info/doc/preprints/gmsh_curved_preprint.pdf
    ///   - At most 10 refinements are performed for min and max
    ///   - Iterations stop if the relative change of the bounds is less than tol
    ///   - Iterations stop if as invalid element is found when refining the min
    pub fn compute_bounds(&mut self, tol: Option<f64>) -> (bool, (f64, f64)) {
        let tol = tol.unwrap_or(1e-1);
        let (mut min, mut max) = self.bounds();
        let mut is_invalid = false;
        for _ in 0..10 {
            self.refine_max();
            let (_, new_max) = self.bounds();
            if (new_max - max).abs() < tol {
                break;
            }
            max = new_max;
        }

        for _ in 0..10 {
            is_invalid = is_invalid || self.refine_min();
            let (new_min, _) = self.bounds();
            if (new_min - min).abs() < tol {
                break;
            }
            min = new_min;
        }

        (is_invalid, (min, max))
    }
}

// #[cfg(test)]
// mod tests {
//     use crate::{
//         Vert2d, Vert3d, assert_delta,
//         mesh::{
//             BoundaryMesh3d, GSimplex, GTetrahedron, Mesh, QuadraticGTetrahedron, Tetrahedron,
//             elements::{
//                 ho_simplex::HOType, quadratic_Tetrahedron::AdativeBoundsQuadraticTetrahedron,
//             },
//         },
//     };

//     fn to_linear_mesh(ge: &QuadraticGTetrahedron<3>, n_splits: u32) -> BoundaryMesh3d {
//         let mut msh = BoundaryMesh3d::from_vecs(
//             vec![
//                 Vert3d::new(0.0, 0.0, 0.0),
//                 Vert3d::new(1.0, 0.0, 0.0),
//                 Vert3d::new(0.0, 1.0, 0.0),
//             ],
//             vec![Tetrahedron::<usize>::new(0, 1, 2)],
//             vec![1],
//             Vec::new(),
//             Vec::new(),
//         );
//         for _ in 0..n_splits {
//             msh = msh.split();
//         }

//         msh.verts_mut()
//             .for_each(|x| *x = ge.vert(&[1.0 - x[0] - x[1], x[0], x[1]]));

//         msh
//     }

//     #[test]
//     fn test_quadratic_Tetrahedron() {
//         let p0 = Vert3d::new(0., 0., 0.);
//         let p1 = Vert3d::new(2., 0., -0.25);
//         let p2 = Vert3d::new(1., 1., 1.0);
//         let p3 = Vert3d::new(1., -0.25, 0.25);
//         let p4 = Vert3d::new(1.5, 0.75, 0.5);
//         let p5 = Vert3d::new(0.25, 0.5, 0.5);

//         let ge = GTetrahedron::new(&p0, &p1, &p2);
//         let ge2 = QuadraticGTetrahedron::new(
//             &p0,
//             &p1,
//             &p2,
//             &(0.5 * (p0 + p1)),
//             &(0.5 * (p1 + p2)),
//             &(0.5 * (p2 + p0)),
//             HOType::Lagrange,
//         );

//         let n = ge.normal(None);
//         let n2 = ge2.normal(Some(&[0.5, 0.5, 0.5]));
//         assert_delta!((n - n2).norm(), 0.0, 1e-12);

//         let v = ge.vol();
//         let v2 = ge2.vol();
//         assert_delta!(v, v2, 1e-12);

//         let [v, w] = [0.1234, 0.2345];
//         let p = ge2.vert(&[1.0 - v - w, v, w]);
//         let n = ge2.normal(Some(&[1.0 - v - w, v, w]));

//         for p2 in [p + 0.1 * n, p + n, p + 10.0 * n] {
//             let (p3, _) = ge2.project(&p2);
//             assert_delta!((p - p3).norm(), 0.0, 1e-12);
//         }

//         let ge2 = QuadraticGTetrahedron::new(&p0, &p1, &p2, &p3, &p4, &p5, HOType::Lagrange);

//         let msh = to_linear_mesh(&ge2, 6);
//         let v = msh.vol();
//         let v2 = ge2.vol();
//         assert_delta!(v, v2, 1e-4);

//         let [v, w] = [0.1234, 0.2345];
//         let p = ge2.vert(&[1.0 - v - w, v, w]);
//         let n = ge2.normal(Some(&[1.0 - v - w, v, w]));

//         for p2 in [p + 0.1 * n, p + n] {
//             let (p3, _) = ge2.project(&p2);
//             assert_delta!((p - p3).norm(), 0.0, 1e-5);
//         }

//         let (mini, maxi) = ge2.bounding_box();
//         for v in msh.verts() {
//             for i in 0..3 {
//                 assert!(v[i] > mini[i] - 1e-12);
//                 assert!(v[i] < maxi[i] + 1e-12);
//             }
//         }
//     }

//     #[test]
//     fn test_bounds() {
//         let p0 = Vert2d::new(0., 0.);
//         let p1 = Vert2d::new(2., 0.);
//         let p2 = Vert2d::new(1., 1.);
//         let p3 = Vert2d::new(1.8, 0.4);
//         let p4 = Vert2d::new(2.1, 0.41);
//         let p5 = Vert2d::new(0.25, 0.5);

//         let tri = QuadraticGTetrahedron::new(&p0, &p1, &p2, &p3, &p4, &p5, HOType::Lagrange);

//         let lu = AdativeBoundsQuadraticTetrahedron::lagrange_to_bezier();
//         let mut adb = AdativeBoundsQuadraticTetrahedron::new(&tri, &lu);

//         let (is_invalid, (min, max)) = adb.compute_bounds(Some(1e-3));
//         assert!(is_invalid);
//         assert_delta!(min, -0.247, 1e-3);
//         assert_delta!(max, 6.12, 1e-3);
//     }
// }
