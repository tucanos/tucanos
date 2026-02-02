use crate::{
    Vertex,
    mesh::{
        Edge, GEdge, GSimplex, GTriangle, Idx, Mesh, QuadraticEdge, QuadraticGEdge, Simplex,
        Triangle, elements::ho_simplex::HOType,
    },
};
use argmin::core::{CostFunction, Executor, Gradient, Hessian};
use nalgebra::{Const, LU, SMatrix, SVector};
use std::fmt::Debug;
use std::ops::Index;

/// Edge
#[derive(Default, Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct QuadraticTriangle<T: Idx>(pub(crate) [T; 6]);

impl<T: Idx> QuadraticTriangle<T> {
    #[must_use]
    pub fn new(i0: usize, i1: usize, i2: usize, i3: usize, i4: usize, i5: usize) -> Self {
        Self([
            i0.try_into().unwrap(),
            i1.try_into().unwrap(),
            i2.try_into().unwrap(),
            i3.try_into().unwrap(),
            i4.try_into().unwrap(),
            i5.try_into().unwrap(),
        ])
    }

    pub fn linear(&self) -> Triangle<T> {
        Triangle::new(
            self.0[0].try_into().unwrap(),
            self.0[1].try_into().unwrap(),
            self.0[2].try_into().unwrap(),
        )
    }
}

impl<T: Idx> IntoIterator for QuadraticTriangle<T> {
    type Item = usize;
    type IntoIter = std::iter::Map<std::array::IntoIter<T, 6>, fn(T) -> usize>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter().map(|x| x.try_into().unwrap())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct QuadraticGTriangle<const D: usize>([Vertex<D>; 6], HOType);

impl<const D: usize> QuadraticGTriangle<D> {
    #[must_use]
    pub const fn new(
        v0: &Vertex<D>,
        v1: &Vertex<D>,
        v2: &Vertex<D>,
        v3: &Vertex<D>,
        v4: &Vertex<D>,
        v5: &Vertex<D>,
        etype: HOType,
    ) -> Self {
        Self([*v0, *v1, *v2, *v3, *v4, *v5], etype)
    }

    fn linear(&self) -> GTriangle<D> {
        GTriangle::new(&self[0], &self[1], &self[2])
    }

    fn mapping(&self, bcoords: &[f64; 3]) -> Vertex<D> {
        let [u, v, w] = bcoords;
        2.0 * u * (u - 0.5) * self[0]
            + 2.0 * v * (v - 0.5) * self[1]
            + 2.0 * w * (w - 0.5) * self[2]
            + 4.0 * u * v * self[3]
            + 4.0 * v * w * self[4]
            + 4.0 * u * w * self[5]
    }

    fn jac_mapping(&self, bcoords: &[f64; 3]) -> [Vertex<D>; 3] {
        let [u, v, w] = bcoords;
        [
            (4.0 * u - 1.0) * self[0] + 4.0 * v * self[3] + 4.0 * w * self[5],
            (4.0 * v - 1.0) * self[1] + 4.0 * u * self[3] + 4.0 * w * self[4],
            (4.0 * w - 1.0) * self[2] + 4.0 * v * self[4] + 4.0 * u * self[5],
        ]
    }

    /// order: uu, vv, ww, uv, uw, vw
    fn hess_mapping(&self, _bcoords: &[f64; 3]) -> [Vertex<D>; 6] {
        [
            4.0 * self[0],
            4.0 * self[1],
            4.0 * self[2],
            4.0 * self[3],
            4.0 * self[5],
            4.0 * self[4],
        ]
    }

    /// Curvature at the center of the triangle
    #[must_use]
    pub fn curvature(&self) -> (Vertex<D>, Vertex<D>) {
        assert_eq!(D, 3);
        let bcoords = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
        let [g_u, mut g_v, mut g_w] = self.jac_mapping(&bcoords);
        g_v -= g_u;
        g_w -= g_u;
        let n = g_v.cross(&g_w).normalize();
        let b = SMatrix::<f64, 2, 2>::new(
            g_v.norm_squared(),
            g_v.dot(&g_w),
            g_v.dot(&g_w),
            g_w.norm_squared(),
        );

        let [h_uu, h_vv, h_ww, h_uv, h_uw, h_vw] = self.hess_mapping(&bcoords);

        let a = SMatrix::<f64, 2, 2>::new(
            (h_uu + h_vv - 2.0 * h_uv).dot(&n),
            (h_uu + h_vw - h_uv - h_uw).dot(&n),
            (h_uu + h_vw - h_uv - h_uw).dot(&n),
            (h_uu + h_ww - 2.0 * h_uw).dot(&n),
        );
        let mut eig = b.symmetric_eigen();
        eig.eigenvalues.iter_mut().for_each(|s| *s = 1.0 / s.sqrt());
        let tmp = eig.recompose();

        let p = tmp * a * tmp;
        let eig = p.symmetric_eigen();
        let p = eig.eigenvectors;
        let tmp = tmp * p;

        let ev0 = if eig.eigenvalues[0].abs() < 1e-16 {
            1e-12
        } else {
            eig.eigenvalues[0]
        };
        let ev1 = if eig.eigenvalues[1].abs() < 1e-16 {
            1e-12
        } else {
            eig.eigenvalues[1]
        };

        let mut u = tmp[0] * g_v + tmp[1] * g_w;
        u.normalize_mut();
        u *= ev0;

        let mut v = tmp[2] * g_v + tmp[3] * g_w;
        v.normalize_mut();
        v *= ev1;

        (u, v)
    }

    fn bezier(&self) -> Self {
        match self.1 {
            HOType::Lagrange => {
                let p1 = 0.5 * (4.0 * self[3] - self[0] - self[1]);
                let p2 = 0.5 * (4.0 * self[4] - self[1] - self[2]);
                let p3 = 0.5 * (4.0 * self[5] - self[2] - self[0]);
                Self([self[0], self[1], self[2], p1, p2, p3], HOType::Bezier)
            }
            HOType::Bezier => *self,
        }
    }
}

impl<const D: usize> Index<usize> for QuadraticGTriangle<D> {
    type Output = Vertex<D>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<const D: usize> IntoIterator for QuadraticGTriangle<D> {
    type Item = Vertex<D>;
    type IntoIter = std::array::IntoIter<Self::Item, 6>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<const D: usize> Default for QuadraticGTriangle<D> {
    fn default() -> Self {
        Self([Vertex::zeros(); 6], HOType::Lagrange)
    }
}

const QUADRATICTRIANGLE2FACE: [QuadraticEdge<usize>; 3] = [
    QuadraticEdge([0, 1, 3]),
    QuadraticEdge([1, 2, 4]),
    QuadraticEdge([2, 0, 5]),
];

impl<T: Idx> Simplex for QuadraticTriangle<T> {
    type T = T;
    type FACE = QuadraticEdge<T>;
    type GEOM<const D: usize> = QuadraticGTriangle<D>;
    const DIM: usize = 2;
    const N_VERTS: usize = 6;
    const N_EDGES: usize = 3;
    const N_FACES: usize = 3;

    fn get(&self, index: usize) -> usize {
        self.0[index].try_into().unwrap()
    }

    fn edge(&self, _i: usize) -> Edge<usize> {
        unreachable!()
    }

    fn face(&self, i: usize) -> Self::FACE {
        Self::FACE::from_iter(QUADRATICTRIANGLE2FACE[i].into_iter().map(|j| self.get(j)))
    }

    fn set(&mut self, i: usize, v: usize) {
        self.0[i] = v.try_into().unwrap();
    }

    fn contains(&self, i: usize) -> bool {
        self.0.contains(&i.try_into().unwrap())
    }

    fn sorted(&self) -> Self {
        fn edge_center(i0: usize, i1: usize) -> usize {
            match i0 {
                0 => match i1 {
                    1 => 3,
                    2 => 5,
                    _ => unreachable!(),
                },
                1 => match i1 {
                    0 => 3,
                    2 => 4,
                    _ => unreachable!(),
                },
                2 => match i1 {
                    0 => 5,
                    1 => 4,
                    _ => unreachable!(),
                },
                _ => unreachable!(),
            }
        }

        let mut indices = [0, 1, 2];
        indices.sort_by_key(|&i| &self.0[i]);

        Self([
            self.0[indices[0]],
            self.0[indices[1]],
            self.0[indices[2]],
            self.0[edge_center(indices[0], indices[1])],
            self.0[edge_center(indices[1], indices[2])],
            self.0[edge_center(indices[2], indices[0])],
        ])
    }

    fn is_same(&self, other: &Self) -> bool {
        self.linear().is_same(&other.linear())
    }

    fn invert(&mut self) {
        self.0.swap(1, 0);
    }

    fn order() -> u8 {
        2
    }
}

impl<const D: usize> GSimplex<D> for QuadraticGTriangle<D> {
    const N_VERTS: usize = 6;
    type ARRAY<T: Debug + Default + Clone + Copy> = [T; 3];
    type BCOORDS = Self::ARRAY<f64>;
    type TOPO = QuadraticTriangle<usize>;
    type FACE = QuadraticGEdge<D>;

    fn ideal_vol() -> f64 {
        1.0
    }

    fn edge(&self, _i: usize) -> GEdge<D> {
        unreachable!()
    }

    fn face(&self, i: usize) -> Self::FACE {
        Self::FACE::from_iter(QUADRATICTRIANGLE2FACE[i].into_iter().map(|j| self[j]))
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
        for &(weight, v, w) in &super::quadratures::QUADRATURE_TRIANGLE_6 {
            let bcoords = [1.0 - v - w, v, w];
            let [du, mut dv, mut dw] = self.jac_mapping(&bcoords);
            dv -= du;
            dw -= du;
            res += if D == 3 {
                weight * f(&bcoords) * dv.cross(&dw).norm()
            } else {
                weight * f(&bcoords) * (dv[0] * dw[1] - dv[1] * dw[0]).abs()
            };
        }
        0.5 * res
    }

    fn normal(&self, bcoords: Option<&Self::BCOORDS>) -> Vertex<D> {
        if Self::has_normal() {
            let [du, mut dv, mut dw] =
                self.jac_mapping(bcoords.unwrap_or(&[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]));
            dv -= du;
            dw -= du;
            0.125 * dv.cross(&dw)
        } else {
            unreachable!()
        }
    }

    fn radius(&self) -> f64 {
        unreachable!()
    }

    fn center_bcoords() -> Self::BCOORDS {
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
    }

    fn bcoords(&self, v: &Vertex<D>) -> Self::BCOORDS {
        let uvw = self.linear().bcoords(v);

        let linesearch = argmin::solver::linesearch::MoreThuenteLineSearch::new();
        let solver = argmin::solver::newton::NewtonCG::new(linesearch)
            .with_tolerance(1e-10)
            .unwrap();

        let res = Executor::new(QuadraticTriangleProjection { v, ge: self }, solver)
            .configure(|state| state.param([uvw[1], uvw[2]].into()).max_iters(100))
            // .add_observer(
            //     argmin_observer_slog::SlogLogger::term(),
            //     argmin::core::observers::ObserverMode::Always,
            // )
            .run()
            .unwrap();
        let res = res.state.best_param.unwrap();
        [1.0 - res[0] - res[1], res[0], res[1]]
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

struct QuadraticTriangleProjection<'a, const D: usize> {
    v: &'a Vertex<D>,
    ge: &'a QuadraticGTriangle<D>,
}

impl<const D: usize> CostFunction for QuadraticTriangleProjection<'_, D> {
    type Param = nalgebra::Vector2<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        let uvw = [1.0 - param[0] - param[1], param[0], param[1]];
        let dx = self.v - self.ge.mapping(&uvw);
        Ok(dx.norm_squared())
    }
}

impl<const D: usize> Gradient for QuadraticTriangleProjection<'_, D> {
    type Param = nalgebra::Vector2<f64>;
    type Gradient = nalgebra::Vector2<f64>;

    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, argmin::core::Error> {
        let uvw = [1.0 - param[0] - param[1], param[0], param[1]];
        let dx = self.v - self.ge.mapping(&uvw);
        let [du, dv, dw] = self.ge.jac_mapping(&uvw);
        Ok([-2.0 * dx.dot(&(dv - du)), -2.0 * dx.dot(&(dw - du))].into())
    }
}

impl<const D: usize> Hessian for QuadraticTriangleProjection<'_, D> {
    type Param = nalgebra::Vector2<f64>;
    type Hessian = nalgebra::Matrix2<f64>;

    fn hessian(&self, param: &Self::Param) -> Result<Self::Hessian, argmin::core::Error> {
        let uvw = [1.0 - param[0] - param[1], param[0], param[1]];
        let dx = self.v - self.ge.mapping(&uvw);
        let [du, dv, dw] = self.ge.jac_mapping(&uvw);
        let [duu, dvv, dww, duv, duw, dvw] = self.ge.hess_mapping(&uvw);

        Ok([
            [
                -2.0 * (dx.dot(&(duu + dvv - 2.0 * duv)) - (dv - du).norm_squared()),
                -2.0 * (dx.dot(&(duu + dvw - duv - duw)) - (dv - du).dot(&(dw - du))),
            ],
            [
                -2.0 * (dx.dot(&(duu + dvw - duv - duw)) - (dv - du).dot(&(dw - du))),
                -2.0 * (dx.dot(&(duu + dww - 2.0 * duw)) - (dw - du).norm_squared()),
            ],
        ]
        .into())
    }
}

/// Adaptive computation of the bounds of the determinant of the jacobian
/// for quadratic triangles
pub struct AdativeBoundsQuadraticTriangle<'a> {
    c0: SVector<f64, 3>,
    c1: SVector<f64, 3>,
    c2: SVector<f64, 3>,
    tri: &'a QuadraticGTriangle<2>,
    b: SVector<f64, 6>,
    children: Option<Box<[Self; 4]>>,
    lu: &'a LU<f64, Const<6>, Const<6>>,
}

impl<'a> AdativeBoundsQuadraticTriangle<'a> {
    fn midpoints(
        c0: &SVector<f64, 3>,
        c1: &SVector<f64, 3>,
        c2: &SVector<f64, 3>,
    ) -> (SVector<f64, 3>, SVector<f64, 3>, SVector<f64, 3>) {
        (0.5 * (c0 + c1), 0.5 * (c1 + c2), 0.5 * (c2 + c0))
    }

    #[must_use]
    pub fn lagrange_to_bezier() -> LU<f64, Const<6>, Const<6>> {
        let c0 = SVector::<f64, 3>::new(1.0, 0.0, 0.0);
        let c1 = SVector::<f64, 3>::new(0.0, 1.0, 0.0);
        let c2 = SVector::<f64, 3>::new(0.0, 0.0, 1.0);
        let (c4, c5, c6) = Self::midpoints(&c0, &c1, &c2);
        let pts = [c0, c1, c2, c4, c5, c6];

        let bezier = |(i, u, v, w)| match i {
            0 => u * u,
            1 => v * v,
            2 => w * w,
            3 => 2.0 * u * v,
            4 => 2.0 * v * w,
            5 => 2.0 * w * u,
            _ => unreachable!(),
        };

        let mat =
            SMatrix::<f64, 6, 6>::from_fn(|i, j| bezier((j, pts[i][0], pts[i][1], pts[i][2])));
        mat.lu()
    }

    /// Compute the distortion (ratio of the max to min of the determinant of the jacobian)
    /// for all elements in the mesh
    #[must_use]
    pub fn element_distortion<T: Idx>(msh: &impl Mesh<2, C = QuadraticTriangle<T>>) -> Vec<f64> {
        let lu = Self::lagrange_to_bezier();

        msh.gelems()
            .map(|ge| {
                let (_, (min, max)) =
                    AdativeBoundsQuadraticTriangle::new(&ge, &lu).compute_bounds(None);
                max / min
            })
            .collect()
    }

    #[must_use]
    pub fn new(tri: &'a QuadraticGTriangle<2>, lu: &'a LU<f64, Const<6>, Const<6>>) -> Self {
        Self::new_with_corners(
            tri,
            lu,
            SVector::<f64, 3>::new(1.0, 0.0, 0.0),
            SVector::<f64, 3>::new(0.0, 1.0, 0.0),
            SVector::<f64, 3>::new(0.0, 0.0, 1.0),
        )
    }

    #[must_use]
    fn new_with_corners(
        tri: &'a QuadraticGTriangle<2>,
        lu: &'a LU<f64, Const<6>, Const<6>>,
        c0: SVector<f64, 3>,
        c1: SVector<f64, 3>,
        c2: SVector<f64, 3>,
    ) -> Self {
        let (c4, c5, c6) = Self::midpoints(&c0, &c1, &c2);
        let pts = [c0, c1, c2, c4, c5, c6];

        let det_jac = |u, v, w| {
            let [du, dv, dw] = tri.jac_mapping(&[u, v, w]);
            let mat = SMatrix::<f64, 2, 2>::from_columns(&[dv - du, dw - du]);
            mat.determinant()
        };
        let vals = SVector::<f64, 6>::from_fn(|i, _| det_jac(pts[i][0], pts[i][1], pts[i][2]));
        let b = lu.solve(&vals).unwrap();
        Self {
            c0,
            c1,
            c2,
            tri,
            b,
            children: None,
            lu,
        }
    }

    fn subdivide(&mut self) {
        let (c4, c5, c6) = Self::midpoints(&self.c0, &self.c1, &self.c2);
        self.children = Some(Box::new([
            AdativeBoundsQuadraticTriangle::new_with_corners(self.tri, self.lu, self.c0, c4, c6),
            AdativeBoundsQuadraticTriangle::new_with_corners(self.tri, self.lu, c4, self.c1, c5),
            AdativeBoundsQuadraticTriangle::new_with_corners(self.tri, self.lu, c6, c5, self.c2),
            AdativeBoundsQuadraticTriangle::new_with_corners(self.tri, self.lu, c4, c5, c6),
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
            let mut children_bounds = [(0.0, 0.0); 4];
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
#[cfg(test)]
mod tests {
    use crate::{
        Vert2d, Vert3d, assert_delta,
        mesh::{
            BoundaryMesh3d, GSimplex, GTriangle, Mesh, QuadraticGTriangle, Triangle,
            elements::{ho_simplex::HOType, quadratic_triangle::AdativeBoundsQuadraticTriangle},
        },
    };

    fn to_linear_mesh(ge: &QuadraticGTriangle<3>, n_splits: u32) -> BoundaryMesh3d {
        let mut msh = BoundaryMesh3d::from_vecs(
            vec![
                Vert3d::new(0.0, 0.0, 0.0),
                Vert3d::new(1.0, 0.0, 0.0),
                Vert3d::new(0.0, 1.0, 0.0),
            ],
            vec![Triangle::<usize>::new(0, 1, 2)],
            vec![1],
            Vec::new(),
            Vec::new(),
        );
        for _ in 0..n_splits {
            msh = msh.split();
        }

        msh.verts_mut()
            .for_each(|x| *x = ge.vert(&[1.0 - x[0] - x[1], x[0], x[1]]));

        msh
    }

    #[test]
    fn test_quadratic_triangle() {
        let p0 = Vert3d::new(0., 0., 0.);
        let p1 = Vert3d::new(2., 0., -0.25);
        let p2 = Vert3d::new(1., 1., 1.0);
        let p3 = Vert3d::new(1., -0.25, 0.25);
        let p4 = Vert3d::new(1.5, 0.75, 0.5);
        let p5 = Vert3d::new(0.25, 0.5, 0.5);

        let ge = GTriangle::new(&p0, &p1, &p2);
        let ge2 = QuadraticGTriangle::new(
            &p0,
            &p1,
            &p2,
            &(0.5 * (p0 + p1)),
            &(0.5 * (p1 + p2)),
            &(0.5 * (p2 + p0)),
            HOType::Lagrange,
        );

        let n = ge.normal(None);
        let n2 = ge2.normal(Some(&[0.5, 0.5, 0.5]));
        assert_delta!((n - n2).norm(), 0.0, 1e-12);

        let v = ge.vol();
        let v2 = ge2.vol();
        assert_delta!(v, v2, 1e-12);

        let [v, w] = [0.1234, 0.2345];
        let p = ge2.vert(&[1.0 - v - w, v, w]);
        let n = ge2.normal(Some(&[1.0 - v - w, v, w]));

        for p2 in [p + 0.1 * n, p + n, p + 10.0 * n] {
            let (p3, _) = ge2.project(&p2);
            assert_delta!((p - p3).norm(), 0.0, 1e-12);
        }

        let ge2 = QuadraticGTriangle::new(&p0, &p1, &p2, &p3, &p4, &p5, HOType::Lagrange);

        let msh = to_linear_mesh(&ge2, 6);
        let v = msh.vol();
        let v2 = ge2.vol();
        assert_delta!(v, v2, 1e-4);

        let [v, w] = [0.1234, 0.2345];
        let p = ge2.vert(&[1.0 - v - w, v, w]);
        let n = ge2.normal(Some(&[1.0 - v - w, v, w]));

        for p2 in [p + 0.1 * n, p + n] {
            let (p3, _) = ge2.project(&p2);
            assert_delta!((p - p3).norm(), 0.0, 1e-5);
        }

        let (mini, maxi) = ge2.bounding_box();
        for v in msh.verts() {
            for i in 0..3 {
                assert!(v[i] > mini[i] - 1e-12);
                assert!(v[i] < maxi[i] + 1e-12);
            }
        }
    }

    #[test]
    fn test_bounds() {
        let p0 = Vert2d::new(0., 0.);
        let p1 = Vert2d::new(2., 0.);
        let p2 = Vert2d::new(1., 1.);
        let p3 = Vert2d::new(1.8, 0.4);
        let p4 = Vert2d::new(2.1, 0.41);
        let p5 = Vert2d::new(0.25, 0.5);

        let tri = QuadraticGTriangle::new(&p0, &p1, &p2, &p3, &p4, &p5, HOType::Lagrange);

        let lu = AdativeBoundsQuadraticTriangle::lagrange_to_bezier();
        let mut adb = AdativeBoundsQuadraticTriangle::new(&tri, &lu);

        let (is_invalid, (min, max)) = adb.compute_bounds(Some(1e-3));
        assert!(is_invalid);
        assert_delta!(min, -0.247, 1e-3);
        assert_delta!(max, 6.12, 1e-3);
    }
}
