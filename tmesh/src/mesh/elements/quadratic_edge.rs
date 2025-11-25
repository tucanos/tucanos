use argmin::core::{CostFunction, Executor, Gradient, Hessian};

use crate::{
    Vertex,
    mesh::{
        Edge, GEdge, GNode, GSimplex, Idx, Node, Simplex,
        elements::{ho_simplex::HOType, quadratures::QUADRATURE_EDGE_6},
    },
};
use std::fmt::Debug;
use std::ops::Index;

/// Edge
#[derive(Default, Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct QuadraticEdge<T: Idx>(pub(crate) [T; 3]);

impl<T: Idx> QuadraticEdge<T> {
    #[must_use]
    pub fn new(i0: usize, i1: usize, i2: usize) -> Self {
        Self([
            i0.try_into().unwrap(),
            i1.try_into().unwrap(),
            i2.try_into().unwrap(),
        ])
    }
}

impl<T: Idx> IntoIterator for QuadraticEdge<T> {
    type Item = usize;
    type IntoIter = std::iter::Map<std::array::IntoIter<T, 3>, fn(T) -> usize>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter().map(|x| x.try_into().unwrap())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct QuadraticGEdge<const D: usize>([Vertex<D>; 3], HOType);

impl<const D: usize> QuadraticGEdge<D> {
    #[must_use]
    pub const fn new(v0: &Vertex<D>, v1: &Vertex<D>, v2: &Vertex<D>, etype: HOType) -> Self {
        Self([*v0, *v1, *v2], etype)
    }

    fn linear(&self) -> GEdge<D> {
        GEdge::new(&self[0], &self[1])
    }

    fn mapping(&self, bcoords: &[f64; 2]) -> Vertex<D> {
        let [u, v] = bcoords;
        2.0 * u * (u - 0.5) * self[0] + 2.0 * v * (v - 0.5) * self[1] + 4.0 * u * v * self[2]
    }

    fn jac_mapping(&self, bcoords: &[f64; 2]) -> [Vertex<D>; 2] {
        let [u, v] = bcoords;
        [
            (4.0 * u - 1.0) * self[0] + 4.0 * v * self[2],
            (4.0 * v - 1.0) * self[1] + 4.0 * u * self[2],
        ]
    }

    fn hess_mapping(&self, _bcoords: &[f64; 2]) -> [Vertex<D>; 3] {
        [4.0 * self[0], 4.0 * self[1], 4.0 * self[2]]
    }

    fn bezier(&self) -> Self {
        match self.1 {
            HOType::Lagrange => {
                let p = 0.5 * (4.0 * self[2] - self[0] - self[1]);
                Self([self[0], self[1], p], HOType::Bezier)
            }
            HOType::Bezier => *self,
        }
    }
}

impl<const D: usize> Index<usize> for QuadraticGEdge<D> {
    type Output = Vertex<D>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<const D: usize> IntoIterator for QuadraticGEdge<D> {
    type Item = Vertex<D>;
    type IntoIter = std::array::IntoIter<Self::Item, 3>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<const D: usize> Default for QuadraticGEdge<D> {
    fn default() -> Self {
        Self([Vertex::zeros(); 3], HOType::Lagrange)
    }
}

const QUADRATICEDGE2FACES: [Node<usize>; 2] = [Node([1]), Node([0])];

impl<T: Idx> Simplex for QuadraticEdge<T> {
    type T = T;
    type FACE = Node<T>;
    type GEOM<const D: usize> = QuadraticGEdge<D>;
    const DIM: usize = 1;
    const N_VERTS: usize = 3;
    const N_EDGES: usize = 1;
    const N_FACES: usize = 2;

    fn get(&self, index: usize) -> usize {
        self.0[index].try_into().unwrap()
    }

    fn edge(&self, _i: usize) -> Edge<usize> {
        unreachable!()
    }

    fn face(&self, i: usize) -> Self::FACE {
        Self::FACE::from_iter(QUADRATICEDGE2FACES[i].into_iter().map(|j| self.get(j)))
    }

    fn set(&mut self, i: usize, v: usize) {
        self.0[i] = v.try_into().unwrap();
    }

    fn contains(&self, i: usize) -> bool {
        self.0.contains(&i.try_into().unwrap())
    }

    fn sorted(&self) -> Self {
        if self.0[0] < self.0[1] {
            Self(self.0)
        } else {
            Self([self.0[1], self.0[0], self.0[2]])
        }
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

impl<const D: usize> GSimplex<D> for QuadraticGEdge<D> {
    const N_VERTS: usize = 3;
    type ARRAY<T: Debug + Default + Clone + Copy> = [T; 2];
    type BCOORDS = Self::ARRAY<f64>;
    type TOPO = QuadraticEdge<usize>;
    type FACE = GNode<D>;

    fn ideal_vol() -> f64 {
        1.0
    }

    fn edge(&self, _i: usize) -> GEdge<D> {
        unreachable!()
    }

    fn face(&self, i: usize) -> Self::FACE {
        Self::FACE::from_iter(QUADRATICEDGE2FACES[i].into_iter().map(|j| self[j]))
    }

    fn set(&mut self, i: usize, v: Vertex<D>) {
        self.0[i] = v;
    }

    fn has_normal() -> bool {
        D == 2
    }

    fn vol(&self) -> f64 {
        self.integrate(|_| 1.0)
    }

    fn integrate<G: Fn(&Self::BCOORDS) -> f64>(&self, f: G) -> f64 {
        let mut res = 0.0;
        for &(weight, v) in &QUADRATURE_EDGE_6 {
            let bcoords = [1.0 - v, v];
            let [mut du, dv] = self.jac_mapping(&bcoords);
            du -= dv;
            res += weight * f(&bcoords) * (du.norm_squared()).sqrt();
        }
        res
    }

    fn normal(&self, bcoords: Option<&Self::BCOORDS>) -> Vertex<D> {
        if Self::has_normal() {
            let [du, dv] = self.jac_mapping(bcoords.unwrap());
            let mut res = Vertex::<D>::zeros();
            res[0] = dv[1] - du[1];
            res[1] = du[0] - dv[0];
            res
        } else {
            unreachable!()
        }
    }

    fn radius(&self) -> f64 {
        unreachable!()
    }

    fn center_bcoords() -> Self::BCOORDS {
        [0.5, 0.5]
    }

    fn bcoords(&self, v: &Vertex<D>) -> Self::BCOORDS {
        let uv = self.linear().bcoords(v);

        let linesearch = argmin::solver::linesearch::MoreThuenteLineSearch::new();
        let solver = argmin::solver::newton::NewtonCG::new(linesearch)
            .with_tolerance(1e-10)
            .unwrap();
        if let Ok(res) = Executor::new(QuadraticEdgeProjection { v, ge: self }, solver)
            .configure(|state| state.param([uv[1]].into()).max_iters(100))
            // .add_observer(
            //     argmin_observer_slog::SlogLogger::term(),
            //     argmin::core::observers::ObserverMode::Always,
            // )
            .run()
        {
            let v = res.state.best_param.unwrap();
            [1.0 - v[0], v[0]]
        } else {
            panic!()
            // [f64::NAN, f64::NAN]
        }
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

struct QuadraticEdgeProjection<'a, const D: usize> {
    v: &'a Vertex<D>,
    ge: &'a QuadraticGEdge<D>,
}

impl<const D: usize> CostFunction for QuadraticEdgeProjection<'_, D> {
    type Param = nalgebra::Vector1<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        let uv = [1.0 - param[0], param[0]];
        let dx = self.v - self.ge.mapping(&uv);
        Ok(dx.norm_squared())
    }
}

impl<const D: usize> Gradient for QuadraticEdgeProjection<'_, D> {
    type Param = nalgebra::Vector1<f64>;
    type Gradient = nalgebra::Vector1<f64>;

    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, argmin::core::Error> {
        let uv = [1.0 - param[0], param[0]];
        let dx = self.v - self.ge.mapping(&uv);
        let [du, dv] = self.ge.jac_mapping(&uv);
        Ok([-2.0 * dx.dot(&(dv - du))].into())
    }
}

impl<const D: usize> Hessian for QuadraticEdgeProjection<'_, D> {
    type Param = nalgebra::Vector1<f64>;
    type Hessian = nalgebra::Matrix1<f64>;

    fn hessian(&self, param: &Self::Param) -> Result<Self::Hessian, argmin::core::Error> {
        let uv = [1.0 - param[0], param[0]];
        let dx = self.v - self.ge.mapping(&uv);
        let [du, dv] = self.ge.jac_mapping(&uv);
        let [duu, dvv, duv] = self.ge.hess_mapping(&uv);
        Ok([-2.0 * (dx.dot(&(duu + dvv - 2.0 * duv)) - (dv - du).norm_squared())].into())
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        Vert2d, assert_delta,
        mesh::{GEdge, GSimplex, QuadraticGEdge, elements::ho_simplex::HOType},
    };

    #[test]
    fn test_quadratic_edge() {
        let p0 = Vert2d::new(0.0, 0.1);
        let p1 = Vert2d::new(0.2, 1.0);

        let ge = GEdge::new(&p0, &p1);
        let p2 = 0.5 * (p0 + p1);
        let ge2 = QuadraticGEdge::new(&p0, &p1, &p2, HOType::Lagrange);

        let n = ge.normal(None);
        let n2 = ge2.normal(Some(&[0.5, 0.5]));
        assert_delta!((n - n2).norm(), 0.0, 1e-12);

        let v = ge.vol();
        let v2 = ge2.vol();
        assert_delta!(v, v2, 1e-12);

        let p2 = Vert2d::new(0.5, 1.2);
        let ge2 = QuadraticGEdge::new(&p0, &p1, &p2, HOType::Lagrange);

        let n = 100;
        let t = (0..=n)
            .map(|i| f64::from(i) / f64::from(n))
            .collect::<Vec<_>>();

        let mut v = 0.0;
        for tmp in t.windows(2) {
            let p0 = ge2.vert(&[1.0 - tmp[0], tmp[0]]);
            let p1 = ge2.vert(&[1.0 - tmp[1], tmp[1]]);
            let ge = GEdge::new(&p0, &p1);
            v += ge.vol();
        }
        let v2 = ge2.vol();

        assert_delta!(v, v2, 0.05 * v);

        let v = 0.1234;
        let p = ge2.vert(&[1.0 - v, v]);
        let n = ge2.normal(Some(&[1.0 - v, v]));

        for p2 in [p + 0.1 * n, p + n, p + 10.0 * n] {
            let (p3, _) = ge2.project(&p2);
            assert!((p - p3).norm() < 1e-12);
        }
    }
}
