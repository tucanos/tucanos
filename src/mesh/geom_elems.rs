use crate::{
    mesh::Point,
    metric::{AnisoMetric, AnisoMetric2d, AnisoMetric3d, Metric},
    Idx,
};
use nalgebra::{Matrix2, Matrix3, Matrix4, Vector1, Vector2, Vector3, Vector4};
use std::fmt::Debug;

pub trait AsSliceF64 {
    fn as_slice_f64(&self) -> &[f64];
}

impl AsSliceF64 for Vector4<f64> {
    fn as_slice_f64(&self) -> &[f64] {
        self.as_slice()
    }
}

impl AsSliceF64 for Vector3<f64> {
    fn as_slice_f64(&self) -> &[f64] {
        self.as_slice()
    }
}

impl AsSliceF64 for Vector2<f64> {
    fn as_slice_f64(&self) -> &[f64] {
        self.as_slice()
    }
}

impl AsSliceF64 for Vector1<f64> {
    fn as_slice_f64(&self) -> &[f64] {
        self.as_slice()
    }
}

const SQRT_3: f64 = 1.732_050_807_568_877_2;
const SQRT_6: f64 = std::f64::consts::SQRT_2 * SQRT_3;

/// Geometric element defined by the coordinate of its vertices as well as metric information
pub trait GElem<const D: usize, M: Metric<D>>: Clone + Copy + Debug + Send {
    type Face: GElem<D, M>;
    type BCoords: AsSliceF64 + Debug;
    const IDEAL_VOL: f64;

    /// Create a `GElem` from its vertices and the metric at each vertex
    fn from_verts<I: Iterator<Item = (Point<D>, M)>>(points_n_metrics: I) -> Self;

    /// Create a `GElem` from a vertex and its opposite face
    fn from_vert_and_face(point: &Point<D>, metric: &M, face: &Self::Face) -> Self;

    /// Get the element's volume
    fn vol(&self) -> f64;

    /// Get the element's center
    fn center(&self) -> Point<D>;

    /// Get the quality of element $`K`$, defined as
    /// ```math
    /// q(K) = \beta_d \frac{|K|_{\mathcal M}^{2/d}}{\sum_e ||e||_{\mathcal M}^2}
    /// ```
    /// where
    ///  - $`|K|_{\mathcal M}`$ is the volume element in metric space. It is computed
    ///    on a discrete mesh as the ratio of the volume in physical space to the minimum
    ///    of the volumes of the metrics at each vertex
    ///  - the sum on the denominator is performed over all the edges of the element
    ///  - $`\beta_d`$ is a normalization factor such that $`q = 1`$ of equilateral elements
    ///     - $`\beta_2 = 1 / (6\sqrt{2}) `$
    ///     - $`\beta_3 = \sqrt{3}/ 4 `$
    fn quality(&self) -> f64;

    /// Get the coordinates of the i-th vertex
    fn vert(&self, i: Idx) -> Point<D>;

    /// Get the coordinates of a vertex given its barycentric coordnates
    fn point(&self, x: &[f64]) -> Point<D>;

    /// Get the barycentric coordinates of a vertex in the element
    /// For a triangle in 3d, the barycentric coordinates are those of the projection
    /// of the vertex onto the plane of the triangle
    ///
    /// If the vertices are $`(x_0, \cdots, x_{d+1})`$ then the barycentric coordinates are
    /// $`(b_0, \cdot, b_{d+1})`$ such that
    /// ```math
    /// x = \sum_{i = 0}^d b_i x_i  
    /// ```
    /// with $`\sum_{i = 0}^d b_i = 1`$
    ///
    /// Numerically the following least squares problem is solved using a QR factorization
    /// ```math
    /// \begin{bmatrix}
    /// 1 & \cdots  & 1 \\
    /// x_{0, 0}& \cdots & x_{d+1, 0} \\
    /// \vdots \\
    /// x_{0, d}& \cdots & x_{d+1, d} \\
    /// \end{bmatrix}
    /// \begin{bmatrix} b_0  \\ \vdots \\ b_{d+1}\end{bmatrix}
    ///  =
    /// \begin{bmatrix} 1 \\ x_0 \\ \vdots \\ x_d\end{bmatrix}
    /// ```
    fn bcoords(&self, p: &Point<D>) -> Self::BCoords; // todo: improve

    /// Get the element normal
    fn scaled_normal(&self) -> Point<D>;

    /// Get the element normal
    fn normal(&self) -> Point<D> {
        let mut n = self.scaled_normal();
        n.normalize_mut();
        n
    }

    /// Get the i-th geometric face
    fn gface(&self, i: Idx) -> Self::Face;

    /// Gamma quality measure, ratio of inscribed radius to circumradius
    /// normalized to be between 0 and 1
    fn gamma(&self) -> f64;
}

#[derive(Clone, Copy, Debug)]
pub struct GTetrahedron<const D: usize, M: Metric<D>> {
    points: [Point<D>; 4],
    metrics: [M; 4],
}

impl<const D: usize, M: Metric<D>> GTetrahedron<D, M> {
    const DIM: f64 = 3.0;
    // Jacobian of the reference to equilateral transformation
    const J_EQ: Matrix3<f64> = Matrix3::new(
        1.0,
        -1. / SQRT_3,
        -1. / SQRT_6,
        0.,
        2. / SQRT_3,
        -1.0 / SQRT_6,
        0.,
        0.,
        3. / SQRT_6,
    );

    /// Get the jacobian of the transformation from the reference to the current element
    pub fn jacobian(&self) -> Matrix3<f64> {
        Matrix3::<f64>::new(
            self.points[1][0] - self.points[0][0],
            self.points[2][0] - self.points[0][0],
            self.points[3][0] - self.points[0][0],
            self.points[1][1] - self.points[0][1],
            self.points[2][1] - self.points[0][1],
            self.points[3][1] - self.points[0][1],
            self.points[1][2] - self.points[0][2],
            self.points[2][2] - self.points[0][2],
            self.points[3][2] - self.points[0][2],
        )
    }

    /// Compute the implied metric
    /// It can be computed using the Jacobian $`J`$ of the transformation from the
    /// reference unit-length element to the physical element as $`(J J^T)^{-1}`$ .
    /// $`J`$ can be decomposed as the product of
    ///  - the Jacobian $`J_0`$ of the transformation from the reference unit-length
    ///    element to the orthogonal element, stored as `Self::J_EQ`
    ///  - the Jacobian $`J_1`$ of the transformation from the orthogonal element to
    ///    the physical element
    ///
    /// (reference: Ph.D. P. Caplan, p. 35)
    pub fn implied_metric(&self) -> AnisoMetric3d {
        let j = self.jacobian() * Self::J_EQ;
        let m = j * j.transpose();
        let m = m.try_inverse().unwrap();
        AnisoMetric3d::from_mat(m)
    }
}

impl<const D: usize, M: Metric<D>> GElem<D, M> for GTetrahedron<D, M> {
    type Face = GTriangle<D, M>;
    type BCoords = Vector4<f64>;
    const IDEAL_VOL: f64 = 1.0 / (6.0 * std::f64::consts::SQRT_2);

    fn vert(&self, i: Idx) -> Point<D> {
        self.points[i as usize]
    }

    fn from_verts<I: Iterator<Item = (Point<D>, M)>>(mut points_n_metrics: I) -> Self {
        assert_eq!(D, 3);
        let p: [_; 4] = std::array::from_fn(|_| points_n_metrics.next().unwrap());
        assert!(points_n_metrics.next().is_none());
        Self {
            points: p.map(|x| x.0),
            metrics: p.map(|x| x.1),
        }
    }

    fn from_vert_and_face(point: &Point<D>, metric: &M, face: &Self::Face) -> Self {
        Self {
            points: [*point, face.points[0], face.points[1], face.points[2]],
            metrics: [*metric, face.metrics[0], face.metrics[1], face.metrics[2]],
        }
    }

    fn vol(&self) -> f64 {
        let e1 = self.points[1] - self.points[0];
        let e2 = self.points[2] - self.points[0];
        let e3 = self.points[3] - self.points[0];
        let n = e1.cross(&e2);
        (1. / 6.) * n.dot(&e3)
    }

    fn center(&self) -> Point<D> {
        (self.points[0] + self.points[1] + self.points[2] + self.points[3]) / 4.0
    }

    fn quality(&self) -> f64 {
        let m = M::min_metric(self.metrics.iter());

        let mut e1 = self.points[1] - self.points[0];
        let e2 = self.points[2] - self.points[0];
        let e3 = self.points[3] - self.points[0];
        let n = e1.cross(&e2);
        let vol = (1. / 6.) * n.dot(&e3);
        if vol < 0.0 {
            return -1.0;
        }

        let mut l = f64::powi(m.length(&e1), 2);
        l += f64::powi(m.length(&e2), 2);
        l += f64::powi(m.length(&e3), 2);
        e1 = self.points[1] - self.points[2];
        l += f64::powi(m.length(&e1), 2);
        e1 = self.points[2] - self.points[3];
        l += f64::powi(m.length(&e1), 2);
        e1 = self.points[3] - self.points[1];
        l += f64::powi(m.length(&e1), 2);

        let l = l / 6.0;
        let vol = vol / m.vol() / Self::IDEAL_VOL;

        f64::powf(vol, 2. / Self::DIM) / l
    }

    fn point(&self, x: &[f64]) -> Point<D> {
        if x.len() == 4 {
            x[0] * self.points[0]
                + x[1] * self.points[1]
                + x[2] * self.points[2]
                + x[3] * self.points[3]
        } else if x.len() == 3 {
            (1. - x[0] - x[1] - x[2]) * self.points[0]
                + x[0] * self.points[1]
                + x[1] * self.points[2]
                + x[2] * self.points[3]
        } else {
            unreachable!();
        }
    }

    fn bcoords(&self, p: &Point<D>) -> Self::BCoords {
        let a = Matrix4::new(
            1.0,
            1.0,
            1.0,
            1.0,
            self.points[0][0],
            self.points[1][0],
            self.points[2][0],
            self.points[3][0],
            self.points[0][1],
            self.points[1][1],
            self.points[2][1],
            self.points[3][1],
            self.points[0][2],
            self.points[1][2],
            self.points[2][2],
            self.points[3][2],
        );
        let b = Vector4::new(1., p[0], p[1], p[2]);
        let decomp = a.lu();
        decomp.solve(&b).unwrap()
    }

    fn scaled_normal(&self) -> Point<D> {
        unreachable!();
    }

    fn gface(&self, i: Idx) -> Self::Face {
        match i {
            0 => GTriangle {
                points: [self.points[1], self.points[2], self.points[3]],
                metrics: [self.metrics[1], self.metrics[2], self.metrics[3]],
            },
            1 => GTriangle {
                points: [self.points[2], self.points[0], self.points[3]],
                metrics: [self.metrics[2], self.metrics[0], self.metrics[3]],
            },
            2 => GTriangle {
                points: [self.points[0], self.points[1], self.points[3]],
                metrics: [self.metrics[0], self.metrics[1], self.metrics[3]],
            },
            3 => GTriangle {
                points: [self.points[0], self.points[2], self.points[1]],
                metrics: [self.metrics[0], self.metrics[2], self.metrics[1]],
            },
            _ => unreachable!(),
        }
    }

    fn gamma(&self) -> f64 {
        let vol = self.vol();
        if vol < f64::EPSILON {
            return 0.0;
        }

        let a = self.points[1] - self.points[0];
        let b = self.points[2] - self.points[0];
        let c = self.points[3] - self.points[0];

        let aa = self.points[3] - self.points[2];
        let bb = self.points[3] - self.points[1];
        let cc = self.points[2] - self.points[1];

        let la = a.norm_squared();
        let lb = b.norm_squared();
        let lc = c.norm_squared();
        let laa = aa.norm_squared();
        let lbb = bb.norm_squared();
        let lcc = cc.norm_squared();

        let lalaa = (la * laa).sqrt();
        let lblbb = (lb * lbb).sqrt();
        let lclcc = (lc * lcc).sqrt();

        let tmp = (lalaa + lblbb + lclcc)
            * (lalaa + lblbb - lclcc)
            * (lalaa - lblbb + lclcc)
            * (-lalaa + lblbb + lclcc);

        // This happens when the 4 points are (nearly) co-planar
        // => R is actually undetermined but the quality is (close to) zero
        if tmp < f64::EPSILON {
            return 0.0;
        }

        let r = tmp.sqrt() / 24.0 / vol;

        let s1 = self.gface(0).vol();
        let s2 = self.gface(1).vol();
        let s3 = self.gface(2).vol();
        let s4 = self.gface(3).vol();
        let rho = 9.0 * vol / (s1 + s2 + s3 + s4);

        rho / r
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GTriangle<const D: usize, M: Metric<D>> {
    points: [Point<D>; 3],
    metrics: [M; 3],
}

impl<const D: usize, M: Metric<D>> GTriangle<D, M> {
    const DIM: f64 = 2.0;
    // Jacobian of the reference to equilateral transformation
    const J_EQ: Matrix2<f64> = Matrix2::new(1.0, -1. / SQRT_3, 0., 2. / SQRT_3);

    /// Get the jacobian of the transformation from the reference to the current element
    pub fn jacobian(&self) -> Matrix2<f64> {
        Matrix2::<f64>::new(
            self.points[1][0] - self.points[0][0],
            self.points[2][0] - self.points[0][0],
            self.points[1][1] - self.points[0][1],
            self.points[2][1] - self.points[0][1],
        )
    }

    /// Compute the implied metric
    /// It can be computed using the Jacobian $`J`$ of the transformation from the
    /// reference unit-length element to the physical element as $`(J J^T)^{-1}`$ .
    /// $`J`$ can be decomposed as the product of
    ///  - the Jacobian $`J_0`$ of the transformation from the reference unit-length
    ///    element to the orthogonal element, stored as `Self::J_EQ`
    ///  - the Jacobian $`J_1`$ of the transformation from the orthogonal element to
    ///    the physical element
    ///
    /// (reference: Ph.D. P. Caplan, p. 35)
    pub fn implied_metric(&self) -> AnisoMetric2d {
        let j = self.jacobian() * Self::J_EQ;
        let m = j * j.transpose();
        let m = m.try_inverse().unwrap();
        AnisoMetric2d::from_mat(m)
    }

    // Get the edges
    pub fn edge(&self, i: Idx) -> Point<D> {
        match i {
            0 => self.points[1] - self.points[0],
            1 => self.points[2] - self.points[0],
            2 => self.points[2] - self.points[1],
            _ => unreachable!(),
        }
    }

    fn cross_norm(e1: &Point<D>, e2: &Point<D>) -> f64 {
        if D == 2 {
            (e1[0] * e2[1] - e1[1] * e2[0]).abs()
        } else {
            let n = e1.cross(e2);
            n.norm()
        }
    }
}

impl<const D: usize, M: Metric<D>> GElem<D, M> for GTriangle<D, M> {
    type Face = GEdge<D, M>;
    type BCoords = Vector3<f64>;
    const IDEAL_VOL: f64 = SQRT_3 / 4.;

    fn vert(&self, i: Idx) -> Point<D> {
        self.points[i as usize]
    }

    fn from_verts<I: Iterator<Item = (Point<D>, M)>>(mut points_n_metrics: I) -> Self {
        let p: [_; 3] = std::array::from_fn(|_| points_n_metrics.next().unwrap());
        assert!(points_n_metrics.next().is_none());
        Self {
            points: p.map(|x| x.0),
            metrics: p.map(|x| x.1),
        }
    }

    fn from_vert_and_face(point: &Point<D>, metric: &M, face: &Self::Face) -> Self {
        Self {
            points: [*point, face.points[0], face.points[1]],
            metrics: [*metric, face.metrics[0], face.metrics[1]],
        }
    }

    fn vol(&self) -> f64 {
        let e1 = self.points[1] - self.points[0];
        let e2 = self.points[2] - self.points[0];
        if D == 2 {
            // <0 if not properly ordered
            0.5 * (e1[0] * e2[1] - e1[1] * e2[0])
        } else {
            0.5 * Self::cross_norm(&e1, &e2)
        }
    }

    fn center(&self) -> Point<D> {
        (self.points[0] + self.points[1] + self.points[2]) / 3.0
    }

    fn quality(&self) -> f64 {
        let m = M::min_metric(self.metrics.iter());

        let mut e1 = self.points[1] - self.points[0];
        let e2 = self.points[2] - self.points[0];
        let vol = if D == 2 {
            0.5 * (e1[0] * e2[1] - e1[1] * e2[0])
        } else {
            let n = e1.cross(&e2);
            0.5 * n.norm()
        };

        let mut l = f64::powi(m.length(&e1), 2);
        l += f64::powi(m.length(&e2), 2);
        e1 = self.points[1] - self.points[2];
        l += f64::powi(m.length(&e1), 2);

        let l = l / 3.0;
        let vol = vol / m.vol() / Self::IDEAL_VOL;

        f64::powf(vol, 2. / Self::DIM) / l
    }

    fn point(&self, x: &[f64]) -> Point<D> {
        if x.len() == 3 {
            x[0] * self.points[0] + x[1] * self.points[1] + x[2] * self.points[2]
        } else if x.len() == 2 {
            (1. - x[0] - x[1]) * self.points[0] + x[0] * self.points[1] + x[1] * self.points[2]
        } else {
            unreachable!();
        }
    }

    fn bcoords(&self, point: &Point<D>) -> Self::BCoords {
        let p0 = &self.points[0];
        let p1 = &self.points[1];
        let p2 = &self.points[2];

        if D == 2 {
            let a = Matrix3::new(1.0, 1.0, 1.0, p0[0], p1[0], p2[0], p0[1], p1[1], p2[1]);
            let b = Vector3::new(1., point[0], point[1]);
            let decomp = a.lu();
            decomp.solve(&b).unwrap()
        } else {
            let u = p1 - p0;
            let v = p2 - p0;
            let n = u.cross(&v);
            let w = point - p0;
            let nrm = n.norm_squared();
            let gamma = u.cross(&w).dot(&n) / nrm;
            let beta = w.cross(&v).dot(&n) / nrm;
            Vector3::new(1.0 - beta - gamma, beta, gamma)
        }
    }

    fn scaled_normal(&self) -> Point<D> {
        if D == 3 {
            let e0 = self.points[1] - self.points[0];
            let e1 = self.points[2] - self.points[0];
            0.5 * e0.cross(&e1)
        } else {
            unreachable!();
        }
    }

    fn gface(&self, i: Idx) -> Self::Face {
        match i {
            0 => GEdge {
                points: [self.points[1], self.points[2]],
                metrics: [self.metrics[1], self.metrics[2]],
            },
            1 => GEdge {
                points: [self.points[2], self.points[0]],
                metrics: [self.metrics[2], self.metrics[0]],
            },
            2 => GEdge {
                points: [self.points[0], self.points[1]],
                metrics: [self.metrics[0], self.metrics[1]],
            },
            _ => unreachable!(),
        }
    }

    fn gamma(&self) -> f64 {
        let mut a = self.points[2] - self.points[1];
        let mut b = self.points[0] - self.points[2];
        let mut c = self.points[1] - self.points[0];

        a.normalize_mut();
        b.normalize_mut();
        c.normalize_mut();

        let sina = Self::cross_norm(&b, &c);
        let sinb = Self::cross_norm(&a, &c);
        let sinc = Self::cross_norm(&a, &b);

        let tmp = sina + sinb + sinc;
        if tmp < 1e-12 {
            0.0
        } else {
            4.0 * sina * sinb * sinc / tmp
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GEdge<const D: usize, M: Metric<D>> {
    points: [Point<D>; 2],
    metrics: [M; 2],
}

impl<const D: usize, M: Metric<D>> GEdge<D, M> {
    const DIM: f64 = 1.0;
}

impl<const D: usize, M: Metric<D>> GElem<D, M> for GEdge<D, M> {
    type Face = GVertex<D, M>;
    type BCoords = Vector2<f64>;
    const IDEAL_VOL: f64 = 1.0;

    fn vert(&self, i: Idx) -> Point<D> {
        self.points[i as usize]
    }

    fn from_verts<I: Iterator<Item = (Point<D>, M)>>(mut points_n_metrics: I) -> Self {
        let p: [_; 2] = std::array::from_fn(|_| points_n_metrics.next().unwrap());
        assert!(points_n_metrics.next().is_none());
        Self {
            points: p.map(|x| x.0),
            metrics: p.map(|x| x.1),
        }
    }

    fn from_vert_and_face(point: &Point<D>, metric: &M, face: &Self::Face) -> Self {
        Self {
            points: [*point, face.points[0]],
            metrics: [*metric, face.metrics[0]],
        }
    }

    fn vol(&self) -> f64 {
        let e1 = self.points[1] - self.points[0];
        e1.norm()
    }

    fn center(&self) -> Point<D> {
        (self.points[0] + self.points[1]) / 2.0
    }

    fn quality(&self) -> f64 {
        // Just to suppress warnings
        Self::IDEAL_VOL / Self::DIM
    }

    fn point(&self, x: &[f64]) -> Point<D> {
        if x.len() == 2 {
            x[0] * self.points[0] + x[1] * self.points[1]
        } else if x.len() == 1 {
            (1. - x[0]) * self.points[0] + x[0] * self.points[1]
        } else {
            unreachable!();
        }
    }

    fn bcoords(&self, _: &Point<D>) -> Self::BCoords {
        unreachable!();
    }

    fn scaled_normal(&self) -> Point<D> {
        if D == 2 {
            let e0 = self.points[1] - self.points[0];
            let mut n = Point::<D>::zeros();
            n[0] = e0[1];
            n[1] = -e0[0];
            n
        } else {
            unreachable!();
        }
    }

    fn gface(&self, _i: Idx) -> Self::Face {
        unreachable!();
    }

    fn gamma(&self) -> f64 {
        1.0
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GVertex<const D: usize, M: Metric<D>> {
    points: [Point<D>; 1],
    metrics: [M; 1],
}

impl<const D: usize, M: Metric<D>> GVertex<D, M> {
    const DIM: f64 = 0.0;
}

impl<const D: usize, M: Metric<D>> GElem<D, M> for GVertex<D, M> {
    type Face = Self;
    type BCoords = Vector1<f64>;
    const IDEAL_VOL: f64 = 1.0;

    fn vert(&self, i: Idx) -> Point<D> {
        self.points[i as usize]
    }

    fn from_verts<I: Iterator<Item = (Point<D>, M)>>(mut points_n_metrics: I) -> Self {
        let (p, m) = points_n_metrics.next().unwrap();
        assert!(points_n_metrics.next().is_none());
        Self {
            points: [p],
            metrics: [m],
        }
    }

    fn from_vert_and_face(_point: &Point<D>, _metric: &M, _face: &Self::Face) -> Self {
        unreachable!();
    }

    fn vol(&self) -> f64 {
        1.0
    }

    fn center(&self) -> Point<D> {
        self.points[0]
    }

    fn quality(&self) -> f64 {
        // Just to suppress warnings
        Self::IDEAL_VOL / Self::DIM
    }

    fn point(&self, _x: &[f64]) -> Point<D> {
        unreachable!()
    }

    fn bcoords(&self, _p: &Point<D>) -> Self::BCoords {
        unreachable!();
    }

    fn scaled_normal(&self) -> Point<D> {
        unreachable!();
    }

    fn gface(&self, _i: Idx) -> Self::Face {
        unreachable!();
    }

    fn gamma(&self) -> f64 {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::{Matrix2, Matrix3};

    use crate::{
        mesh::{GElem, GTetrahedron, GTriangle, Point},
        metric::{AnisoMetric2d, AnisoMetric3d, IsoMetric},
    };

    #[test]
    fn test_vol_2d() {
        let p0 = Point::<2>::new(0., 0.);
        let p1 = Point::<2>::new(1., 0.);
        let p2 = Point::<2>::new(0., 1.);
        let points = [p0, p1, p2];

        let m = IsoMetric::<2>::from(1.0);
        let tri = GTriangle::from_verts(points.iter().map(|x| (*x, m)));
        assert!(f64::abs(tri.vol() - 1. / 2.) < 1e-12);

        let points = [p0, p2, p1];
        let tri = GTriangle::from_verts(points.iter().map(|x| (*x, m)));
        assert!(f64::abs(tri.vol() + 1. / 2.) < 1e-12);
    }

    #[test]
    fn test_implied_metric_2d() {
        let h0 = 10.;
        let h1 = 0.1;
        let p0 = Point::<2>::new(0., 0.);
        let p1 = Point::<2>::new(h0, 0.);
        let p2 = Point::<2>::new(0.5 * h0, 0.5 * f64::sqrt(3.0) * h1);
        let points = [p0, p1, p2];

        let metric = IsoMetric::<2>::from(1.0);

        let tri = GTriangle::from_verts(points.iter().map(|x| (*x, metric)));
        let m: Vec<f64> = tri.implied_metric().into_iter().collect();
        assert!(f64::abs(m[0] - 1. / h0.powi(2)) < 1e-8);
        assert!(f64::abs(m[1] - 1. / h1.powi(2)) < 1e-8);
        assert!(f64::abs(m[2] - 0.) < 1e-8);

        let theta = 0.25 * std::f64::consts::PI;
        let m = Matrix2::new(theta.cos(), -theta.sin(), theta.sin(), theta.cos());
        let p1 = m * p1;
        let p2 = m * p2;
        let points = [p0, p1, p2];

        let tri = GTriangle::from_verts(points.iter().map(|x| (*x, metric)));
        let m: Vec<f64> = tri.implied_metric().into_iter().collect();

        let s0 = h0 * 0.5 * f64::sqrt(2.0) * Point::<2>::new(1., 1.);
        let s1 = h1 * 0.5 * f64::sqrt(2.0) * Point::<2>::new(-1., 1.);
        let m_ref: Vec<f64> = AnisoMetric2d::from_sizes(&s0, &s1).into_iter().collect();

        assert!(f64::abs(m[0] - m_ref[0]) < 1e-8);
        assert!(f64::abs(m[1] - m_ref[1]) < 1e-8);
        assert!(f64::abs(m[2] - m_ref[2]) < 1e-8);
    }

    #[test]
    fn test_vol_3d() {
        let p0 = Point::<3>::new(0., 0., 0.);
        let p1 = Point::<3>::new(1., 0., 0.);
        let p2 = Point::<3>::new(0., 1., 0.);
        let p3 = Point::<3>::new(0., 0., 1.);
        let points = [p0, p1, p2, p3];

        let m = IsoMetric::<3>::from(1.0);

        let tet = GTetrahedron::from_verts(points.iter().map(|x| (*x, m)));
        assert!(f64::abs(tet.vol() - 1. / 6.) < 1e-12);

        let points = [p1, p0, p2, p3];
        let tet = GTetrahedron::from_verts(points.iter().map(|x| (*x, m)));
        assert!(f64::abs(tet.vol() + 1. / 6.) < 1e-12);

        let points = [p0, p1, p2];
        let tri = GTriangle::from_verts(points.iter().map(|x| (*x, m)));
        assert!(f64::abs(tri.vol() - 1. / 2.) < 1e-12);

        let points = [p0, p2, p1];
        let tri = GTriangle::from_verts(points.iter().map(|x| (*x, m)));
        assert!(f64::abs(tri.vol() - 1. / 2.) < 1e-12);

        let points = [p0, p1, p3];
        let tri = GTriangle::from_verts(points.iter().map(|x| (*x, m)));
        assert!(f64::abs(tri.vol() - 1. / 2.) < 1e-12);

        let points = [p0, p2, p3];
        let tri = GTriangle::from_verts(points.iter().map(|x| (*x, m)));
        assert!(f64::abs(tri.vol() - 1. / 2.) < 1e-12);
    }

    #[test]
    fn test_implied_metric_3d() {
        let h0 = 10.;
        let h1 = 0.1;
        let h2 = 2.;

        let p0 = Point::<3>::new(0., 0., 0.);
        let p1 = Point::<3>::new(h0, 0., 0.);
        let p2 = Point::<3>::new(0.5 * h0, 0.5 * f64::sqrt(3.0) * h1, 0.);
        let p3 = Point::<3>::new(
            0.5 * h0,
            0.5 * h1 / f64::sqrt(3.0),
            h2 * f64::sqrt(2.) / f64::sqrt(3.),
        );
        let points = [p0, p1, p2, p3];

        let metric = IsoMetric::<3>::from(1.0);

        let tet = GTetrahedron::from_verts(points.iter().map(|x| (*x, metric)));
        let m: Vec<f64> = tet.implied_metric().into_iter().collect();
        assert!(f64::abs(m[0] - 1. / h0.powi(2)) < 1e-8);
        assert!(f64::abs(m[1] - 1. / h1.powi(2)) < 1e-8);
        assert!(f64::abs(m[2] - 1. / h2.powi(2)) < 1e-8);
        assert!(f64::abs(m[3] - 0.) < 1e-8);
        assert!(f64::abs(m[4] - 0.) < 1e-8);
        assert!(f64::abs(m[5] - 0.) < 1e-8);

        let m = Matrix3::new(
            1. / f64::sqrt(3.),
            1. / f64::sqrt(6.),
            -1. / f64::sqrt(2.),
            1. / f64::sqrt(3.),
            1. / f64::sqrt(6.),
            1. / f64::sqrt(2.),
            1. / f64::sqrt(3.),
            -2. / f64::sqrt(6.),
            0.,
        );

        let p1 = m * p1;
        let p2 = m * p2;
        let p3 = m * p3;
        let points = [p0, p1, p2, p3];

        let tet = GTetrahedron::from_verts(points.iter().map(|x| (*x, metric)));
        let m: Vec<f64> = tet.implied_metric().into_iter().collect();

        let s0 = h0 / f64::sqrt(3.) * Point::<3>::new(1., 1., 1.);
        let s1 = h1 / f64::sqrt(6.) * Point::<3>::new(1., 1., -2.);
        let s2 = h2 / f64::sqrt(2.) * Point::<3>::new(-1., 1., 0.);

        let m_ref: Vec<f64> = AnisoMetric3d::from_sizes(&s0, &s1, &s2)
            .into_iter()
            .collect();

        for i in 0..6 {
            assert!(f64::abs(m[i] - m_ref[i]) < 1e-8);
        }
    }

    // TODO: test qualities & barycentric coordinates
}
