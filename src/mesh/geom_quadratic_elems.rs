use crate::{mesh::Point, metric::Metric, Idx};
use nalgebra::{Matrix2, Matrix3, Vector1, Vector2, Vector3, Vector4};
use std::fmt::Debug;

#[allow(dead_code)]
pub trait AsSliceF64 {
    fn as_slice_f64(&self) -> &[f64];
}

#[allow(dead_code)]
impl AsSliceF64 for Vector4<f64> {
    fn as_slice_f64(&self) -> &[f64] {
        self.as_slice()
    }
}

#[allow(dead_code)]
impl AsSliceF64 for Vector3<f64> {
    fn as_slice_f64(&self) -> &[f64] {
        self.as_slice()
    }
}

#[allow(dead_code)]
impl AsSliceF64 for Vector2<f64> {
    fn as_slice_f64(&self) -> &[f64] {
        self.as_slice()
    }
}

#[allow(dead_code)]
impl AsSliceF64 for Vector1<f64> {
    fn as_slice_f64(&self) -> &[f64] {
        self.as_slice()
    }
}

const SQRT_3: f64 = 1.732_050_807_568_877_2;

pub trait GQuadraticElem<M: Metric<3>>: Clone + Copy + Debug + Send {
    type Face: GQuadraticElem<M>;
    type BCoords: AsSliceF64 + Debug;
    const IDEAL_VOL: f64;

    #[allow(dead_code)]
    /// Create a `GQuadraticElem` from its vertices and the metric at each vertex
    fn from_verts<I: Iterator<Item = (Point<3>, M)>>(points_n_metrics: I) -> Self;

    fn center(&self) -> Point<3>;

    #[allow(dead_code)]
    /// Get the coordinates of the i-th vertex
    fn vert(&self, i: Idx) -> Point<3>;

    #[allow(dead_code)]
    /// Get the coordinates of a vertex given its barycentric coordinates
    fn point(&self, x: &[f64]) -> Point<3>;

    #[allow(dead_code)]
    /// Get the i-th geometric face
    fn gface(&self, i: Idx) -> Self::Face;

    /// Get the element normal
    fn scaled_normal(&self) -> Point<3>;

    /// Get the element normal
    fn normal(&self) -> Point<3> {
        let mut n = self.scaled_normal();
        n.normalize_mut();
        n
    }

    // fn project(&self, p: Point<3>, x_init: &[f64], max_iter: i32, tol: f64) -> Point<3>;
    // fn distance_to_minimize_for_proj(&self, x: &[f64], p: Point<3>) -> f64;
    // fn gradient_distance_to_minimize_for_projection(&self, x: &[f64], p: Point<3>) -> [f64];
    // fn hessian_distance_to_minimize_for_projection(&self, x: &[f64], p: Point<3>) -> Matrix2<f64>;
}

/// GQuadraticTriangle
#[derive(Clone, Copy, Debug)]
pub struct GQuadraticTriangle<M: Metric<3>> {
    points: [Point<3>; 6],
    metrics: [M; 6],
}

impl<M: Metric<3>> GQuadraticTriangle<M> {
    /// Get the jacobian of the transformation from the reference to the current element
    pub fn jacobian(&self, x: &[f64]) -> Matrix3<f64> {

        let p0 = Vector3::from(self.points[0]);
        let p1 = Vector3::from(self.points[1]);
        let p2 = Vector3::from(self.points[2]);
        let p3 = Vector3::from(self.points[3]);
        let p4 = Vector3::from(self.points[4]);
        let p5 = Vector3::from(self.points[5]);
        if x.len() == 2 {
            let u = x[0];
            let v = x[1];
            let w = 1. - x[0] - x[1];

            // let du = 2.0 * u * (p0 + p2 - 2.0 * p5) + 2.0 * (p5 - p2) + 2.0 * v * (p2 + p3 - p4 - p5);
            // let dv = 2.0 * v * (p1 + p2 - 2.0 * p4) + 2.0 * (p4 - p2) + 2.0 * u * (p2 + p3 - p4 - p5);

            let du = 2. * p0 + 2. * v * p3 + 2. * w * p5;
            let dv = 2. * p1 + 2. * u * p3 + 2. * w * p4;
            let dw = 2. * p2 + 2. * v * p4 + 2. * u * p5;

            Matrix3::from_columns(&[du, dv, dw])
        } else if x.len() == 3 {

            let u = x[0];
            let v = x[1];
            let w = x[2];

            let du = 2. * p0 + 2. * v * p3 + 2. * w * p5;
            let dv = 2. * p1 + 2. * u * p3 + 2. * w * p4;
            let dw = 2. * p2 + 2. * v * p4 + 2. * u * p5;

            Matrix3::from_columns(&[du, dv, dw])

        } else {
            unreachable!();
        }
    }

    pub fn gradient_mapping_bezier_without_w(&self, x: &[f64]) -> (Vector3<f64>, Vector3<f64>) {
        let u = x[0];
        let v = x[1];
        let p0 = Vector3::from(self.points[0]);
        let p1 = Vector3::from(self.points[1]);
        let p2 = Vector3::from(self.points[2]);
        let p3 = Vector3::from(self.points[3]);
        let p4 = Vector3::from(self.points[4]);
        let p5 = Vector3::from(self.points[5]);

        let du = 2.0 * u * (p0 + p2 - 2.0 * p5) + 2.0 * (p5 - p2) + 2.0 * v * (p2 + p3 - p4 - p5);

        let dv = 2.0 * v * (p1 + p2 - 2.0 * p4) + 2.0 * (p4 - p2) + 2.0 * u * (p2 + p3 - p4 - p5);

        (du, dv)
    }

    /// Compute the Hessian (second partial derivatives) of the Bezier mapping
    pub fn hessian_mapping_bezier(&self) -> [[Vector3<f64>; 2]; 2] {
        let p0 = Vector3::from(self.points[0]);
        let p1 = Vector3::from(self.points[1]);
        let p2 = Vector3::from(self.points[2]);
        let p3 = Vector3::from(self.points[3]);
        let p4 = Vector3::from(self.points[4]);
        let p5 = Vector3::from(self.points[5]);

        let du2 = 2.0 * (p0 + p2 - 2.0 * p5);
        let dv2 = 2.0 * (p1 + p2 - 2.0 * p4);
        let dudv = 2.0 * (p2 + p3 - p4 - p5);

        [[du2, dudv], [dudv, dv2]]
    }

    #[allow(dead_code)]
    fn distance_to_minimize_for_proj(&self, x: &[f64], p: Point<3>) -> f64 {
        let position = self.point(x);
        (position - p).dot(&(position - p))
    }

    #[allow(dead_code)]
    fn gradient_distance_to_minimize_for_projection(&self, x: &[f64], p: Point<3>) -> [f64; 2] {
        let (du, dv) = self.gradient_mapping_bezier_without_w(x);
        let distance = self.point(x) - p;
        let grad_u = 2. * du.dot(&distance);
        let grad_v = 2. * dv.dot(&distance);
        [grad_u, grad_v]
    }

    #[allow(dead_code)]
    fn hessian_distance_to_minimize_for_projection(&self, x: &[f64], p: Point<3>) -> Matrix2<f64> {
        let (du, dv) = self.gradient_mapping_bezier_without_w(x);
        let h = self.hessian_mapping_bezier();
        let du2 = h[0][0];
        let dv2 = h[1][1];
        let dudv = h[0][1];
        let distance = self.point(x) - p;
        let h11 = 2. * du2.dot(&distance) + 2. * du.dot(&du);
        let h22 = 2. * dv2.dot(&distance) + 2. * dv.dot(&dv);
        let h12 = 2. * dudv.dot(&distance) + 2. * dv.dot(&du);
        Matrix2::new(h11, h12, h12, h22)
    }

    #[allow(dead_code)]
    fn project(&self, p: Point<3>, x_init: &[f64], max_iter: i32, tol: f64) -> Point<3> {
        // let u = x_init[0];
        // let v = x_init[1];
        let mut x = [x_init[0], x_init[1]];
        for _i in 0..max_iter {
            let grad_distance = self.gradient_distance_to_minimize_for_projection(&x, p);
            let grad = Vector2::new(grad_distance[0], grad_distance[1]);
            let hessian = self.hessian_distance_to_minimize_for_projection(&x, p);
            let delta = hessian.lu().solve(&grad).expect("Singular Matrix");
            x[0] -= delta[0];
            x[1] -= delta[1];

            // Calcul de l'erreur
            let err = grad.norm();
            if err < tol {
                return self.point(&x);
            }
        }
        // Si on n'a pas convergé dans max_iter, retourne la dernière estimation
        self.point(&x)
    }
}

impl<M: Metric<3>> GQuadraticElem<M> for GQuadraticTriangle<M> {
    type Face = GQuadraticEdge<M>;
    type BCoords = Vector3<f64>;
    const IDEAL_VOL: f64 = SQRT_3 / 4.;

    fn from_verts<I: Iterator<Item = (Point<3>, M)>>(mut points_n_metrics: I) -> Self {
        let p: [_; 6] = std::array::from_fn(|_| points_n_metrics.next().unwrap());
        assert!(points_n_metrics.next().is_none());
        Self {
            points: p.map(|x| x.0),
            metrics: p.map(|x| x.1),
        }
    }

    fn vert(&self, i: Idx) -> Point<3> {
        self.points[i as usize]
    }

    fn center(&self) -> Point<3> {
        (self.points[0]
            + self.points[1]
            + self.points[2]
            + self.points[3]
            + self.points[4]
            + self.points[5])
            / 6.0
    }

    fn point(&self, x: &[f64]) -> Point<3> {
        if x.len() == 3 {
            x[0].powf(2.) * self.points[0]
                + x[1].powf(2.) * self.points[1]
                + x[2].powf(2.) * self.points[2]
                + 2.0 * x[0] * x[1] * self.points[3]
                + 2.0 * x[0] * x[2] * self.points[5]
                + 2.0 * x[1] * x[2] * self.points[4]
        } else if x.len() == 2 {
            x[0].powf(2.) * self.points[0]
                + x[1].powf(2.) * self.points[1]
                + (1. - x[0] - x[1]).powf(2.) * self.points[2]
                + 2.0 * x[0] * x[1] * self.points[3]
                + 2.0 * (1. - x[0] - x[1]) * x[1] * self.points[4]
                + 2.0 * (1. - x[0] - x[1]) * x[0] * self.points[5]
        } else {
            unreachable!();
        }
    }

    fn gface(&self, i: Idx) -> Self::Face {
        match i {
            0 => GQuadraticEdge {
                points: [self.points[0], self.points[1], self.points[3]],
                metrics: [self.metrics[0], self.metrics[1], self.metrics[3]],
            },
            1 => GQuadraticEdge {
                points: [self.points[1], self.points[2], self.points[4]],
                metrics: [self.metrics[1], self.metrics[2], self.metrics[4]],
            },
            2 => GQuadraticEdge {
                points: [self.points[0], self.points[5], self.points[5]],
                metrics: [self.metrics[0], self.metrics[5], self.metrics[5]],
            },
            _ => unreachable!(),
        }
    }

    fn scaled_normal(&self) -> Point<3> {
        todo!();
        // let e0 = self.points[1] - self.points[3];
        // let e1 = self.points[3] - self.points[0];
        // let e2 = self.points[4] - self.points[5];
        // let e3 = self.points[2] - self.points[5];
        // let e4 = self.points[5] - self.points[0];
        // let e5 = self.points[4] - self.points[3];
        // let u = (2.0 / 3.0) * (e0 + e1 + e2);
        // let v = (2.0 / 3.0) * (e3 + e4 + e5);
        // let w = u.cross(&v);
        // let norm = w.norm();
        // w / norm
    }
}

/// GQuadraticEdge
#[derive(Clone, Copy, Debug)]
pub struct GQuadraticEdge<M: Metric<3>> {
    points: [Point<3>; 3],
    metrics: [M; 3],
}

impl<M: Metric<3>> GQuadraticEdge<M> {
    #[allow(dead_code)]
    const DIM: f64 = 1.0;

    pub fn gradient_bezier_curve(self, x: &[f64]) -> Vector3<f64> {
        2. * x[0] * (self.points[0] + self.points[1] - self.points[2])
            + 2. * (self.points[2] - self.points[1])
    }

    pub fn hessian_bezier_curve(self) -> Vector3<f64> {
        2. * (self.points[0] + self.points[1] - self.points[2])
    }
}

impl<M: Metric<3>> GQuadraticElem<M> for GQuadraticEdge<M> {
    type Face = GVertex<M>;
    type BCoords = Vector2<f64>;
    const IDEAL_VOL: f64 = 1.0;

    fn from_verts<I: Iterator<Item = (Point<3>, M)>>(mut points_n_metrics: I) -> Self {
        let p: [_; 3] = std::array::from_fn(|_| points_n_metrics.next().unwrap());
        assert!(points_n_metrics.next().is_none());
        Self {
            points: p.map(|x| x.0),
            metrics: p.map(|x| x.1),
        }
    }

    fn vert(&self, i: Idx) -> Point<3> {
        self.points[i as usize]
    }

    fn center(&self) -> Point<3> {
        (self.points[0] + self.points[1] + self.points[2]) / 3.0
    }

    fn point(&self, x: &[f64]) -> Point<3> {
        if x.len() == 2 {
            x[0].powf(2.) * self.points[0]
                + x[1].powf(2.) * self.points[1]
                + 2.0 * x[0] * x[1] * self.points[2]
        } else if x.len() == 1 {
            x[0].powf(2.) * self.points[0]
                + (1. - x[0]).powf(2.) * self.points[1]
                + 2.0 * x[0] * (1. - x[0]) * self.points[2]
        } else {
            unreachable!();
        }
    }

    fn gface(&self, i: Idx) -> Self::Face {
        match i {
            0 => GVertex {
                points: [self.points[0]],
                metrics: [self.metrics[0]],
            },
            1 => GVertex {
                points: [self.points[1]],
                metrics: [self.metrics[1]],
            },
            2 => GVertex {
                points: [self.points[2]],
                metrics: [self.metrics[2]],
            },
            _ => unreachable!(),
        }
    }

    fn scaled_normal(&self) -> Point<3> {
        todo!()
        // let e0 = self.points[1] - self.points[0];
        // let e1 = self.points[2] - self.points[0];
        // 0.5 * e0.cross(&e1)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GVertex<M: Metric<3>> {
    points: [Point<3>; 1],
    #[allow(dead_code)]
    metrics: [M; 1],
}

impl<M: Metric<3>> GVertex<M> {
    #[allow(dead_code)]
    const DIM: f64 = 0.0;
}

impl<M: Metric<3>> GQuadraticElem<M> for GVertex<M> {
    type Face = Self;
    type BCoords = Vector1<f64>;
    const IDEAL_VOL: f64 = 1.0;

    fn from_verts<I: Iterator<Item = (Point<3>, M)>>(mut points_n_metrics: I) -> Self {
        let (p, m) = points_n_metrics.next().unwrap();
        assert!(points_n_metrics.next().is_none());
        Self {
            points: [p],
            metrics: [m],
        }
    }

    fn vert(&self, i: Idx) -> Point<3> {
        self.points[i as usize]
    }

    fn center(&self) -> Point<3> {
        self.points[0]
    }

    fn point(&self, _x: &[f64]) -> Point<3> {
        unreachable!()
    }

    fn gface(&self, _i: Idx) -> Self::Face {
        unreachable!();
    }

    fn scaled_normal(&self) -> Point<3> {
        unreachable!();
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::SMatrix;

    use super::*;
    use crate::mesh::Point;
    use crate::metric::{AnisoMetric, AnisoMetric3d};

    #[test]
    fn test_quad_triangle() {
        let mat = SMatrix::<f64, 3, 3>::new(1., 0.1, 0.2, 0.1, 1.0, 0.3, 0.2, 0.3, 1.0);
        let metric = AnisoMetric3d::from_mat(mat);

        let points = [
            Point::<3>::from([1., 0., 0.]),
            Point::<3>::from([0., 1., 0.]),
            Point::<3>::from([0., 0., 1.]),
            Point::<3>::from([0.5, 0.5, 0.]),
            Point::<3>::from([0., 0.5, 0.5]),
            Point::<3>::from([0.5, 0., 0.5]),
        ];

        let points_and_metrics = points.iter().map(|&p| (p, metric));
        let triangle = GQuadraticTriangle::from_verts(points_and_metrics);

        assert_eq!(triangle.vert(0), points[0]);
        assert_eq!(
            triangle.center(),
            Point::<3>::new(1. / 3., 1. / 3., 1. / 3.)
        );
        let p = Point::<3>::new(1. / 3., 1. / 3., 1. / 3.);
        let x_init = [0.2, 0.2];
        let proj = triangle.project(p, &x_init, 20, 0.0000001);
        println!("{}", proj);
    }
}
