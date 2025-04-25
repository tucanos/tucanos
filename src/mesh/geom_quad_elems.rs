use crate::{mesh::Point, metric::Metric, Idx};
use nalgebra::{Matrix3, Vector1, Vector2, Vector3, Vector4};
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

pub trait GQuadElem<M: Metric<3>>: Clone + Copy + Debug + Send {
    type Face: GQuadElem<M>;
    type BCoords: AsSliceF64 + Debug;
    const IDEAL_VOL: f64;

    #[allow(dead_code)]
    /// Create a `GQuadElem` from its vertices and the metric at each vertex
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
        if x.len() == 2 {
            Matrix3::<f64>::new(
                2.0 * (1.0 - x[0] - x[1]) * self.points[0][0]
                    + 2.0 * x[0] * self.points[3][0]
                    + 2.0 * x[1] * self.points[5][0],
                2.0 * x[0] * self.points[1][0]
                    + 2.0 * (1.0 - x[0] - x[1]) * self.points[3][0]
                    + 2.0 * x[1] * self.points[4][0],
                2.0 * x[1] * self.points[1][0]
                    + 2.0 * (1.0 - x[0] - x[1]) * self.points[5][0]
                    + 2.0 * x[0] * self.points[4][0],
                2.0 * (1.0 - x[0] - x[1]) * self.points[0][1]
                    + 2.0 * x[0] * self.points[3][1]
                    + 2.0 * x[1] * self.points[5][1],
                2.0 * x[0] * self.points[1][1]
                    + 2.0 * (1.0 - x[0] - x[1]) * self.points[3][1]
                    + 2.0 * x[1] * self.points[4][1],
                2.0 * x[1] * self.points[1][1]
                    + 2.0 * (1.0 - x[0] - x[1]) * self.points[5][1]
                    + 2.0 * x[0] * self.points[4][1],
                2.0 * (1.0 - x[0] - x[1]) * self.points[0][2]
                    + 2.0 * x[0] * self.points[3][2]
                    + 2.0 * x[1] * self.points[5][2],
                2.0 * x[0] * self.points[1][2]
                    + 2.0 * (1.0 - x[0] - x[1]) * self.points[3][2]
                    + 2.0 * x[1] * self.points[4][2],
                2.0 * x[1] * self.points[1][2]
                    + 2.0 * (1.0 - x[0] - x[1]) * self.points[5][2]
                    + 2.0 * x[0] * self.points[4][2],
            )
        } else if x.len() == 3 {
            Matrix3::<f64>::new(
                2.0 * x[0] * self.points[0][0]
                    + 2.0 * x[1] * self.points[3][0]
                    + 2.0 * x[2] * self.points[5][0],
                2.0 * x[1] * self.points[1][0]
                    + 2.0 * x[0] * self.points[3][0]
                    + 2.0 * x[2] * self.points[4][0],
                2.0 * x[2] * self.points[1][0]
                    + 2.0 * x[0] * self.points[5][0]
                    + 2.0 * x[1] * self.points[4][0],
                2.0 * x[0] * self.points[0][1]
                    + 2.0 * x[1] * self.points[3][1]
                    + 2.0 * x[2] * self.points[5][1],
                2.0 * x[1] * self.points[1][1]
                    + 2.0 * x[0] * self.points[3][1]
                    + 2.0 * x[2] * self.points[4][1],
                2.0 * x[2] * self.points[1][1]
                    + 2.0 * x[0] * self.points[5][1]
                    + 2.0 * x[1] * self.points[4][1],
                2.0 * x[0] * self.points[0][2]
                    + 2.0 * x[1] * self.points[3][2]
                    + 2.0 * x[2] * self.points[5][2],
                2.0 * x[1] * self.points[1][2]
                    + 2.0 * x[0] * self.points[3][2]
                    + 2.0 * x[2] * self.points[4][2],
                2.0 * x[2] * self.points[1][2]
                    + 2.0 * x[0] * self.points[5][2]
                    + 2.0 * x[1] * self.points[4][2],
            )
        } else {
            unreachable!();
        }
    }
}

impl<M: Metric<3>> GQuadElem<M> for GQuadraticTriangle<M> {
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
            (1. - x[0] - x[1]).powf(2.) * self.points[0]
                + x[0].powf(2.) * self.points[1]
                + x[1].powf(2.) * self.points[2]
                + 2.0 * (1. - x[0] - x[1]) * x[0] * self.points[3]
                + 2.0 * (1. - x[0] - x[1]) * x[1] * self.points[5]
                + 2.0 * x[1] * x[2] * self.points[4]
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
        let e0 = self.points[1] - self.points[0];
        let e1 = self.points[2] - self.points[0];
        0.5 * e0.cross(&e1)
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
}

impl<M: Metric<3>> GQuadElem<M> for GQuadraticEdge<M> {
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
            (1. - x[0]).powf(2.) * self.points[0]
                + x[0].powf(2.) * self.points[1]
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
        let e0 = self.points[1] - self.points[0];
        let e1 = self.points[2] - self.points[0];
        0.5 * e0.cross(&e1)
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

impl<M: Metric<3>> GQuadElem<M> for GVertex<M> {
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
    }
}
