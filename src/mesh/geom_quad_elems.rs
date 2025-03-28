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

    #[allow(dead_code)]
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
                points: [self.points[1], self.points[2], self.points[3]],
                metrics: [self.metrics[1], self.metrics[2], self.metrics[3]],
            },
            1 => GQuadraticEdge {
                points: [self.points[2], self.points[0], self.points[4]],
                metrics: [self.metrics[2], self.metrics[0], self.metrics[4]],
            },
            2 => GQuadraticEdge {
                points: [self.points[0], self.points[1], self.points[5]],
                metrics: [self.metrics[0], self.metrics[1], self.metrics[5]],
            },
            _ => unreachable!(),
        }
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
}

#[cfg(test)]
mod tests {
    use nalgebra::SMatrix;

    use super::*;
    use crate::mesh::Point;
    use crate::metric::{AnisoMetric, AnisoMetric3d};
    use std::error::Error;
    use std::fmt;
    use std::fmt::Debug;
    use std::iter::IntoIterator;

    // Mock a simple 2D metric type to test
    #[derive(Clone, Copy, Debug, Default)]
    pub struct SimpleMetric {
        values: [f64; 4], // For a 2D metric, we can use a 2x2 matrix flattened into an array.
    }

    impl SimpleMetric {
        fn new(values: [f64; 4]) -> Self {
            SimpleMetric { values }
        }
    }

    impl fmt::Display for SimpleMetric {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(
                f,
                "[{}, {}, {}, {}]",
                self.values[0], self.values[1], self.values[2], self.values[3]
            )
        }
    }

    impl IntoIterator for SimpleMetric {
        type Item = f64;
        type IntoIter = std::array::IntoIter<f64, 4>; // Since the array is fixed size 4

        fn into_iter(self) -> Self::IntoIter {
            self.values.into_iter() // Converts the `values` array into an iterator
        }
    }

    impl Metric<2> for SimpleMetric {
        const N: usize = 2;

        fn from_slice(m: &[f64]) -> Self {
            assert_eq!(m.len(), 4);
            SimpleMetric {
                values: [m[0], m[1], m[2], m[3]],
            }
        }

        fn check(&self) -> Result<(), Box<dyn Error>> {
            if self.values.iter().all(|&v| v > 0.0) {
                Ok(()) // Return Ok if all values are positive
            } else {
                Err("Invalid metric: all values must be positive.".into()) // Return an error with a boxed string
            }
        }

        fn length(&self, e: &Point<2>) -> f64 {
            let x = e[0];
            let y = e[1];
            (self.values[0] * x * x
                + self.values[1] * x * y
                + self.values[2] * x * y
                + self.values[3] * y * y)
                .sqrt()
        }

        fn vol(&self) -> f64 {
            (self.values[0] * self.values[3] - self.values[1] * self.values[2]).sqrt()
        }

        fn interpolate<'a, I: Iterator<Item = (f64, &'a Self)>>(weights_and_metrics: I) -> Self {
            let mut sum_weights = 0.0;
            let mut interpolated_values = [0.0; 4];
            for (weight, metric) in weights_and_metrics {
                sum_weights += weight;
                for i in 0..4 {
                    interpolated_values[i] += metric.values[i] * weight;
                }
            }
            for i in 0..4 {
                interpolated_values[i] /= sum_weights;
            }
            SimpleMetric::new(interpolated_values)
        }

        fn sizes(&self) -> [f64; 2] {
            [self.values[0].sqrt(), self.values[3].sqrt()]
        }

        fn scale(&mut self, s: f64) {
            for v in &mut self.values {
                *v *= s;
            }
        }

        fn scale_with_bounds(&mut self, s: f64, h_min: f64, h_max: f64) {
            for v in &mut self.values {
                *v = (*v * s).clamp(h_min, h_max);
            }
        }

        fn intersect(&self, other: &Self) -> Self {
            let mut res = [0.0; 4];
            for i in 0..4 {
                res[i] = self.values[i].min(other.values[i]);
            }
            SimpleMetric::new(res)
        }

        fn span(&self, _e: &Point<2>, beta: f64, t: f64) -> Self {
            let mut new_values = self.values;
            for v in &mut new_values {
                *v *= beta * t;
            }
            SimpleMetric::new(new_values)
        }

        fn differs_from(&self, other: &Self, tol: f64) -> bool {
            self.values
                .iter()
                .zip(other.values.iter())
                .any(|(a, b)| (a - b).abs() > tol)
        }

        fn step(&self, other: &Self) -> (f64, f64) {
            let min = self.vol().min(other.vol());
            let max = self.vol().max(other.vol());
            (min, max)
        }

        fn control_step(&mut self, other: &Self, f: f64) {
            for i in 0..4 {
                self.values[i] = self.values[i] * f + other.values[i] * (1.0 - f);
            }
        }

        fn edge_length(p0: &Point<2>, m0: &Self, p1: &Point<2>, m1: &Self) -> f64 {
            let e = *p1 - *p0;
            let l0 = m0.length(&e);
            let l1 = m1.length(&e);
            let r = l0 / l1;
            if (r - 1.0).abs() > 0.01 {
                l0 * (r - 1.0) / r / f64::ln(r)
            } else {
                l0
            }
        }

        fn min_metric<'a, I: Iterator<Item = &'a Self>>(mut metrics: I) -> &'a Self {
            let m = metrics.next().unwrap();
            let mut vol = m.vol();
            let mut res = m;
            for m in metrics {
                let volm = m.vol();
                if volm < vol {
                    res = m;
                    vol = volm;
                }
            }
            res
        }
    }

    #[test]
    fn test_simple_metric_creation() {
        let metric = SimpleMetric::from_slice(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(metric.values, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_check_valid_metric() {
        let metric = SimpleMetric::from_slice(&[1.0, 2.0, 3.0, 4.0]);
        assert!(metric.check().is_ok());
    }

    #[test]
    fn test_check_invalid_metric() {
        let metric = SimpleMetric::from_slice(&[-1.0, 2.0, 3.0, 4.0]);
        assert!(metric.check().is_err());
    }

    #[test]
    fn test_length() {
        let metric = SimpleMetric::from_slice(&[1.0, 0.0, 0.0, 1.0]);
        let point = Point::<2>::from([3.0, 4.0]);
        assert_eq!(metric.length(&point), 5.0);
    }

    #[test]
    fn test_vol() {
        let metric = SimpleMetric::from_slice(&[1.0, 0.0, 0.0, 1.0]);
        assert_eq!(metric.vol(), 1.0);
    }

    #[test]
    fn test_interpolate() {
        let metric1 = SimpleMetric::from_slice(&[1.0, 0.0, 0.0, 1.0]);
        let metric2 = SimpleMetric::from_slice(&[2.0, 0.0, 0.0, 2.0]);
        let interpolated =
            SimpleMetric::interpolate(vec![(0.5, &metric1), (0.5, &metric2)].into_iter());
        assert_eq!(interpolated.values, [1.5, 0.0, 0.0, 1.5]);
    }

    #[test]
    fn test_quad_tri() {
        let mat = SMatrix::<f64, 3, 3>::new(1., 0.1, 0.2, 0.1, 1.0, 0.3, 0.2, 0.3, 1.0);
        let metric = AnisoMetric3d::from_mat(mat);

        let points = [
            Point::<3>::from([0., 0., 0.]),
            Point::<3>::from([1., 0., 0.]),
            Point::<3>::from([0., 1., 0.]),
            Point::<3>::from([0.5, 0.5, 0.]),
            Point::<3>::from([0.5, 0., 0.5]),
            Point::<3>::from([0., 0.5, 0.5]),
        ];

        let points_and_metrics = points.iter().map(|&p| (p, metric));
        let triangle = GQuadraticTriangle::from_verts(points_and_metrics);
        assert_eq!(triangle.vert(0), points[0]);
    }
}
