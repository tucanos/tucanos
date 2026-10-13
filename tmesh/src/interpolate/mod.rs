//! Interpolation
use crate::{
    Vertex,
    mesh::{GSimplex, Mesh},
    spatialindex::{ObjectIndex, PointIndex},
};

/// Interpolation method
pub enum InterpolationMethod {
    /// Nearest neighbor interpolation
    Nearest,
    /// Linear interpolation in the nearest element. If the barycentric coordinates
    /// of a point are within $`[-tol, 1 + tol]`$, they are used as is (i.e. the field
    /// is linearly extrapolated); otherwise the point is outside of the mesh and
    /// its negative barycentric coordinates are clamped to 0, which interpolates at
    /// a point of the nearest element
    Linear(f64),
}

/// Interpolator
pub struct Interpolator<'a, const D: usize, M: Mesh<D>> {
    /// Mesh from which the data in interpolated
    mesh: &'a M,
    /// Interpolation method
    method: InterpolationMethod,
    /// Index for nearest neighbor interpolation
    point_index: Option<PointIndex<D>>,
    /// Index for linear interpolation
    elem_index: Option<ObjectIndex<D, M>>,
}

impl<'a, const D: usize, M: Mesh<D> + Clone> Interpolator<'a, D, M> {
    /// Create the interpolator (initialize the indices)
    pub fn new(mesh: &'a M, method: InterpolationMethod) -> Self {
        let (point_index, elem_index) = match method {
            InterpolationMethod::Nearest => (Some(PointIndex::new(mesh.verts())), None),
            InterpolationMethod::Linear(_) => (None, Some(ObjectIndex::new((*mesh).clone()))),
        };
        Self {
            mesh,
            method,
            point_index,
            elem_index,
        }
    }

    /// Interpolate `f` defined at the mesh vertices at locations `verts`
    ///   `f` can be a vector of `m*n_verts` f64 or nalgebra vectors
    pub fn interpolate<
        T: Default + std::ops::Mul<f64, Output = T> + std::ops::Add<T, Output = T> + Copy,
    >(
        &self,
        f: &[T],
        verts: impl ExactSizeIterator<Item = Vertex<D>>,
    ) -> Vec<T> {
        let n = self.mesh.n_verts();
        assert_eq!(f.len() % n, 0);
        let m = f.len() / n;

        match self.method {
            InterpolationMethod::Nearest => {
                let index = self.point_index.as_ref().unwrap();
                verts
                    .flat_map(|v| {
                        let (i_vert, _) = index.nearest_vert(&v);
                        (0..m).map(move |j| f[m * i_vert + j])
                    })
                    .collect()
            }
            InterpolationMethod::Linear(tol) => {
                let tol = tol.max(1e-8);
                let index = self.elem_index.as_ref().unwrap();
                verts
                    .flat_map(|v| {
                        let i_elem = index.nearest_elem(&v);
                        let e = self.mesh.elem(i_elem);
                        let ge = self.mesh.gelem(&e);
                        let mut x = ge.bcoords(&v).into_iter().collect::<Vec<_>>();
                        if !x.iter().all(|c| (-tol..1.0 + tol).contains(c)) {
                            for c in &mut x {
                                *c = c.max(0.0);
                            }
                            let sum = x.iter().sum::<f64>();
                            for c in &mut x {
                                *c /= sum;
                            }
                        }
                        (0..m).map(move |j| {
                            let iter = e.into_iter().zip(x.iter().copied());
                            iter.fold(T::default(), |a, (i, w)| a + f[m * i + j] * w)
                        })
                    })
                    .collect()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        Vert2d, Vert3d,
        mesh::{Mesh, Mesh2d, Mesh3d, Simplex, box_mesh, disk_mesh, rectangle_mesh},
    };
    use nalgebra::{Rotation2, Rotation3};
    use std::f64::consts::FRAC_PI_4;

    use super::{InterpolationMethod, Interpolator};

    #[test]
    fn test_interpolate_2d_outside() {
        let mesh = disk_mesh::<Mesh2d>(2);
        let interp = Interpolator::new(&mesh, InterpolationMethod::Linear(1e-3));
        let f: Vec<f64> = mesh.verts().map(|p| p[0]).collect();

        // a point on the exact circle, outside of the polygonal disk
        let fc = mesh.face(0);
        let c: Vert2d = 0.5 * (mesh.vert(fc.get(0)) + mesh.vert(fc.get(1)));
        let p: Vert2d = 0.5 * c / c.norm();
        assert!(p.norm() - c.norm() > 1e-3);

        // the value is interpolated at a point of the nearest element, close to c
        let res = interp.interpolate(&f, std::iter::once(p));
        assert!(f64::abs(res[0] - c[0]) < 1e-2, "{} vs {}", res[0], c[0]);
    }

    #[test]
    fn test_interpolate_2d() {
        let mesh = rectangle_mesh::<Mesh2d>(1.0, 9, 1.0, 9);
        let interp = Interpolator::new(&mesh, InterpolationMethod::Linear(0.0));

        let fun = |p: Vert2d| 1.0 * p[0] + 2.0 * p[1];
        let f: Vec<f64> = mesh.verts().map(fun).collect();

        let rot = Rotation2::new(FRAC_PI_4);

        let mut other = rectangle_mesh::<Mesh2d>(1.0, 9, 1.0, 9);
        other.verts_mut().for_each(|x| {
            let p = Vert2d::new(0.5, 0.5);
            let tmp = 0.5 * (rot * (*x - p));
            *x = p + tmp;
        });

        let other = other.split().split().split();
        let f_other = interp.interpolate(&f, other.verts());

        for (a, b) in other.verts().map(fun).zip(f_other.iter().copied()) {
            assert!(f64::abs(b - a) < 1e-10);
        }
    }

    #[test]
    fn test_interpolate_2d_nearest() {
        let mesh = rectangle_mesh::<Mesh2d>(1.0, 17, 1.0, 17);
        let interp = Interpolator::new(&mesh, InterpolationMethod::Nearest);

        let fun = |p: Vert2d| 1.0 * p[0] + 2.0 * p[1];
        let f: Vec<f64> = mesh.verts().map(fun).collect();

        let rot = Rotation2::new(FRAC_PI_4);

        let mut other = rectangle_mesh::<Mesh2d>(1.0, 9, 1.0, 9);
        other.verts_mut().for_each(|x| {
            let p = Vert2d::new(0.5, 0.5);
            let tmp = 0.5 * (rot * (*x - p));
            *x = p + tmp;
        });

        let f_other = interp.interpolate(&f, other.verts());

        for (a, b) in other.verts().map(fun).zip(f_other.iter().copied()) {
            assert!(f64::abs(b - a) < f64::midpoint(1.0, 2.0) / 16.0 + 1e-6);
        }
    }

    #[test]
    fn test_interpolate_3d() {
        let mesh = box_mesh::<Mesh3d>(1.0, 9, 1.0, 9, 1.0, 9);
        let interp = Interpolator::new(&mesh, InterpolationMethod::Linear(0.0));

        let fun = |p: Vert3d| 1.0 * p[0] + 2.0 * p[1] + 3.0 * p[2];

        let f: Vec<f64> = mesh.verts().map(fun).collect();

        let rot = Rotation3::from_euler_angles(FRAC_PI_4, FRAC_PI_4, FRAC_PI_4);

        let mut other = box_mesh::<Mesh3d>(1.0, 9, 1.0, 9, 1.0, 9);
        other.verts_mut().for_each(|x| {
            let p = Vert3d::new(0.5, 0.5, 0.5);
            let tmp = 0.5 * (rot * (*x - p));
            *x = p + tmp;
        });

        let f_other = interp.interpolate(&f, other.verts());

        for (a, b) in other.verts().map(fun).zip(f_other.iter().copied()) {
            assert!(f64::abs(b - a) < 1e-10);
        }
    }

    #[test]
    fn test_interpolate_3d_nearest() {
        let mesh = box_mesh::<Mesh3d>(1.0, 9, 1.0, 9, 1.0, 9);
        let interp = Interpolator::new(&mesh, InterpolationMethod::Linear(0.0));

        let fun = |p: Vert3d| 1.0 * p[0] + 2.0 * p[1] + 3.0 * p[2];

        let f: Vec<f64> = mesh.verts().map(fun).collect();

        let rot = Rotation3::from_euler_angles(FRAC_PI_4, FRAC_PI_4, FRAC_PI_4);

        let mut other = box_mesh::<Mesh3d>(1.0, 9, 1.0, 9, 1.0, 9);
        other.verts_mut().for_each(|x| {
            let p = Vert3d::new(0.5, 0.5, 0.5);
            let tmp = 0.5 * (rot * (*x - p));
            *x = p + tmp;
        });

        let f_other = interp.interpolate(&f, other.verts());

        for (a, b) in other.verts().map(fun).zip(f_other.iter().copied()) {
            assert!(f64::abs(b - a) < 0.5 * (1.0 + 2.0 + 3.0) / 8.0 + 1e-6);
        }
    }
}
