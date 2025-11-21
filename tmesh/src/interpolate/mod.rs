//! Interpolation
use std::marker::PhantomData;

use crate::{
    Vertex,
    mesh::{GSimplex, Mesh, Simplex},
    spatialindex::{ObjectIndex, PointIndex},
};

/// Interpolation method
pub enum InterpolationMethod {
    /// Nearest neighbor interpolation
    Nearest,
    /// Linear interpolation
    Linear(f64),
}

/// Interpolator
pub struct Interpolator<'a, const D: usize, C: Simplex, M: Mesh<D, C>> {
    /// Mesh from which the data in interpolated
    mesh: &'a M,
    /// Interpolation method
    method: InterpolationMethod,
    /// Index for nearest neighbor interpolation
    point_index: Option<PointIndex<D>>,
    /// Index for linear interpolation
    elem_index: Option<ObjectIndex<D>>,
    _c: PhantomData<C>,
}

impl<'a, const D: usize, C: Simplex, M: Mesh<D, C>> Interpolator<'a, D, C, M> {
    /// Create the interpolator (initialize the indices)
    pub fn new(mesh: &'a M, method: InterpolationMethod) -> Self {
        let (point_index, elem_index) = match method {
            InterpolationMethod::Nearest => (Some(PointIndex::new(mesh.verts())), None),
            InterpolationMethod::Linear(_) => (None, Some(ObjectIndex::new(mesh))),
        };
        Self {
            mesh,
            method,
            point_index,
            elem_index,
            _c: PhantomData::<C>,
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
                        let x = ge.bcoords(&v);
                        assert!(
                            x.into_iter().all(|c| (-tol..1.0 + tol).contains(&c)),
                            "{x:?}, bcoords = {x:?}"
                        );
                        (0..m).map(move |j| {
                            let iter = e.into_iter().zip(x);
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
        mesh::{Mesh, Mesh2d, Mesh3d, box_mesh, rectangle_mesh},
    };
    use nalgebra::{Rotation2, Rotation3};
    use std::f64::consts::FRAC_PI_4;

    use super::{InterpolationMethod, Interpolator};

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
            assert!(f64::abs(b - a) < 0.5 * (1.0 + 2.0) / 16.0 + 1e-6);
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
