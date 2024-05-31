use crate::{
    geom_elems::{AsSliceF64, GElem},
    mesh::SimplexMesh,
    spatialindex::{DefaultObjectIndex, DefaultPointIndex, ObjectIndex, PointIndex},
    topo_elems::Elem,
    Result,
};

impl<const D: usize, E: Elem> SimplexMesh<D, E> {
    pub fn interpolate_nearest(
        &self,
        tree: &DefaultPointIndex<D>,
        other: &Self,
        f: &[f64],
    ) -> Result<Vec<f64>> {
        let n_verts = self.n_verts() as usize;
        let n_verts_other = other.n_verts() as usize;

        assert_eq!(f.len() % n_verts, 0);
        let n_comp = f.len() / n_verts;

        let mut res = Vec::with_capacity(n_verts_other * n_comp);
        for vert in other.verts() {
            let (i_vert, _) = tree.nearest_vert(&vert);
            for j in 0..n_comp {
                res.push(f[n_comp * i_vert as usize + j]);
            }
        }

        Ok(res)
    }

    pub fn interpolate_linear(
        &self,
        tree: &DefaultObjectIndex<D>,
        other: &Self,
        f: &[f64],
        tol: Option<f64>,
    ) -> Result<Vec<f64>> {
        let n_verts = self.n_verts() as usize;
        let n_verts_other = other.n_verts() as usize;
        let tol = tol.unwrap_or(1e-12);

        assert_eq!(f.len() % n_verts, 0);
        let n_comp = f.len() / n_verts;

        let mut res = Vec::with_capacity(n_verts_other * n_comp);
        for vert in other.verts() {
            let i_elem = tree.nearest_elem(&vert);
            let e = self.elem(i_elem);
            let ge = self.gelem(e);
            let x = ge.bcoords(&vert);
            assert!(
                x.as_slice_f64()
                    .iter()
                    .all(|c| (-tol..1.0 + tol).contains(c)),
                "{x:?}, bcoords = {x:?}"
            );
            for j in 0..n_comp {
                let iter = e.iter().copied().zip(x.as_slice_f64().iter().copied());
                res.push(iter.fold(0.0, |a, (i, w)| a + f[n_comp * i as usize + j] * w));
            }
        }
        Ok(res)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        mesh::Point,
        test_meshes::{test_mesh_2d, test_mesh_3d},
        Result,
    };
    use nalgebra::{Rotation2, Rotation3};
    use std::f64::consts::FRAC_PI_4;

    #[test]
    fn test_interpolate_2d() -> Result<()> {
        let mesh = test_mesh_2d().split().split().split();
        let tree = mesh.compute_elem_tree();

        let fun = |p: Point<2>| 1.0 * p[0] + 2.0 * p[1];

        let f: Vec<f64> = mesh.verts().map(fun).collect();

        let rot = Rotation2::new(FRAC_PI_4);

        let mut other = test_mesh_2d();
        other.mut_verts().for_each(|x| {
            let p = Point::<2>::new(0.5, 0.5);
            let tmp = 0.5 * (rot * (*x - p));
            *x = p + tmp;
        });

        let other = other.split().split().split();
        let f_other = mesh.interpolate_linear(&tree, &other, &f, None)?;

        for (a, b) in other.verts().map(fun).zip(f_other.iter().copied()) {
            assert!(f64::abs(b - a) < 1e-10);
        }

        Ok(())
    }

    #[test]
    fn test_interpolate_2d_nearest() -> Result<()> {
        let mesh = test_mesh_2d().split().split().split().split();
        let tree = mesh.compute_vert_tree();

        let fun = |p: Point<2>| 1.0 * p[0] + 2.0 * p[1];

        let f: Vec<f64> = mesh.verts().map(fun).collect();

        let rot = Rotation2::new(FRAC_PI_4);

        let mut other = test_mesh_2d();
        other.mut_verts().for_each(|x| {
            let p = Point::<2>::new(0.5, 0.5);
            let tmp = 0.5 * (rot * (*x - p));
            *x = p + tmp;
        });

        let other = other.split().split().split();
        let f_other = mesh.interpolate_nearest(&tree, &other, &f)?;

        for (a, b) in other.verts().map(fun).zip(f_other.iter().copied()) {
            assert!(f64::abs(b - a) < 0.5 * (1.0 + 2.0) / 16.0 + 1e-6);
        }

        Ok(())
    }

    #[test]
    fn test_interpolate_3d() -> Result<()> {
        let mesh = test_mesh_3d().split().split().split();
        let tree = mesh.compute_elem_tree();

        let fun = |p: Point<3>| 1.0 * p[0] + 2.0 * p[1] + 3.0 * p[2];

        let f: Vec<f64> = mesh.verts().map(fun).collect();

        let rot = Rotation3::from_euler_angles(FRAC_PI_4, FRAC_PI_4, FRAC_PI_4);

        let mut other = test_mesh_3d();
        other.mut_verts().for_each(|x| {
            let p = Point::<3>::new(0.5, 0.5, 0.5);
            let tmp = 0.5 * (rot * (*x - p));
            *x = p + tmp;
        });
        let other = other.split().split().split();

        let f_other = mesh.interpolate_linear(&tree, &other, &f, None)?;

        for (a, b) in other.verts().map(fun).zip(f_other.iter().copied()) {
            assert!(f64::abs(b - a) < 1e-10);
        }

        Ok(())
    }

    #[test]
    fn test_interpolate_3d_nearest() -> Result<()> {
        let mesh = test_mesh_3d().split().split().split();
        let tree = mesh.compute_vert_tree();

        let fun = |p: Point<3>| 1.0 * p[0] + 2.0 * p[1] + 3.0 * p[2];

        let f: Vec<f64> = mesh.verts().map(fun).collect();

        let rot = Rotation3::from_euler_angles(FRAC_PI_4, FRAC_PI_4, FRAC_PI_4);

        let mut other = test_mesh_3d();
        other.mut_verts().for_each(|x| {
            let p = Point::<3>::new(0.5, 0.5, 0.5);
            let tmp = 0.5 * (rot * (*x - p));
            *x = p + tmp;
        });
        let other = other.split().split().split();

        let f_other = mesh.interpolate_nearest(&tree, &other, &f)?;

        for (a, b) in other.verts().map(fun).zip(f_other.iter().copied()) {
            assert!(f64::abs(b - a) < 0.5 * (1.0 + 2.0 + 3.0) / 8.0 + 1e-6);
        }

        Ok(())
    }
}
