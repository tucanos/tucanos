use crate::{
    geom_elems::AsSliceF64, geom_elems::GElem, mesh::SimplexMesh, topo_elems::Elem, Error, Mesh,
    Result,
};

impl<const D: usize, E: Elem> SimplexMesh<D, E> {
    pub fn interpolate(&self, other: &Self, f: &[f64]) -> Result<Vec<f64>> {
        if self.tree.is_none() {
            return Err(Error::from(
                "compute_octree() not called before interpolate is called",
            ));
        }

        let n_verts = self.n_verts() as usize;
        let n_verts_other = other.n_verts() as usize;
        assert_eq!(f.len() % n_verts, 0);

        let n_comp = f.len() / n_verts;

        let mut res = Vec::with_capacity(n_verts_other * n_comp);
        let tree = self.tree.as_ref().unwrap();
        for vert in other.verts() {
            let i_elem = tree.nearest(&vert);
            let e = self.elem(i_elem);
            let ge = self.gelem(e);
            let x = ge.bcoords(&vert);
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
        test_meshes::{test_mesh_2d, test_mesh_3d},
        Result,
    };

    #[test]
    fn test_interpolate_2d() -> Result<()> {
        let mut mesh = test_mesh_2d().split().split().split();
        mesh.compute_octree();

        let f: Vec<f64> = mesh.verts().map(|p| p[0]).collect();

        let other = test_mesh_2d().split().split().split().split();
        let f_other = mesh.interpolate(&other, &f)?;

        for (a, b) in other.verts().map(|p| p[0]).zip(f_other.iter().copied()) {
            assert!(f64::abs(b - a) < 1e-10);
        }

        let other = test_mesh_2d().split();
        let f_other = mesh.interpolate(&other, &f)?;

        for (a, b) in other.verts().map(|p| p[0]).zip(f_other.iter().copied()) {
            assert!(f64::abs(b - a) < 1e-10);
        }

        Ok(())
    }

    #[test]
    fn test_interpolate_3d() -> Result<()> {
        let mut mesh = test_mesh_3d().split().split().split();
        mesh.compute_octree();

        let f: Vec<f64> = mesh.verts().map(|p| p[0]).collect();

        let other = test_mesh_3d().split().split().split().split();
        let f_other = mesh.interpolate(&other, &f)?;

        for (a, b) in other.verts().map(|p| p[0]).zip(f_other.iter().copied()) {
            assert!(f64::abs(b - a) < 1e-10);
        }

        let other = test_mesh_3d().split();
        let f_other = mesh.interpolate(&other, &f)?;

        for (a, b) in other.verts().map(|p| p[0]).zip(f_other.iter().copied()) {
            assert!(f64::abs(b - a) < 1e-10);
        }

        Ok(())
    }
}
