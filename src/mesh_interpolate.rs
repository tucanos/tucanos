use crate::{
    geom_elems::{AsSliceF64, GElem},
    mesh::SimplexMesh,
    spatialindex::ObjectIndex as _,
    topo_elems::Elem,
    Result,
};

impl<const D: usize, E: Elem> SimplexMesh<D, E> {
    pub fn interpolate(&self, other: &Self, f: &[f64]) -> Result<Vec<f64>> {
        self.interpolate_check(other, f, false)
    }
    pub fn interpolate_check(&self, other: &Self, f: &[f64], check: bool) -> Result<Vec<f64>> {
        let n_verts = self.n_verts() as usize;
        let n_verts_other = other.n_verts() as usize;
        assert_eq!(f.len() % n_verts, 0);

        let n_comp = f.len() / n_verts;

        let mut res = Vec::with_capacity(n_verts_other * n_comp);
        let tree = self.get_octree()?;
        for vert in other.verts() {
            let i_elem = tree.nearest(&vert);
            let e = self.elem(i_elem);
            let ge = self.gelem(e);
            let x = ge.bcoords(&vert);
            if check {
                x.as_slice_f64()
                    .iter()
                    .for_each(|c| assert!((-5e-16..=1.).contains(c), "{x:?}"));
            }
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
    use std::time::Instant;

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
        let f_other = mesh.interpolate_check(&other, &f, true)?;

        for (a, b) in other.verts().map(|p| p[0]).zip(f_other.iter().copied()) {
            assert!(f64::abs(b - a) < 1e-10);
        }

        let other = test_mesh_2d().split();
        let f_other = mesh.interpolate_check(&other, &f, true)?;

        for (a, b) in other.verts().map(|p| p[0]).zip(f_other.iter().copied()) {
            assert!(f64::abs(b - a) < 1e-10);
        }

        Ok(())
    }

    #[test]
    fn test_interpolate_3d() -> Result<()> {
        let add_split = 0;
        let mut mesh = test_mesh_3d();
        for _ in 0..(3 + add_split) {
            mesh = mesh.split();
        }
        let start = Instant::now();
        mesh.compute_octree();
        println!(
            "Time to compute spatial index of {} cells: {:?}",
            mesh.n_elems(),
            start.elapsed()
        );

        let f: Vec<f64> = mesh.verts().map(|p| p[0]).collect();

        let mut other = test_mesh_3d();
        for _ in 0..(4 + add_split) {
            other = other.split();
        }
        let start = Instant::now();
        let f_other = mesh.interpolate_check(&other, &f, true)?;
        println!(
            "Time to interpolate a {} cells mesh: {:?}",
            other.n_elems(),
            start.elapsed()
        );

        for (a, b) in other.verts().map(|p| p[0]).zip(f_other.iter().copied()) {
            assert!(f64::abs(b - a) < 1e-10);
        }

        let mut other = test_mesh_3d();
        (0..=add_split).for_each(|_| {
            other = other.split();
        });
        let start = Instant::now();
        let f_other = mesh.interpolate_check(&other, &f, true)?;
        println!(
            "Time to interpolate a {} cells mesh: {:?}",
            other.n_elems(),
            start.elapsed()
        );

        for (a, b) in other.verts().map(|p| p[0]).zip(f_other.iter().copied()) {
            assert!(f64::abs(b - a) < 1e-10);
        }

        Ok(())
    }
}
