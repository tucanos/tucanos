use crate::{
    geom_elems::GElem,
    mesh::{Point, SimplexMesh},
    topo_elems::Elem,
    Result,
};
use log::debug;
use nalgebra::SMatrix;

impl<const D: usize, E: Elem> SimplexMesh<D, E> {
    /// Compute the gradient of a scalar field defined at the mesh vertices
    /// as the L2 projection of the element-based gradient. The gradient at vertex
    /// $`i`$ is approximated as
    /// ```math
    /// \nabla u_i = \frac{\sum_{K_j \ni i} |K_j| \nabla u_{K_j}}{\sum_{K_j \ni i} |K_j|}
    /// ```
    /// where the sum is considered over all elements that contain vertex $`i`$.
    ///
    pub fn gradient_l2proj(&self, f: &[f64]) -> Result<Vec<f64>> {
        debug!("Compute gradient using L2 projection");

        assert_eq!(f.len(), self.n_verts() as usize);

        let mut res = vec![0.0; D * self.n_verts() as usize];

        for i_elem in 0..self.n_elems() {
            let e = self.elem(i_elem);
            let ge = self.gelem(e);
            let mut grad = Point::<D>::zeros();
            for i_face in 0..E::N_FACES {
                let i_vert = e[i_face as usize] as usize;
                let gf = ge.gface(i_face);
                grad += f[i_vert] * gf.scaled_normal();
            }
            for i_vert in e {
                for i in 0..D {
                    res[D * i_vert as usize + i] += grad[i];
                }
            }
        }

        let vol = self.get_vertex_volumes()?;
        let fac = match D {
            2 => -6.0,
            3 => -12.0,
            _ => unreachable!(),
        };
        for i_vert in 0..self.n_verts() {
            for i in 0..D {
                res[D * i_vert as usize + i] /= fac * vol[i_vert as usize];
            }
        }
        Ok(res)
    }

    /// Compute the hessian of a scalar field defined at the mesh vertices
    /// applying twice the L2 projection gradient.
    /// NB: this actually does not converge the the hessian!
    pub fn hessian_l2proj(&self, gradf: &[f64]) -> Result<Vec<f64>> {
        debug!("Compute hessian using L2 projection");

        assert_eq!(gradf.len(), D * self.n_verts() as usize);

        let indices = if D == 2 {
            vec![0, 3, 1]
        } else if D == 3 {
            vec![0, 4, 8, 1, 5, 2]
        } else {
            unreachable!()
        };
        let n = indices.len();

        let mut res = vec![0.0; n * self.n_verts() as usize];
        for i_elem in 0..self.n_elems() {
            let e = self.elem(i_elem);
            let ge = self.gelem(e);
            let mut hess = SMatrix::<f64, D, D>::zeros();
            for i_face in 0..E::N_FACES {
                let i_vert = e[i_face as usize] as usize;
                let start = D * i_vert;
                let end = start + D;
                let grad = &gradf[start..end];
                let gf = ge.gface(i_face);
                let n = gf.scaled_normal();
                for i in 0..D {
                    for j in 0..D {
                        hess[D * i + j] += grad[i] * n[j];
                    }
                }
            }
            hess = 0.5 * (hess + hess.transpose());

            for i_vert in e {
                for (i, &j) in indices.iter().enumerate() {
                    res[n * i_vert as usize + i] += hess[j];
                }
            }
        }

        let vol = self.get_vertex_volumes()?;
        let fac = match D {
            2 => -6.0,
            3 => -12.0,
            _ => unreachable!(),
        };
        for i_vert in 0..self.n_verts() {
            for i in 0..n {
                res[n * i_vert as usize + i] /= fac * vol[i_vert as usize];
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
        topo_elems::Elem,
        Result,
    };

    #[test]
    fn test_gradient_2d_linear() -> Result<()> {
        let mesh = test_mesh_2d();
        let mut mesh = mesh.split().split().split().split();

        mesh.compute_volumes();

        let f: Vec<_> = mesh.verts().map(|p| p[0] + 2.0 * p[1]).collect();
        let res = mesh.gradient_l2proj(&f)?;
        for i_vert in 0..mesh.n_verts() as usize {
            assert!(f64::abs(res[2 * i_vert] - 1.) < 1e-10);
            assert!(f64::abs(res[2 * i_vert + 1] - 2.) < 1e-10);
        }

        Ok(())
    }

    fn f_2d(p: Point<2>) -> f64 {
        p[0] * p[0] + 2.0 * p[1] * p[1] + 3.0 * p[0] * p[1]
    }

    fn gradf_2d(p: Point<2>) -> [f64; 2] {
        [2.0 * p[0] + 3.0 * p[1], 3.0 * p[0] + 4.0 * p[1]]
    }

    const fn hess_2d(_p: Point<2>) -> [f64; 3] {
        [2.0, 4.0, 3.0]
    }

    fn run_gradient_2d(n: u32) -> Result<f64> {
        let mut mesh = test_mesh_2d();
        for _ in 0..n {
            mesh = mesh.split();
        }

        mesh.compute_volumes();

        let v = mesh.get_vertex_volumes()?;

        let f: Vec<_> = mesh.verts().map(f_2d).collect();
        let res = mesh.gradient_l2proj(&f)?;
        let mut nrm = 0.0;
        for (i_vert, (p, w)) in mesh.verts().zip(v.iter()).enumerate() {
            let grad_ref = gradf_2d(p);
            for i in 0..2 {
                nrm += w * f64::powi(res[2 * i_vert + i] - grad_ref[i], 2);
            }
        }

        Ok(nrm.sqrt())
    }

    #[test]
    fn test_gradient_2d() -> Result<()> {
        let mut prev = f64::MAX;
        for n in 3..7 {
            let nrm = run_gradient_2d(n)?;
            assert!(nrm < 0.5 * prev);
            prev = nrm;
        }
        Ok(())
    }

    #[test]
    fn test_hessian_2d_quadratic() -> Result<()> {
        let mesh = test_mesh_2d();
        let mut mesh = mesh.split().split().split().split();

        mesh.compute_volumes();

        let f: Vec<_> = mesh.verts().map(f_2d).collect();
        let grad = mesh.gradient_l2proj(&f)?;
        let hess = mesh.hessian_l2proj(&grad)?;

        // flag the internal vertices, that don't belong to a cell that touches the boundary
        let (_, bdy_ids) = mesh.boundary();
        let mut flg = vec![0; mesh.n_verts() as usize];
        bdy_ids.iter().for_each(|&i| flg[i as usize] = 1);
        mesh.elems().for_each(|e| {
            if e.iter().any(|&i| flg[i as usize] == 1) {
                for &i in e.iter() {
                    if flg[i as usize] == 0 {
                        flg[i as usize] = 2;
                    }
                }
            }
        });

        for (i_vert, p) in mesh.verts().enumerate() {
            if flg[i_vert] == 0 {
                let hess_ref = hess_2d(p);
                for i in 0..3 {
                    assert!(f64::abs(hess[3 * i_vert + i] - hess_ref[i]) < 1e-10);
                }
            }
        }

        Ok(())
    }

    fn f_3d(p: Point<3>) -> f64 {
        p[0] * p[0]
            + 2.0 * p[1] * p[1]
            + 3.0 * p[2] * p[2]
            + 4.0 * p[0] * p[1]
            + 5.0 * p[1] * p[2]
            + 6.0 * p[0] * p[2]
    }

    fn gradf_3d(p: Point<3>) -> [f64; 3] {
        [
            2.0 * p[0] + 4.0 * p[1] + 6.0 * p[2],
            4.0 * p[0] + 4.0 * p[1] + 5.0 * p[2],
            6.0 * p[0] + 5.0 * p[1] + 6.0 * p[2],
        ]
    }

    // fn hess_3d(_p: Point<3>) -> [f64; 6] {
    //     [2.0, 4.0, 6.0, 4.0, 5.0, 6.0]
    // }

    #[test]
    fn test_gradient_3d_linear() -> Result<()> {
        let mesh = test_mesh_3d();
        let mut mesh = mesh.split().split().split().split();

        mesh.compute_volumes();

        let f: Vec<_> = mesh
            .verts()
            .map(|p| p[0] + 2.0 * p[1] + 3.0 * p[2])
            .collect();
        let res = mesh.gradient_l2proj(&f)?;
        for i_vert in 0..mesh.n_verts() as usize {
            assert!(f64::abs(res[3 * i_vert] - 1.) < 1e-10);
            assert!(f64::abs(res[3 * i_vert + 1] - 2.) < 1e-10);
            assert!(f64::abs(res[3 * i_vert + 2] - 3.) < 1e-10);
        }

        Ok(())
    }

    fn run_gradient_3d(n: u32) -> Result<f64> {
        let mut mesh = test_mesh_3d();
        for _ in 0..n {
            mesh = mesh.split();
        }

        mesh.compute_vertex_to_vertices();
        mesh.compute_volumes();

        let v = mesh.get_vertex_volumes()?;

        let f: Vec<_> = mesh.verts().map(f_3d).collect();
        let res = mesh.gradient_l2proj(&f)?;
        let mut nrm = 0.0;
        for (i_vert, (p, w)) in mesh.verts().zip(v.iter()).enumerate() {
            let grad_ref = gradf_3d(p);
            for i in 0..3 {
                nrm += w * f64::powi(res[3 * i_vert + i] - grad_ref[i], 2);
            }
        }
        Ok(nrm.sqrt())
    }

    #[test]
    fn test_gradient_3d() -> Result<()> {
        let mut prev = f64::MAX;
        for n in 2..5 {
            let nrm = run_gradient_3d(n)?;
            assert!(nrm < 0.5 * prev);
            prev = nrm;
        }
        Ok(())
    }

    #[test]
    fn test_hessian_3d_quadratic() -> Result<()> {
        let mesh = test_mesh_3d();
        let mut mesh = mesh.split().split().split().split();

        mesh.compute_volumes();

        let f: Vec<_> = mesh.verts().map(f_3d).collect();

        let grad = mesh.gradient_l2proj(&f)?;
        let _hess = mesh.hessian_l2proj(&grad)?;

        // No test!

        // // flag the internal vertices, that don't belong to a cell that touches the boundary
        // let (_, bdy_ids) = mesh.boundary();
        // let mut flg = vec![0; mesh.n_verts() as usize];
        // bdy_ids.iter().for_each(|&i| flg[i as usize] = 1);
        // mesh.elems().for_each(|e| {
        //     if e.iter().any(|&i| flg[i as usize] == 1) {
        //         for &i in e.iter() {
        //             if flg[i as usize] == 0 {
        //                 flg[i as usize] = 2;
        //             }
        //         }
        //     }
        // });

        // for (i_vert, p) in mesh.verts().enumerate() {
        //     if flg[i_vert] == 0 {
        //         let hess_ref = hess_3d(p);
        //         for i in 0..6 {
        //             println!("{} {}", hess[6 * i_vert + i], hess_ref[i]);
        //             assert!(hess[6 * i_vert + i] < 3.0 * hess_ref[i]);
        //             // assert!(f64::abs(hess[6 * i_vert + i] - hess_ref[i]) < 1e-10);
        //         }
        //     }
        // }

        Ok(())
    }
}
