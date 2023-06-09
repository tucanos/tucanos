use crate::{
    linalg::lapack_qr_least_squares, mesh::SimplexMesh, topo_elems::Elem, Error, FieldType, Idx,
    Mesh, Result,
};

use log::info;

impl<const D: usize, E: Elem> SimplexMesh<D, E> {
    /// Compute a linear or quadratic approximation of a function defined at the mesh vertices
    ///
    /// A weighted least squares is used: the function $`f`$ is approximated as around $`x_i`$
    /// ```math
    /// f(x) \approx \alpha + a_j \delta_j + a_{j,k} \delta_j \delta_k
    /// ```
    /// with $`\delta = x - x_i`$. For a 1st order approximation, the $`a_{j, k}`$ are
    /// set to 0. The coefficients are then chosen to minimize
    /// ```math
    /// W_0 (\alpha)^2 + \sum_{j \in N(i)} W_i ((f_j - f_i) - (\alpha + a_k \delta_k^{(j)} + a_{k,l} \delta_k^{(j)} \delta_l^{(j)}))^2
    /// ```
    /// with $`\delta^{(j)} = x_j - x_i`$ for a neighborhood $`N(i) = \{j_0, \cdots, j_N\}`$ of vertex
    /// $`i`$ and a weighting $`W_i=\frac{1}{\left \| \delta_i \right \|^P}`$.
    /// P is `weight_exp` and typically $`P \in \left \{ 0,1,2 \right \}`$.
    /// $`W_0`$ is set here to $`\sqrt{2} \max(W_j)`$.
    ///
    /// Numerically the following least squares problem is solved using a QR factorization
    /// ```math
    /// min \left \|
    /// \begin{bmatrix}
    /// W_0 & 0 & \cdots  & 0 \\
    /// W_{j_0} & W_{j_0}.\delta_0^{(j_0)} & \cdots & W_{j_0}.\delta_d^{(j_0)}\delta_d^{(j_0)}\\
    /// \vdots \\
    /// W_{j_N} & W_{j_N}.\delta_0^{(j_N)} & \cdots & W_{j_N}.\delta_d^{(j_N)}.\delta_d^{(j_N)}
    /// \end{bmatrix}.
    /// \begin{bmatrix} \alpha \\ a_0 \\ \vdots \\ a_{d,d}\end{bmatrix}-
    /// \begin{bmatrix} 0 \\ \widetilde{f}_{j_0} \\ \vdots \\ \widetilde{f}_{j_N} \end{bmatrix} \right \|^2
    /// ```
    ///
    /// If the number of neighbors is not sufficient for this problem to be solved, or if the problem is
    /// too ill-conditioned, None is returned
    fn least_squares<const N: usize>(
        &self,
        i_vert: Idx,
        weight_exp: i32,
        f: &[f64],
    ) -> Option<[f64; N]> {
        debug_assert!(D != 2 || N == 3 || N == 6);
        debug_assert!(D != 3 || N == 4 || N == 10);
        let others = self.vertex_to_vertices.as_ref().unwrap().row(i_vert);
        let n_others = others.len();
        if n_others < N {
            return None;
        }
        let nrows = 1 + n_others;
        let p0 = self.vert(i_vert);
        let f0 = f[i_vert as usize];
        let mut max_weight = 0.0;
        // One allocation to rule them all. An extra N for b and 2 extra N for work.
        let mut buf = vec![0.; nrows * (N + 3)];
        let (a, b) = buf.split_at_mut(nrows * N);
        let (b, work) = b.split_at_mut(nrows);
        for irow in 1..nrows {
            let dp = self.vert(others[irow - 1]) - p0;
            let row = &mut a[irow..];
            row[0] = 1.0;
            row[nrows] = dp[0];
            row[2 * nrows] = dp[1];
            if D == 2 && N == 6 {
                row[3 * nrows] = 0.5 * dp[0] * dp[0];
                row[4 * nrows] = 0.5 * dp[1] * dp[1];
                row[5 * nrows] = dp[0] * dp[1];
            } else if D == 3 {
                row[3 * nrows] = dp[2];
                if N == 10 {
                    row[4 * nrows] = 0.5 * dp[0] * dp[0];
                    row[5 * nrows] = 0.5 * dp[1] * dp[1];
                    row[6 * nrows] = 0.5 * dp[2] * dp[2];
                    row[7 * nrows] = dp[0] * dp[1];
                    row[8 * nrows] = dp[1] * dp[2];
                    row[9 * nrows] = dp[0] * dp[2];
                }
            }
            let weight = if weight_exp > 0 {
                1. / f64::powi(dp.norm(), weight_exp)
            } else {
                1.
            };
            max_weight = f64::max(weight, max_weight);
            for i in 0..N {
                row[i * nrows] *= weight;
            }

            b[irow] = weight * (f[others[irow - 1] as usize] - f0);
        }
        a[0] = max_weight * (2_f64).sqrt();
        lapack_qr_least_squares(a, b, work)
    }

    /// Smooth a vertex field using a 1st order weighted least square approximation
    pub fn smooth(&self, f: &[f64], weight_exp: i32) -> Result<Vec<f64>> {
        info!(
            "Compute smoothing using 1st order LS (weight = {})",
            weight_exp
        );
        if self.vertex_to_vertices.is_none() {
            return Err(Error::from("vertex to vertex connection not available"));
        }

        let m = self.n_comps(FieldType::Scalar);
        let mut res = Vec::with_capacity((m * self.n_verts()) as usize);

        for i_vert in 0..self.n_verts() {
            if D == 2 {
                let sol = self.least_squares::<3>(i_vert, weight_exp, f);
                if let Some(sol) = sol {
                    res.push(f[i_vert as usize] + sol[0]);
                } else {
                    // If the least squared approximation could not be computed, use the original value
                    res.push(f[i_vert as usize]);
                }
            } else if D == 3 {
                let sol = self.least_squares::<4>(i_vert, weight_exp, f);
                if let Some(sol) = sol {
                    res.push(f[i_vert as usize] + sol[0]);
                } else {
                    // If the least squared approximation could not be computed, use the original value
                    res.push(f[i_vert as usize]);
                }
            }
        }

        if res.iter().copied().any(f64::is_nan) {
            return Err(Error::from("NaN in smoothing computation"));
        }
        Ok(res)
    }

    /// For vertices for which a least square approximation could not be computed, use the average over valid neighbors
    /// res: The result from the least square approximation
    /// failed: Flag that indicates if not valid approximation has been computed
    /// m: The number of components (1 for scalar, D for gradients, D*(D-1)/2 for hessians)
    /// `max_iter`: The max number of iteration through the mesh vertices
    fn fix_not_computed(
        &self,
        res: &mut [f64],
        failed: &mut [bool],
        m: usize,
        max_iter: usize,
    ) -> bool {
        let v2v = self.vertex_to_vertices.as_ref().unwrap();
        for _ in 0..max_iter {
            for i_vert in 0..self.n_verts() as usize {
                if failed[i_vert] {
                    let mut n = 0;
                    for i_other in v2v.row(i_vert as Idx).iter().copied() {
                        if !failed[i_other as usize] {
                            for i in 0..m {
                                res[m * i_vert + i] += res[m * i_other as usize + i];
                            }
                            n += 1;
                        }
                    }
                    if n > 0 {
                        let fac = 1. / f64::from(n);
                        for i in 0..m {
                            res[m * i_vert + i] *= fac;
                        }
                        failed[i_vert] = false;
                    }
                }
            }
            if failed.iter().copied().all(|x| !x) {
                return true;
            }
        }
        false
    }

    /// Compute the gradient of a vertex field using a 1st order weighted least square approximation
    pub fn gradient(&self, f: &[f64], weight_exp: i32) -> Result<Vec<f64>> {
        info!(
            "Compute gradient using 1st order LS (weight = {})",
            weight_exp
        );
        if self.vertex_to_vertices.is_none() {
            return Err(Error::from("vertex to vertex connection not available"));
        }

        let m = self.n_comps(FieldType::Vector) as usize;
        let mut res = Vec::with_capacity(m * self.n_verts() as usize);

        let mut failed = vec![false; self.n_verts() as usize];

        for i_vert in 0..self.n_verts() {
            if D == 2 {
                let sol = self.least_squares::<3>(i_vert, weight_exp, f);
                if let Some(sol) = sol {
                    res.extend(sol.iter().skip(1).take(D));
                } else {
                    failed[i_vert as usize] = true;
                    res.resize(res.len() + m, 0.0);
                }
            } else if D == 3 {
                let sol = self.least_squares::<4>(i_vert, weight_exp, f);
                if let Some(sol) = sol {
                    res.extend(sol.iter().skip(1).take(D));
                } else {
                    failed[i_vert as usize] = true;
                    res.resize(res.len() + m, 0.0);
                }
            }
        }

        // For vertices where no valid approximation could be computed, average over the valid neighbors
        if self.fix_not_computed(&mut res, &mut failed, m, 3) {
            if res.iter().copied().any(f64::is_nan) {
                return Err(Error::from("NaN in gradient computation"));
            }
            Ok(res)
        } else {
            Err(Error::from("Cannot compute the value everywhere"))
        }
    }

    /// Compute the hessian of a vertex field using a 2nd order weighted least square approximation
    pub fn hessian(&self, f: &[f64], weight_exp: i32) -> Result<Vec<f64>> {
        info!(
            "Compute hessian using 2nd order LS (weight = {})",
            weight_exp
        );
        if self.vertex_to_vertices.is_none() {
            return Err(Error::from("vertex to vertex connection not available"));
        }

        let m = self.n_comps(FieldType::SymTensor) as usize;
        let mut res = Vec::with_capacity(m * self.n_verts() as usize);

        let mut failed = vec![false; self.n_verts() as usize];

        for i_vert in 0..self.n_verts() {
            if D == 2 {
                let sol = self.least_squares::<6>(i_vert, weight_exp, f);
                if let Some(sol) = sol {
                    res.extend(sol.iter().skip(D + 1));
                } else {
                    failed[i_vert as usize] = true;
                    res.resize(res.len() + m, 0.0);
                }
            } else if D == 3 {
                let sol = self.least_squares::<10>(i_vert, weight_exp, f);
                if let Some(sol) = sol {
                    res.extend(sol.iter().skip(D + 1));
                } else {
                    failed[i_vert as usize] = true;
                    res.resize(res.len() + m, 0.0);
                }
            }
        }

        // For vertices where no valid approximation could be computed, average over the valid neighbors
        if self.fix_not_computed(&mut res, &mut failed, m, 3) {
            if res.iter().copied().any(f64::is_nan) {
                return Err(Error::from("NaN in hessian computation"));
            }
            Ok(res)
        } else {
            Err(Error::from("Cannot compute the value everywhere"))
        }
    }
}

#[cfg(test)]
mod tests {

    use crate::{
        test_meshes::{test_mesh_2d, test_mesh_3d},
        Mesh, Result,
    };

    #[test]
    fn test_smooth_2d() -> Result<()> {
        let mesh = test_mesh_2d();
        let mut mesh = mesh.split().split().split();

        mesh.compute_vertex_to_vertices();

        let f: Vec<_> = mesh.verts().map(|p| p[0] + 2.0 * p[1]).collect();
        let res = mesh.smooth(&f, 2)?;
        for i_vert in 0..mesh.n_verts() as usize {
            assert!(f64::abs(res[i_vert] - f[i_vert]) < 1e-10);
        }

        let f: Vec<_> = mesh.verts().map(|p| p[0] * p[1]).collect();
        let res = mesh.smooth(&f, 2)?;
        for i_vert in 0..mesh.n_verts() as usize {
            assert!(f64::abs(res[i_vert] - f[i_vert]) < 1e-2);
        }

        Ok(())
    }

    #[test]
    fn test_smooth_3d() -> Result<()> {
        let mesh = test_mesh_3d();
        let mut mesh = mesh.split().split().split();

        mesh.compute_vertex_to_vertices();

        let f: Vec<_> = mesh
            .verts()
            .map(|p| p[0] + 2.0 * p[1] + 3.0 * p[2])
            .collect();
        let res = mesh.smooth(&f, 2)?;
        for i_vert in 0..mesh.n_verts() as usize {
            assert!(f64::abs(res[i_vert] - f[i_vert]) < 1e-10);
        }

        let f: Vec<_> = mesh.verts().map(|p| p[0] * p[1] * p[2]).collect();
        let res = mesh.smooth(&f, 2)?;
        for i_vert in 0..mesh.n_verts() as usize {
            assert!(f64::abs(res[i_vert] - f[i_vert]) < 2e-2);
        }

        Ok(())
    }

    #[test]
    fn test_gradient_2d_linear() -> Result<()> {
        let mesh = test_mesh_2d();
        let mut mesh = mesh.split().split().split().split();

        mesh.compute_vertex_to_vertices();

        let f: Vec<_> = mesh.verts().map(|p| p[0] + 2.0 * p[1]).collect();
        let res = mesh.gradient(&f, 2)?;
        for i_vert in 0..mesh.n_verts() as usize {
            assert!(f64::abs(res[2 * i_vert] - 1.) < 1e-10);
            assert!(f64::abs(res[2 * i_vert + 1] - 2.) < 1e-10);
        }

        Ok(())
    }

    fn run_gradient_2d(n: u32) -> Result<f64> {
        let mut mesh = test_mesh_2d();
        for _ in 0..n {
            mesh = mesh.split();
        }

        mesh.compute_vertex_to_vertices();
        mesh.compute_volumes();

        let v = mesh.vert_vol.as_ref().unwrap();

        let f: Vec<_> = mesh.verts().map(|p| p[0] * p[1]).collect();
        let res = mesh.gradient(&f, 2)?;
        let mut nrm = 0.0;
        for (i_vert, (p, w)) in mesh.verts().zip(v.iter()).enumerate() {
            nrm += w
                * (f64::powi(res[2 * i_vert] - p[1], 2) + f64::powi(res[2 * i_vert + 1] - p[0], 2));
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
    fn test_gradient_3d_linear() -> Result<()> {
        let mesh = test_mesh_3d();
        let mut mesh = mesh.split().split().split().split();

        mesh.compute_vertex_to_vertices();

        let f: Vec<_> = mesh
            .verts()
            .map(|p| p[0] + 2.0 * p[1] + 3.0 * p[2])
            .collect();
        let res = mesh.gradient(&f, 2)?;
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

        let v = mesh.vert_vol.as_ref().unwrap();

        let f: Vec<_> = mesh.verts().map(|p| p[0] * p[1] * p[2]).collect();
        let res = mesh.gradient(&f, 2)?;
        let mut nrm = 0.0;
        for (i_vert, (p, w)) in mesh.verts().zip(v.iter()).enumerate() {
            nrm += w
                * (f64::powi(res[3 * i_vert] - p[1] * p[2], 2)
                    + f64::powi(res[3 * i_vert + 1] - p[0] * p[2], 2)
                    + f64::powi(res[3 * i_vert + 2] - p[0] * p[1], 2));
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
    fn test_hessian_2d_quadratic() -> Result<()> {
        let mesh = test_mesh_2d();
        let mut mesh = mesh.split().split();

        mesh.compute_vertex_to_vertices();

        let f: Vec<_> = mesh
            .verts()
            .map(|p| p[0] * p[0] + 2.0 * p[1] * p[1] + 3.0 * p[0] * p[1])
            .collect();
        let res = mesh.hessian(&f, 2)?;
        for i_vert in 0..mesh.n_verts() as usize {
            assert!(f64::abs(res[3 * i_vert] - 2.) < 1e-10);
            assert!(f64::abs(res[3 * i_vert + 1] - 4.) < 1e-10);
            assert!(f64::abs(res[3 * i_vert + 2] - 3.) < 1e-10);
        }

        Ok(())
    }

    #[test]
    fn test_hessian_3d_quadratic() -> Result<()> {
        let mesh = test_mesh_3d();
        let mut mesh = mesh.split().split().split();

        mesh.compute_vertex_to_vertices();

        let f: Vec<_> = mesh
            .verts()
            .map(|p| {
                p[0] * p[0]
                    + 2.0 * p[1] * p[1]
                    + 3.0 * p[2] * p[2]
                    + 4.0 * p[0] * p[1]
                    + 5.0 * p[1] * p[2]
                    + 6.0 * p[0] * p[2]
            })
            .collect();
        let res = mesh.hessian(&f, 2)?;
        for i_vert in 0..mesh.n_verts() as usize {
            assert!(f64::abs(res[6 * i_vert] - 2.) < 1e-10);
            assert!(f64::abs(res[6 * i_vert + 1] - 4.) < 1e-10);
            assert!(f64::abs(res[6 * i_vert + 2] - 6.) < 1e-10);
            assert!(f64::abs(res[6 * i_vert + 3] - 4.) < 1e-10);
            assert!(f64::abs(res[6 * i_vert + 4] - 5.) < 1e-10);
            assert!(f64::abs(res[6 * i_vert + 5] - 6.) < 1e-10);
        }

        Ok(())
    }

    fn run_hessian_2d(n: u32) -> Result<f64> {
        let mut mesh = test_mesh_2d();
        for _ in 0..n {
            mesh = mesh.split();
        }

        mesh.compute_vertex_to_vertices();
        mesh.compute_volumes();

        let v = mesh.vert_vol.as_ref().unwrap();

        let f: Vec<_> = mesh
            .verts()
            .map(|p| p[0] * p[0] * p[1] + 2.0 * p[0] * p[1] * p[1])
            .collect();
        let res = mesh.hessian(&f, 2)?;
        let mut nrm = 0.0;
        for (i_vert, (p, w)) in mesh.verts().zip(v.iter()).enumerate() {
            nrm += w
                * (f64::powi(res[3 * i_vert] - 2.0 * p[1], 2)
                    + f64::powi(res[3 * i_vert + 1] - 4.0 * p[0], 2)
                    + f64::powi(res[3 * i_vert + 2] - 2.0 * p[0] - 4.0 * p[1], 2));
        }

        Ok(nrm.sqrt())
    }

    #[test]
    fn test_hessian_2d() -> Result<()> {
        let mut prev = f64::MAX;
        for n in 3..7 {
            let nrm = run_hessian_2d(n)?;
            assert!(nrm < 0.5 * prev);
            prev = nrm;
        }
        Ok(())
    }

    fn run_hessian_3d(num_split: u32) -> Result<f64> {
        let mut mesh = test_mesh_3d();
        for _ in 0..num_split {
            mesh = mesh.split();
        }

        mesh.compute_vertex_to_vertices();
        mesh.compute_volumes();

        let vols = mesh.vert_vol.as_ref().unwrap();

        let test_f: Vec<_> = mesh
            .verts()
            .map(|p| {
                let x = p[0];
                let y = p[1];
                let z = p[2];
                x * x * y * z + 2.0 * x * y * y * z + 3.0 * x * y * z * z
            })
            .collect();
        let res = mesh.hessian(&test_f, 2)?;
        let mut nrm = 0.0;
        for (i_vert, (p, w)) in mesh.verts().zip(vols.iter()).enumerate() {
            let x = p[0];
            let y = p[1];
            let z = p[2];
            nrm += w
                * (f64::powi(res[6 * i_vert] - 2.0 * y * z, 2)
                    + f64::powi(res[6 * i_vert + 1] - 4.0 * x * z, 2)
                    + f64::powi(res[6 * i_vert + 2] - 6.0 * x * y, 2)
                    + f64::powi(res[6 * i_vert + 3] - z * (2.0 * x + 4.0 * y + 3.0 * z), 2)
                    + f64::powi(res[6 * i_vert + 4] - x * (x + 4.0 * y + 6.0 * z), 2)
                    + f64::powi(res[6 * i_vert + 5] - 2.0 * y * (x + y + 3.0 * z), 2));
        }

        Ok(nrm.sqrt())
    }

    #[test]
    fn test_hessian_3d() -> Result<()> {
        let mut prev = f64::MAX;
        for n in 2..5 {
            let nrm = run_hessian_3d(n)?;
            assert!(nrm < 0.5 * prev);
            prev = nrm;
        }
        Ok(())
    }
}
