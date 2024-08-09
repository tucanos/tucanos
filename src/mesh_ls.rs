use crate::{
    mesh::{Point, SimplexMesh},
    topo_elems::Elem,
    Error, Idx, Result,
};

use log::debug;
use rayon::{
    iter::{
        IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator,
        ParallelIterator,
    },
    slice::ParallelSliceMut,
};
use rustc_hash::FxHashSet;

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
    /// $`i`$
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
    /// `dx_df_w` is an iterator that yields $`(\delta^{(j)}, \widetilde{f}_{j}, W_{j})`$ for $`j \in N(i)`$
    ///
    /// If the number of neighbors is not sufficient for this problem to be solved, or if the problem is
    /// too ill-conditioned, None is returned
    fn least_squares<const N: usize, I: ExactSizeIterator<Item = (Point<D>, f64, f64)>>(
        dx_df_w: I,
        w0: Option<f64>,
    ) -> Option<nalgebra::DVector<f64>> {
        debug_assert!(D != 2 || N == 3 || N == 6);
        debug_assert!(D != 3 || N == 4 || N == 10);
        let n_others = dx_df_w.len();
        if n_others < N {
            return None;
        }
        let nrows = 1 + n_others;
        // One allocation to rule them all. An extra N for b and 2 extra N for work.
        // let mut buf = vec![0.; nrows * (N + 3)];
        // let (a, b) = buf.split_at_mut(nrows * N);
        // let (b, work) = b.split_at_mut(nrows);

        let mut mat = nalgebra::DMatrix::<f64>::zeros(nrows, N);
        let mut rhs = nalgebra::DVector::<f64>::zeros(nrows);

        let mut w_max = 0.0;
        for (irow, (dp, df, w)) in dx_df_w.enumerate() {
            let irow = irow + 1;
            let mut row = nalgebra::OMatrix::<f64, nalgebra::Const<1>, nalgebra::Dyn>::zeros(N);
            row[0] = w * 1.0;
            row[1] = w * dp[0];
            row[2] = w * dp[1];
            if D == 2 && N == 6 {
                row[3] = w * 0.5 * dp[0] * dp[0];
                row[4] = w * 0.5 * dp[1] * dp[1];
                row[5] = w * dp[0] * dp[1];
            } else if D == 3 {
                row[3] = w * dp[2];
                if N == 10 {
                    row[4] = w * 0.5 * dp[0] * dp[0];
                    row[5] = w * 0.5 * dp[1] * dp[1];
                    row[6] = w * 0.5 * dp[2] * dp[2];
                    row[7] = w * dp[0] * dp[1];
                    row[8] = w * dp[1] * dp[2];
                    row[9] = w * dp[0] * dp[2];
                }
            }
            mat.set_row(irow, &row);

            rhs[irow] = w * df;
            w_max = f64::max(w_max, w);
        }
        mat[0] = w0.unwrap_or_else(|| f64::sqrt(2.0) * w_max);

        // Solve the least squares problem using a QR decomposition
        let qr = mat.qr();
        qr.q_tr_mul(&mut rhs);
        let r = qr.r();

        let (mind, maxd) = r
            .diagonal()
            .iter()
            .map(|x| (x.abs(), x.abs()))
            .reduce(|(mi1, ma1), (mi2, ma2)| (mi1.min(mi2), ma1.max(ma2)))
            .unwrap();
        if maxd / mind > 1e8 {
            None
        } else {
            assert!(r.solve_upper_triangular_mut(&mut rhs));
            Some(rhs)
        }
    }

    /// Smooth a vertex field using a 1st order weighted least square approximation
    ///
    /// $`N(i)`$ is the set of 1st order neighbors of $`i`$ (i.e. the vertices connected
    /// to $`i`$ by an edge), and a weighting $`W_i=\frac{1}{\left \| \delta_i \right \|^P}`$ is used.
    ///
    /// P is `weight_exp` and typically $`P \in \left \{ 0,1,2 \right \}`$.
    /// $`W_0`$ is set here to $`\sqrt{2} \max(W_j)`$.
    pub fn smooth(&self, f: &[f64], weight_exp: i32) -> Result<Vec<f64>> {
        debug!(
            "Compute smoothing using 1st order LS (weight = {})",
            weight_exp
        );
        let n = self.n_verts() as usize;
        assert_eq!(f.len(), n);
        let mut res = vec![0.0; n];

        let v2v = self.get_vertex_to_vertices()?;
        res.par_iter_mut().enumerate().for_each(|(i_vert, s)| {
            let neighbors = v2v.row(i_vert as Idx);
            let dx_df_w = neighbors.iter().map(|&i| {
                let dx = self.vert(i) - self.vert(i_vert as Idx);
                let df = f[i as usize] - f[i_vert];
                let w = if weight_exp > 0 {
                    1. / dx.norm().powi(weight_exp)
                } else {
                    1.0
                };
                (dx, df, w)
            });

            if D == 2 {
                let sol = Self::least_squares::<3, _>(dx_df_w, None);
                if let Some(sol) = sol {
                    *s = f[i_vert] + sol[0];
                } else {
                    // If the least squared approximation could not be computed, use the original value
                    *s = f[i_vert];
                }
            } else if D == 3 {
                let sol = Self::least_squares::<4, _>(dx_df_w, None);
                if let Some(sol) = sol {
                    *s = f[i_vert] + sol[0];
                } else {
                    // If the least squared approximation could not be computed, use the original value
                    *s = f[i_vert];
                }
            }
        });

        if res.par_iter().copied().any(f64::is_nan) {
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
        let v2v = self.get_vertex_to_vertices().unwrap();
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
    ///
    /// $`N(i)`$ is the set of 1st order neighbors of $`i`$ (i.e. the vertices connected
    /// to $`i`$ by an edge), and a weighting $`W_i=\frac{1}{\left \| \delta_i \right \|^P}`$ is used.
    ///
    /// P is `weight_exp` and typically $`P \in \left \{ 0,1,2 \right \}`$.
    /// $`W_0`$ is set here to $`\sqrt{2} \max(W_j)`$.
    pub fn gradient(&self, f: &[f64], weight_exp: i32) -> Result<Vec<f64>> {
        debug!(
            "Compute gradient using 1st order LS (weight = {})",
            weight_exp
        );
        let n = self.n_verts() as usize;
        assert_eq!(f.len(), n);

        let mut res = vec![0.0; D * n];
        let mut failed = vec![false; self.n_verts() as usize];

        let flg = self.boundary_flag();

        let v2v = self.get_vertex_to_vertices()?;
        res.par_chunks_mut(D)
            .zip(failed.par_iter_mut())
            .enumerate()
            .for_each(|(i_vert, (grad, fail))| {
                // Don't use a LS scheme for boundary vertices
                if flg[i_vert] {
                    *fail = true;
                    grad.iter_mut().for_each(|x| *x = 0.0);
                } else {
                    let neighbors = v2v.row(i_vert as Idx);
                    let dx_df_w = neighbors.iter().map(|&i| {
                        let dx = self.vert(i) - self.vert(i_vert as Idx);
                        let df = f[i as usize] - f[i_vert];
                        let w = if weight_exp > 0 {
                            1. / dx.norm().powi(weight_exp)
                        } else {
                            1.0
                        };
                        (dx, df, w)
                    });

                    if D == 2 {
                        let sol = Self::least_squares::<3, _>(dx_df_w, None);
                        if let Some(sol) = sol {
                            grad.iter_mut()
                                .zip(sol.iter().skip(1).take(D))
                                .for_each(|(x, y)| *x = *y);
                        } else {
                            *fail = true;
                            grad.iter_mut().for_each(|x| *x = 0.0);
                        }
                    } else if D == 3 {
                        let sol = Self::least_squares::<4, _>(dx_df_w, None);
                        if let Some(sol) = sol {
                            grad.iter_mut()
                                .zip(sol.iter().skip(1).take(D))
                                .for_each(|(x, y)| *x = *y);
                        } else {
                            *fail = true;
                            grad.iter_mut().for_each(|x| *x = 0.0);
                        }
                    }
                }
            });

        // For vertices where no valid approximation could be computed, average over the valid neighbors
        if self.fix_not_computed(&mut res, &mut failed, D, 3) {
            if res.iter().copied().any(f64::is_nan) {
                return Err(Error::from("NaN in gradient computation"));
            }
            Ok(res)
        } else {
            Err(Error::from("Cannot compute the value everywhere"))
        }
    }

    /// Compute the hessian of a vertex field using a 2nd order weighted least square approximation
    ///
    /// $`N(i)`$ is the set of 1st order or second order (i.e. the 1st order neighbors of the
    /// 1st order neighbors that are not $`i`$) depending on `use_second_order_neighbors`
    ///
    /// if `weight_exp` is `None` the weights are chosen as $W_0 = 10$, $W_i = 1$ for 1st order neighbors
    /// and $W_i = 0.1$ for 2nd order neighbors (see the PhD of L. Frazza, p. 206). Otherwise, a
    /// weighting $`W_i=\frac{1}{\left \| \delta_i \right \|^P}`$ is used with typical values
    /// $`P \in \left \{ 0,1,2 \right \}`$. In this case, $`W_0`$ is set here to $`\sqrt{2} \max(W_j)`$.
    pub fn hessian(
        &self,
        f: &[f64],
        weight_exp: Option<i32>,
        use_second_order_neighbors: bool,
    ) -> Result<Vec<f64>> {
        debug!("Compute hessian using 2nd order LS");
        if let Some(weight_exp) = weight_exp {
            debug!("  using weight_exp = {weight_exp}");
        } else {
            debug!("  using weights = (10.0, 1.0, 0.1)");
        }
        if use_second_order_neighbors {
            debug!("  using second order neighbors");
        }
        let n = self.n_verts() as usize;
        assert_eq!(f.len(), n);

        let mut res = vec![0.0; D * (D + 1) / 2 * n];
        let mut failed = vec![false; self.n_verts() as usize];

        let v2v = self.get_vertex_to_vertices()?;
        res.par_chunks_mut(D * (D + 1) / 2)
            .zip(failed.par_iter_mut())
            .enumerate()
            .for_each(|(i_vert, (hess, fail))| {
                let first_order_neighbors = v2v.row(i_vert as Idx);
                let mut neighbors = first_order_neighbors
                    .iter()
                    .map(|&i| (i, 1))
                    .collect::<FxHashSet<_>>();
                if use_second_order_neighbors {
                    first_order_neighbors
                        .iter()
                        .for_each(|&i| neighbors.extend(v2v.row(i).iter().map(|&i| (i, 2))));
                    neighbors.remove(&(i_vert as Idx, 2));
                }

                let dx_df_w = neighbors.iter().map(|&(i, order)| {
                    let dx = self.vert(i) - self.vert(i_vert as Idx);
                    let df = f[i as usize] - f[i_vert];
                    let w = weight_exp.map_or(if order == 1 { 1.0 } else { 0.1 }, |weight_exp| {
                        if weight_exp > 0 {
                            1. / dx.norm().powi(weight_exp)
                        } else {
                            1.0
                        }
                    });
                    (dx, df, w)
                });
                let w0 = if weight_exp.is_some() {
                    None
                } else {
                    Some(10.0)
                };

                if D == 2 {
                    let sol = Self::least_squares::<6, _>(dx_df_w, w0);
                    if let Some(sol) = sol {
                        hess.iter_mut()
                            .zip(sol.iter().skip(D + 1).take(D * (D + 1) / 2))
                            .for_each(|(x, y)| *x = *y);
                    } else {
                        *fail = true;
                        hess.iter_mut().for_each(|x| *x = 0.0);
                    }
                } else if D == 3 {
                    let sol = Self::least_squares::<10, _>(dx_df_w, w0);
                    if let Some(sol) = sol {
                        hess.iter_mut()
                            .zip(sol.iter().skip(D + 1).take(D * (D + 1) / 2))
                            .for_each(|(x, y)| *x = *y);
                    } else {
                        *fail = true;
                        hess.iter_mut().for_each(|x| *x = 0.0);
                    }
                }
            });

        // For vertices where no valid approximation could be computed, average over the valid neighbors
        if self.fix_not_computed(&mut res, &mut failed, D * (D + 1) / 2, 3) {
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
        Result,
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

        let v = mesh.get_vertex_volumes()?;

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

        let v = mesh.get_vertex_volumes()?;

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

        // with only the 1st order neighbors
        let res = mesh.hessian(&f, Some(2), false)?;
        for i_vert in 0..mesh.n_verts() as usize {
            assert!(f64::abs(res[3 * i_vert] - 2.) < 1e-10);
            assert!(f64::abs(res[3 * i_vert + 1] - 4.) < 1e-10);
            assert!(f64::abs(res[3 * i_vert + 2] - 3.) < 1e-10);
        }

        // with the 2nd order neighbors
        let res = mesh.hessian(&f, Some(2), true)?;
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

        // with only the 1st order neighbors
        let res = mesh.hessian(&f, Some(2), false)?;
        for i_vert in 0..mesh.n_verts() as usize {
            assert!(f64::abs(res[6 * i_vert] - 2.) < 1e-10);
            assert!(f64::abs(res[6 * i_vert + 1] - 4.) < 1e-10);
            assert!(f64::abs(res[6 * i_vert + 2] - 6.) < 1e-10);
            assert!(f64::abs(res[6 * i_vert + 3] - 4.) < 1e-10);
            assert!(f64::abs(res[6 * i_vert + 4] - 5.) < 1e-10);
            assert!(f64::abs(res[6 * i_vert + 5] - 6.) < 1e-10);
        }

        // with the 2nd order neighbors
        let res = mesh.hessian(&f, Some(2), true)?;
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

    fn run_hessian_2d(
        n: u32,
        weight_exp: Option<i32>,
        use_second_order_neighbors: bool,
    ) -> Result<f64> {
        let mut mesh = test_mesh_2d();
        for _ in 0..n {
            mesh = mesh.split();
        }

        mesh.compute_vertex_to_vertices();
        mesh.compute_volumes();

        let v = mesh.get_vertex_volumes()?;

        let f: Vec<_> = mesh
            .verts()
            .map(|p| p[0] * p[0] * p[1] + 2.0 * p[0] * p[1] * p[1])
            .collect();
        let res = mesh.hessian(&f, weight_exp, use_second_order_neighbors)?;
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
        // with only the 1st order neighbors & classical weights
        for w in 0..3 {
            let mut prev = f64::MAX;
            for n in 3..7 {
                let nrm = run_hessian_2d(n, Some(w), false)?;
                assert!(nrm < 0.5 * prev);
                prev = nrm;
            }
        }

        // with only the 2nd order neighbors & classical weights
        for w in 0..3 {
            let mut prev = f64::MAX;
            for n in 3..7 {
                let nrm = run_hessian_2d(n, Some(w), true)?;
                assert!(nrm < 0.5 * prev);
                prev = nrm;
            }
        }

        // with only the 2nd order neighbors & the weights given by Frazza
        let mut prev = f64::MAX;
        for n in 3..7 {
            let nrm = run_hessian_2d(n, None, true)?;
            assert!(nrm < 0.5 * prev);
            prev = nrm;
        }

        Ok(())
    }

    fn run_hessian_3d(
        num_split: u32,
        weight_exp: Option<i32>,
        use_second_order_neighbors: bool,
    ) -> Result<f64> {
        let mut mesh = test_mesh_3d();
        for _ in 0..num_split {
            mesh = mesh.split();
        }

        mesh.compute_vertex_to_vertices();
        mesh.compute_volumes();

        let vols = mesh.get_vertex_volumes()?;

        let test_f: Vec<_> = mesh
            .verts()
            .map(|p| {
                let x = p[0];
                let y = p[1];
                let z = p[2];
                x * x * y * z + 2.0 * x * y * y * z + 3.0 * x * y * z * z
            })
            .collect();
        let res = mesh.hessian(&test_f, weight_exp, use_second_order_neighbors)?;
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
        // with only the 1st order neighbors & classical weights
        for w in 0..3 {
            let mut prev = f64::MAX;
            for n in 2..5 {
                let nrm = run_hessian_3d(n, Some(w), false)?;
                assert!(nrm < 0.5 * prev);
                prev = nrm;
            }
        }

        // with only the 2nd order neighbors & classical weights
        for w in 0..3 {
            let mut prev = f64::MAX;
            for n in 2..5 {
                let nrm = run_hessian_3d(n, Some(w), true)?;
                assert!(nrm < 0.5 * prev);
                prev = nrm;
            }
        }

        // with only the 2nd order neighbors & the weights given by Frazza
        let mut prev = f64::MAX;
        for n in 2..5 {
            let nrm = run_hessian_3d(n, None, true)?;
            assert!(nrm < 0.55 * prev);
            prev = nrm;
        }

        Ok(())
    }
}
