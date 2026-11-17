//! Weighted least square gradient computation
use crate::{Error, Result, graph::CSRGraph, mesh::Mesh};
use nalgebra::{Const, DMatrix, DVector, Dim, Dyn, OMatrix, SMatrix, SVector};
use rayon::{
    iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator},
    slice::ParallelSliceMut,
};
use rustc_hash::FxHashSet;

/// Weighted least-squares derivative reconstruction with SVD preconditioning.
///
/// For a vertex `x_i`, we use neighbor offsets `delta_j = x_j - x_i` and
/// differences `df_j = f_j - f_i` and solve a weighted least-squares problem
/// for the local polynomial coefficients.
///
/// We minimize:
/// `min_c sum_j w_j^2 (phi(delta_j)^T c - df_j)^2 + lambda c_0^2`
///
/// where `c = [c_0, a_1, ..., a_D, ...]` contains the constant correction,
/// gradient and (for order 2) Hessian-related coefficients, and `phi(delta_j)`
/// is the basis vector evaluated at `delta_j`.
///
/// Order 1 basis:
/// `phi = [1, delta_1, ..., delta_D]`
///
/// Order 2 basis extends order 1 with:
/// - `1/2 delta_k^2` terms
/// - `delta_k delta_l` cross terms (`k < l`)
///
/// This yields the model residual:
/// `r_j = c_0 + a_k delta_{j,k} + 1/2 H_kk delta_{j,k}^2 + H_kl delta_{j,k} delta_{j,l} - df_j`
///
/// and the objective `sum_j w_j^2 r_j^2` (plus the `lambda c_0^2` stabilization
/// on the constant term).
///
/// where repeated indices denote component-wise sums and the exact basis terms
/// depend on `D`.
///
/// Neighbor contributions are weighted with
/// `w_j = 1 / ||delta_j||^p`, where `p = weight_exp`:
/// - `p = 0`: all neighbors contribute equally
/// - larger `p`: closer neighbors contribute more
///
/// The implementation normalizes by `w_max` so only relative weights matter.
///
/// ## SVD preconditioning
///
/// To improve conditioning on stretched/anisotropic stencils, we build the
/// weighted coordinate matrix `X_w` with rows `w_j * delta_j^T`, then compute
/// `X_w = U Sigma V^T`.
///
/// A whitening transform is built from clamped inverse singular values
/// (`1 / max(sigma_i, sigma_max * 1e-14)`), yielding:
/// - `P = Sigma^{-1} V^T` (used to transform coordinates)
/// - `T_inv = V Sigma^{-1}` (used to map derivatives back)
///
/// Each offset is transformed as `delta'_j = P delta_j` before basis
/// evaluation.
///
/// ## Linear system and solve
///
/// The weighted normal equations are assembled directly in transformed space:
/// - `N = A^T A`
/// - `B = A^T W` (stored as `b_mat`)
///
/// For each query, with `df` stacked in a vector:
/// - `rhs = B * df`
/// - solve `N x = rhs` using Cholesky factorization
///
/// ## Recovering physical derivatives
///
/// - Smoothed value correction is `x[0]`
/// - Gradient coefficients are mapped back with `grad = T_inv * g'`
/// - Hessian coefficients are reconstructed as a symmetric `H'`, then mapped
///   with `H = T_inv * H' * T_inv^T`
///
/// Output Hessian storage uses:
/// - 2D: `[xx, yy, xy]`
/// - 3D: `[xx, yy, zz, xy, yz, xz]`
pub struct LeastSquaresGradient<const D: usize> {
    cholesky: nalgebra::Cholesky<f64, Dyn>,
    b_mat: DMatrix<f64>, // M x N matrix caching weights for rapid RHS assembly
    t_inv: SMatrix<f64, D, D>,
}

/// Implementation of the SVD-Preconditioned Least Squares gradient computation.
impl<const D: usize> LeastSquaresGradient<D> {
    pub fn new(
        order: i32,
        weight_exp: i32,
        dx_iter: impl ExactSizeIterator<Item = SVector<f64, D>>,
    ) -> Result<Self> {
        let dx: Vec<_> = dx_iter.collect();
        let n_neighbors = dx.len();
        let n_cols = if order == 1 {
            D + 1
        } else {
            D + 1 + D * (D + 1) / 2
        };

        let mut weights = vec![0.0; n_neighbors];
        let mut w_max = 0.0;

        for (i, dp) in dx.iter().enumerate() {
            let w = 1.0 / dp.norm().powi(weight_exp);
            weights[i] = w;
            w_max = f64::max(w_max, w);
        }

        for w in &mut weights {
            *w /= w_max;
        }
        w_max = 1.0;

        // 1. Whitening Transformation via SVD on Weighted Coordinates
        let mut x_w = OMatrix::<f64, Dyn, Const<D>>::zeros(n_neighbors);
        for (i, dp) in dx.iter().enumerate() {
            for j in 0..D {
                x_w[(i, j)] = weights[i] * dp[j];
            }
        }

        let svd = x_w.svd(false, true);
        let v_t = svd.v_t.unwrap();
        let v = v_t.transpose();

        let mut inv_sv = SVector::<f64, D>::zeros();
        // println!("singular values: {:?}", svd.singular_values);
        let max_sv = svd.singular_values[0].max(1e-20);
        let min_sv = max_sv * 1e-14;

        for i in 0..D {
            let sv = svd.singular_values[i];
            inv_sv[i] = 1.0 / sv.max(min_sv); // Clamp numerical noise without exploding
        }

        let p_mat = SMatrix::<f64, D, D>::from_diagonal(&inv_sv) * v_t;
        let t_inv = v * SMatrix::<f64, D, D>::from_diagonal(&inv_sv);

        // 2. Direct Assembly of Normal Matrix and RHS Projection Matrix
        let mut n_mat = DMatrix::<f64>::zeros(n_cols, n_cols);
        n_mat[(0, 0)] = 2.0 * w_max * w_max; // Central node constraint squared

        let mut b_mat = DMatrix::<f64>::zeros(n_cols, n_neighbors);
        let mut basis = vec![0.0; n_cols]; // Pre-allocate small buffer for basis evaluation

        for (i, dp_orig) in dx.iter().enumerate() {
            let w = weights[i];
            let dp = p_mat * dp_orig;

            // Evaluate the unweighted polynomial basis
            basis[0] = 1.0;
            for j in 0..D {
                basis[j + 1] = dp[j];
            }

            if order == 2 {
                if D == 2 {
                    basis[3] = 0.5 * dp[0] * dp[0];
                    basis[4] = 0.5 * dp[1] * dp[1];
                    basis[5] = dp[0] * dp[1];
                } else if D == 3 {
                    basis[4] = 0.5 * dp[0] * dp[0];
                    basis[5] = 0.5 * dp[1] * dp[1];
                    basis[6] = 0.5 * dp[2] * dp[2];
                    basis[7] = dp[0] * dp[1];
                    basis[8] = dp[1] * dp[2];
                    basis[9] = dp[0] * dp[2];
                }
            }

            // Both N_mat and B_mat scale by w^2 (since tilde_a already includes a weight factor)
            let w2 = w * w;

            // Accumulate RHS Projection Matrix (B = A^T * W)
            for j in 0..n_cols {
                b_mat[(j, i)] = w2 * basis[j];
            }

            // Accumulate Symmetric Normal Matrix (N = A^T * A)
            for row in 0..n_cols {
                let row_val = w2 * basis[row];
                for col in 0..n_cols {
                    n_mat[(row, col)] += row_val * basis[col];
                }
            }
        }

        // 3. Fast Square Inversion
        let cholesky = nalgebra::Cholesky::new(n_mat)
            .ok_or_else(|| Error::from("Failed Cholesky decomposition of normal equations"))?;

        Ok(Self {
            cholesky,
            b_mat,
            t_inv,
        })
    }

    fn compute(&self, df: impl ExactSizeIterator<Item = f64>) -> DVector<f64> {
        let df_vec = DVector::from_iterator(df.len(), df);
        let rhs = &self.b_mat * df_vec;
        self.cholesky.solve(&rhs)
    }

    pub fn smooth(&self, df: impl ExactSizeIterator<Item = f64>) -> f64 {
        self.compute(df)[0]
    }

    pub fn gradient(&self, df: impl ExactSizeIterator<Item = f64>) -> SVector<f64, D> {
        let x = self.compute(df);
        let g = x.fixed_view::<D, 1>(1, 0).into_owned();
        self.t_inv * g // Recover physical gradient
    }

    pub fn hessian(&self, df: impl ExactSizeIterator<Item = f64>, res: &mut [f64]) {
        let rhs = self.compute(df);
        let mut h_prime = SMatrix::<f64, D, D>::zeros();

        if D == 2 {
            h_prime[(0, 0)] = rhs[3];
            h_prime[(1, 1)] = rhs[4];
            h_prime[(0, 1)] = rhs[5];
            h_prime[(1, 0)] = rhs[5];
        } else if D == 3 {
            h_prime[(0, 0)] = rhs[4];
            h_prime[(1, 1)] = rhs[5];
            h_prime[(2, 2)] = rhs[6];
            h_prime[(0, 1)] = rhs[7];
            h_prime[(1, 0)] = rhs[7];
            h_prime[(1, 2)] = rhs[8];
            h_prime[(2, 1)] = rhs[8];
            h_prime[(0, 2)] = rhs[9];
            h_prime[(2, 0)] = rhs[9];
        }

        // Recover physical Hessian
        let h = self.t_inv * h_prime * self.t_inv.transpose();

        if D == 2 {
            res[0] = h[(0, 0)];
            res[1] = h[(1, 1)];
            res[2] = h[(0, 1)];
        } else if D == 3 {
            res[0] = h[(0, 0)];
            res[1] = h[(1, 1)];
            res[2] = h[(2, 2)];
            res[3] = h[(0, 1)];
            res[4] = h[(1, 2)];
            res[5] = h[(0, 2)];
        }
    }

    pub fn gradient_weights(&self) -> impl ExactSizeIterator<Item = SVector<f64, D>> + '_ {
        // Solves the matrix W directly via normal equations for all neighbors at once
        let w_mat = self.cholesky.solve(&self.b_mat);

        (0..self.b_mat.ncols()).map(move |i| {
            let g = w_mat.fixed_view::<D, 1>(1, i).into_owned();
            self.t_inv * g
        })
    }
}

/// For vertices for which a least square approximation could not be computed, use the average over valid neighbors
/// res: The result from the least square approximation
/// failed: Flag that indicates if not valid approximation has been computed
/// m: The number of components (1 for scalar, D for gradients, D*(D-1)/2 for hessians)
/// `max_iter`: The max number of iteration through the mesh vertices
fn fix_not_computed(
    v2v: &CSRGraph,
    res: &mut [f64],
    failed: &mut [bool],
    m: usize,
    max_iter: usize,
) -> bool {
    for _ in 0..max_iter {
        for i_vert in 0..v2v.n() {
            if failed[i_vert] {
                let mut n = 0;
                for i_other in v2v.row(i_vert).iter().copied() {
                    if !failed[i_other] {
                        for i in 0..m {
                            res[m * i_vert + i] += res[m * i_other + i];
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

/// Compute the gradient of a field defined of the mesh vertices using weighted
/// least squares
pub fn gradient<const D: usize, M: Mesh<D>>(
    msh: &M,
    v2v: &CSRGraph,
    order: i32,
    weight: i32,
    f: &[f64],
) -> Result<Vec<f64>>
where
    Const<D>: Dim,
{
    let mut res = vec![0.0; D * msh.n_verts()];
    let flg = msh.boundary_flag();
    let mut failed = vec![false; msh.n_verts()];

    res.par_chunks_mut(D)
        .zip(failed.par_iter_mut())
        .enumerate()
        .for_each(|(i, (grad, fail))| {
            if flg[i] {
                *fail = true;
            } else {
                let x = msh.vert(i);
                let first_order_neighbors = v2v.row(i);
                let mut neighbors = first_order_neighbors
                    .iter()
                    .copied()
                    .collect::<FxHashSet<_>>();
                if order == 2 {
                    for &i in first_order_neighbors {
                        neighbors.extend(v2v.row(i).iter().copied());
                    }
                    neighbors.remove(&i);
                }
                let dx = neighbors.iter().map(|&j| msh.vert(j) - x);
                if let Ok(ls) = LeastSquaresGradient::new(order, weight, dx) {
                    let df = neighbors.iter().map(|&j| f[j] - f[i]);
                    grad.iter_mut()
                        .zip(ls.gradient(df).as_slice())
                        .for_each(|(x, y)| *x = *y);
                } else {
                    *fail = true;
                }
            }
        });

    if fix_not_computed(v2v, &mut res, &mut failed, D, 6) {
        if res.iter().copied().any(f64::is_nan) {
            return Err(Error::from("NaN in gradient computation"));
        }
        Ok(res)
    } else {
        Err(Error::from("Cannot compute the value everywhere"))
    }
}

/// Compute the hessian of a field defined of the mesh vertices using weighted
/// least squares
pub fn hessian<const D: usize, M: Mesh<D>>(
    msh: &M,
    v2v: &CSRGraph,
    weight: i32,
    f: &[f64],
) -> Result<Vec<f64>>
where
    Const<D>: Dim,
{
    let mut res = vec![0.0; D * (D + 1) / 2 * msh.n_verts()];
    let flg = msh.boundary_flag();
    let mut failed = vec![false; msh.n_verts()];

    res.par_chunks_mut(D * (D + 1) / 2)
        .zip(failed.par_iter_mut())
        .enumerate()
        .for_each(|(i, (hess, fail))| {
            if flg[i] {
                *fail = true;
            } else {
                let x = msh.vert(i);
                let first_order_neighbors = v2v.row(i);
                let mut neighbors = first_order_neighbors
                    .iter()
                    .copied()
                    .collect::<FxHashSet<_>>();
                for &j in first_order_neighbors {
                    neighbors.extend(v2v.row(j).iter().copied());
                }
                neighbors.remove(&i);
                let dx = neighbors.iter().map(|&j| msh.vert(j) - x);
                if let Ok(ls) = LeastSquaresGradient::new(2, weight, dx) {
                    let df = neighbors.iter().map(|&j| f[j] - f[i]);
                    ls.hessian(df, hess);
                } else {
                    *fail = true;
                }
            }
        });

    if fix_not_computed(v2v, &mut res, &mut failed, D * (D + 1) / 2, 6) {
        if res.iter().copied().any(f64::is_nan) {
            return Err(Error::from("NaN in gradient computation"));
        }
        Ok(res)
    } else {
        Err(Error::from("Cannot compute the value everywhere"))
    }
}

/// Compute the gradient of a field defined of the mesh vertices using weighted
/// least squares
pub fn smooth<const D: usize, M: Mesh<D>>(
    msh: &M,
    v2v: &CSRGraph,
    order: i32,
    weight: i32,
    f: &[f64],
) -> Vec<f64>
where
    Const<D>: Dim,
{
    let mut res = f.to_vec();

    let flg = msh.boundary_flag();

    res.par_iter_mut().enumerate().for_each(|(i, f_new)| {
        if !flg[i] {
            let x = msh.vert(i);
            let first_order_neighbors = v2v.row(i);
            let mut neighbors = first_order_neighbors
                .iter()
                .copied()
                .collect::<FxHashSet<_>>();
            if order == 2 {
                for &i in first_order_neighbors {
                    neighbors.extend(v2v.row(i).iter().copied());
                }
                neighbors.remove(&i);
            }
            let dx = neighbors.iter().map(|&j| msh.vert(j) - x);
            if let Ok(ls) = LeastSquaresGradient::new(order, weight, dx) {
                let df = neighbors.iter().map(|&j| f[j] - f[i]);
                *f_new += ls.smooth(df);
            }
        }
    });

    res
}

#[cfg(test)]
mod tests {
    use super::LeastSquaresGradient;
    use nalgebra::{SMatrix, SVector};
    use rand::{RngExt, SeedableRng, rngs::StdRng};

    fn assert_ls_gradient<const D: usize>(
        ls: &LeastSquaresGradient<D>,
        df: &[f64],
        expected: SVector<f64, D>,
        tol: f64,
    ) {
        let grad_1 = ls.gradient(df.iter().copied());
        let err = (expected - grad_1).norm() / expected.norm();
        assert!(err < tol, "relative gradient error {err:.2e} >= {tol:.2e}");

        let mut grad_2 = SVector::<f64, D>::zeros();
        for (g, dfi) in ls.gradient_weights().zip(df.iter().copied()) {
            grad_2 += dfi * g;
        }
        let err_2 = (grad_2 - grad_1).norm() / expected.norm();
        assert!(
            err_2 < tol,
            "gradient weights mismatch {err_2:.2e} >= {tol:.2e}"
        );
    }

    fn run_ls_case_2d(n_neighbors: i32, aniso: f64, weight: i32, order: i32, tol: f64) {
        let mut rng = StdRng::seed_from_u64(1234);
        let h = [1.0, aniso];
        let a = 10.0_f64.to_radians();
        let (s, c) = a.sin_cos();
        let rot = SMatrix::<f64, 2, 2>::new(c, s, -s, c);
        let dx = (0..n_neighbors)
            .map(|_| rot * SVector::<f64, 2>::from_fn(|i, _| h[i] * (rng.random::<f64>() - 0.5)))
            .collect::<Vec<_>>();
        let grad = SVector::<f64, 2>::new(1.2 / h[0], 2.3 / h[1]);
        let df = dx.iter().map(|dp| grad.dot(dp)).collect::<Vec<_>>();

        let ls = LeastSquaresGradient::new(order, weight, dx.iter().copied()).unwrap();
        assert_ls_gradient(&ls, &df, grad, tol);
    }

    fn run_ls_case_3d(n_neighbors: i32, aniso: f64, weight: i32, order: i32, tol: f64) {
        let mut rng = StdRng::seed_from_u64(1234);
        let h = [1.0, aniso.sqrt(), aniso];
        let a = 10.0_f64.to_radians();
        let (s, c) = a.sin_cos();
        let rot_x = SMatrix::<f64, 3, 3>::new(1.0, 0.0, 0.0, 0.0, c, s, 0.0, -s, c);
        let rot_y = SMatrix::<f64, 3, 3>::new(c, 0.0, -s, 0.0, 1.0, 0.0, s, 0.0, c);
        let rot = rot_y * rot_x;
        let dx = (0..n_neighbors)
            .map(|_| rot * SVector::<f64, 3>::from_fn(|i, _| h[i] * (rng.random::<f64>() - 0.5)))
            .collect::<Vec<_>>();
        let grad = SVector::<f64, 3>::new(1.2 / h[0], 2.3 / h[1], 3.4 / h[2]);
        let df = dx.iter().map(|dp| grad.dot(dp)).collect::<Vec<_>>();

        let ls = LeastSquaresGradient::new(order, weight, dx.iter().copied()).unwrap();
        assert_ls_gradient(&ls, &df, grad, tol);
    }

    #[test]
    fn test_ls_2d() {
        let cases = [
            (1.0, 0, 1, 1e-14),
            (1.0, 1, 1, 1e-14),
            (1.0, 2, 1, 1e-14),
            (1.0, 0, 2, 1e-14),
            (1.0, 1, 2, 1e-14),
            (1.0, 2, 2, 1e-14),
            (1e-6, 0, 1, 1e-10),
            (1e-6, 1, 1, 1e-10),
            (1e-6, 2, 1, 1e-10),
            (1e-6, 0, 2, 1e-10),
            (1e-6, 1, 2, 1e-10),
            (1e-6, 2, 2, 1e-10),
        ];

        for (aniso, weight, order, tol) in cases {
            run_ls_case_2d(10, aniso, weight, order, tol);
        }
    }

    #[test]
    fn test_ls_3d() {
        let cases = [
            (1.0, 0, 1, 1e-14),
            (1.0, 1, 1, 1e-14),
            (1.0, 2, 1, 1e-14),
            (1.0, 0, 2, 1e-14),
            (1.0, 1, 2, 1e-14),
            (1.0, 2, 2, 1e-14),
            (1e-6, 0, 1, 1e-10),
            (1e-6, 1, 1, 1e-10),
            (1e-6, 2, 1, 1e-10),
            (1e-6, 0, 2, 1e-10),
            (1e-6, 1, 2, 1e-10),
            (1e-6, 2, 2, 1e-10),
        ];

        for (aniso, weight, order, tol) in cases {
            run_ls_case_3d(30, aniso, weight, order, tol);
        }
    }
}
