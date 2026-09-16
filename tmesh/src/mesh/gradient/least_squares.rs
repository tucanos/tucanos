//! Weighted least square gradient computation
use crate::{Error, Result, graph::CSRGraph, mesh::Mesh};
use nalgebra::{Const, DMatrix, DVector, Dim, Dyn, OMatrix, QR, SMatrix, SVector};
use rayon::{
    iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator},
    slice::ParallelSliceMut,
};
use rustc_hash::FxHashSet;

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
pub struct LeastSquaresGradientQR<const D: usize> {
    qr: QR<f64, Dyn, Dyn>,
    r: DMatrix<f64>,
    weights: Vec<f64>,
    order: i32,
}

impl<const D: usize> LeastSquaresGradientQR<D> {
    /// Initialize the WLS computation
    pub fn new(
        order: i32,
        weight_exp: i32,
        dx: impl ExactSizeIterator<Item = SVector<f64, D>>,
    ) -> Result<Self> {
        assert!(D == 2 || D == 3);
        assert!(order == 1 || order == 2);

        let n_rows = 1 + dx.len();
        let n_cols = if order == 1 {
            D + 1
        } else {
            D + 1 + D * (D + 1) / 2
        };

        let mut mat = DMatrix::zeros(n_rows, n_cols);
        let mut weights = vec![0.0; n_rows - 1];

        let mut w_max = 0.0;
        for (irow, dp) in dx.enumerate() {
            let w = 1.0 / dp.norm().powi(weight_exp);
            weights[irow] = w;
            let irow = irow + 1;
            let mut row = OMatrix::<f64, Const<1>, Dyn>::zeros(n_cols);
            row[0] = w * 1.0;
            for i in 0..D {
                row[i + 1] = w * dp[i];
            }
            if order == 2 {
                if D == 2 {
                    row[3] = w * 0.5 * dp[0] * dp[0];
                    row[4] = w * 0.5 * dp[1] * dp[1];
                    row[5] = w * dp[0] * dp[1];
                } else if D == 3 {
                    row[4] = w * 0.5 * dp[0] * dp[0];
                    row[5] = w * 0.5 * dp[1] * dp[1];
                    row[6] = w * 0.5 * dp[2] * dp[2];
                    row[7] = w * dp[0] * dp[1];
                    row[8] = w * dp[1] * dp[2];
                    row[9] = w * dp[0] * dp[2];
                }
            }

            mat.set_row(irow, &row);
            w_max = f64::max(w_max, w);
        }
        mat[0] = f64::sqrt(2.0) * w_max;

        let qr = mat.qr();
        let r = qr.r();

        let (mind, maxd) = r
            .diagonal()
            .iter()
            .map(|x| (x.abs(), x.abs()))
            .reduce(|(mi1, ma1), (mi2, ma2)| (mi1.min(mi2), ma1.max(ma2)))
            .unwrap();
        if maxd / mind > 1e8 {
            Err(Error::from(&format!(
                "Poor conditionning in QR: {:.2e}",
                maxd / mind
            )))
        } else {
            Ok(Self {
                qr,
                r,
                weights,
                order,
            })
        }
    }

    fn compute(&self, df: impl ExactSizeIterator<Item = f64>) -> DVector<f64> {
        assert_eq!(df.len(), self.weights.len());
        let mut rhs = DVector::<f64>::zeros(df.len() + 1);
        for (irow, df) in df.enumerate() {
            let w = self.weights[irow];
            let irow = irow + 1;
            let tmp = w * df;
            rhs[irow] = tmp;
        }
        self.qr.q_tr_mul(&mut rhs);
        assert!(self.r.solve_upper_triangular_mut(&mut rhs));
        rhs
    }

    /// Smoothing
    pub fn smooth(&self, df: impl ExactSizeIterator<Item = f64>) -> f64 {
        let rhs = self.compute(df);

        rhs[0]
    }

    /// Compute the gradient
    pub fn gradient(&self, df: impl ExactSizeIterator<Item = f64>) -> SVector<f64, D> {
        let rhs = self.compute(df);

        rhs.fixed_view::<D, 1>(1, 0).into()
    }

    /// Compute the gradient
    pub fn hessian(&self, df: impl ExactSizeIterator<Item = f64>, res: &mut [f64]) {
        assert_eq!(self.order, 2);
        assert_eq!(res.len(), D * (D + 1) / 2);

        let rhs = self.compute(df);
        res.iter_mut()
            .zip(rhs.iter().skip(D + 1))
            .for_each(|(x, y)| *x = *y);
    }

    /// Compute the gradient weights
    #[allow(dead_code)]
    #[must_use]
    pub fn gradient_weights(&self) -> impl ExactSizeIterator<Item = SVector<f64, D>> + '_ {
        let mut rhs = DMatrix::<f64>::zeros(self.weights.len() + 1, 1);

        self.weights.iter().enumerate().map(move |(irow, &w)| {
            rhs.fill(0.0);
            rhs[irow + 1] = w;
            self.qr.q_tr_mul(&mut rhs);
            assert!(self.r.solve_upper_triangular_mut(&mut rhs));
            rhs.fixed_view::<D, 1>(1, 0).into()
        })
    }
}

/// SVD-Preconditioned Least Squares utilizing fast Normal Equations
pub struct PreconditionedLeastSquares<const D: usize> {
    cholesky: nalgebra::Cholesky<f64, Dyn>,
    b_mat: DMatrix<f64>, // M x N matrix caching weights for rapid RHS assembly
    t_inv: SMatrix<f64, D, D>,
}

impl<const D: usize> PreconditionedLeastSquares<D> {
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

    #[must_use]
    pub fn gradient_weights(&self) -> impl ExactSizeIterator<Item = SVector<f64, D>> + '_ {
        // Solves the matrix W directly via normal equations for all neighbors at once
        let w_mat = self.cholesky.solve(&self.b_mat);

        (0..self.b_mat.ncols()).map(move |i| {
            let g = w_mat.fixed_view::<D, 1>(1, i).into_owned();
            self.t_inv * g
        })
    }
}

/// Unified Interface mapping to the standard or preconditioned solver
pub enum LeastSquaresGradient<const D: usize> {
    Standard(LeastSquaresGradientQR<D>),
    Preconditioned(PreconditionedLeastSquares<D>),
}

impl<const D: usize> LeastSquaresGradient<D> {
    pub fn new(
        order: i32,
        weight_exp: i32,
        preconditioned: bool,
        dx_iter: impl ExactSizeIterator<Item = SVector<f64, D>> + Clone,
    ) -> Result<Self> {
        if preconditioned {
            Ok(Self::Preconditioned(PreconditionedLeastSquares::new(
                order, weight_exp, dx_iter,
            )?))
        } else {
            Ok(Self::Standard(LeastSquaresGradientQR::new(
                order, weight_exp, dx_iter,
            )?))
        }
    }

    pub fn smooth(&self, df: impl ExactSizeIterator<Item = f64>) -> f64 {
        match self {
            Self::Standard(ls) => ls.smooth(df),
            Self::Preconditioned(ls) => ls.smooth(df),
        }
    }

    pub fn gradient(&self, df: impl ExactSizeIterator<Item = f64>) -> SVector<f64, D> {
        match self {
            Self::Standard(ls) => ls.gradient(df),
            Self::Preconditioned(ls) => ls.gradient(df),
        }
    }

    pub fn hessian(&self, df: impl ExactSizeIterator<Item = f64>, res: &mut [f64]) {
        match self {
            Self::Standard(ls) => ls.hessian(df, res),
            Self::Preconditioned(ls) => ls.hessian(df, res),
        }
    }

    #[must_use]
    pub fn gradient_weights(&self) -> Box<dyn ExactSizeIterator<Item = SVector<f64, D>> + '_> {
        match self {
            Self::Standard(ls) => Box::new(ls.gradient_weights()),
            Self::Preconditioned(ls) => Box::new(ls.gradient_weights()),
        }
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
    preconditioned: bool,
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
                if let Ok(ls) = LeastSquaresGradient::new(order, weight, preconditioned, dx) {
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
    preconditioned: bool,
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
                if let Ok(ls) = LeastSquaresGradient::new(2, weight, preconditioned, dx) {
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
    preconditioned: bool,
    boundary: bool,
    f: &[f64],
) -> Vec<f64>
where
    Const<D>: Dim,
{
    let mut res = f.to_vec();

    let flg = msh.boundary_flag();

    res.par_iter_mut().enumerate().for_each(|(i, f_new)| {
        if boundary || !flg[i] {
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
            if let Ok(ls) = LeastSquaresGradient::new(order, weight, preconditioned, dx) {
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

    #[test]
    fn test_ls_2d() {
        let test = |n_neighbors: i32, aniso: f64, weight: i32, order: i32, tol: f64| {
            let mut rng = StdRng::seed_from_u64(1234);
            let h = [1.0, aniso];
            let a = 10.0_f64.to_radians();
            let (s, c) = a.sin_cos();
            let rot = SMatrix::<f64, 2, 2>::new(c, s, -s, c);
            let dx = (0..n_neighbors)
                .map(|_| {
                    rot * SVector::<f64, 2>::from_fn(|i, _| h[i] * (rng.random::<f64>() - 0.5))
                })
                .collect::<Vec<_>>();
            // the gradient is not exactly aligned with the mesh anisotropy
            let grad = SVector::<f64, 2>::new(1.2 / h[0], 2.3 / h[1]);

            for pc in [false, true] {
                let df = dx.iter().map(|dx| grad.dot(dx)).collect::<Vec<_>>();
                if let Ok(ls) = LeastSquaresGradient::new(order, weight, pc, dx.iter().copied()) {
                    let grad_1 = ls.gradient(df.iter().copied());
                    let err = (grad - grad_1).norm() / grad.norm();
                    println!("pc = {pc}, err = {err:.2e}");
                    assert!(err < tol);

                    let mut grad_2 = SVector::<f64, 2>::zeros();
                    for (g, df) in ls.gradient_weights().zip(df) {
                        grad_2 += df * g;
                    }
                    let err_2 = (grad_2 - grad_1).norm() / grad.norm();
                    // println!("pc = {pc}, err = {err_2:.2e}");
                    assert!(err_2 < tol);
                } else {
                    assert!(!pc);
                    println!("pc = {pc}, LeastSquaresGradient fails");
                }
            }
        };

        // isotropic
        println!("iso / linear");
        test(10, 1.0, 0, 1, 1e-14);
        test(10, 1.0, 1, 1, 1e-14);
        test(10, 1.0, 2, 1, 1e-14);

        // isotropic + quadratic
        println!("iso / quadratic");
        test(10, 1.0, 0, 2, 1e-14);
        test(10, 1.0, 1, 2, 1e-14);
        test(10, 1.0, 2, 2, 1e-14);

        // anisotropic
        println!("aniso / linear");
        test(10, 1e-6, 0, 1, 1e-10);
        test(10, 1e-6, 1, 1, 1e-10);
        test(10, 1e-6, 2, 1, 1e-10);

        // anisotropic
        println!("aniso / quadratic");
        test(10, 1e-6, 0, 2, 1e-10);
        test(10, 1e-6, 1, 2, 1e-10);
        test(10, 1e-6, 2, 2, 1e-10);
    }

    #[test]
    fn test_ls_3d() {
        let test = |n_neighbors: i32, aniso: f64, weight: i32, order: i32, tol: f64| {
            let mut rng = StdRng::seed_from_u64(1234);
            let h = [1.0, aniso.sqrt(), aniso];
            let a = 10.0_f64.to_radians();
            let (s, c) = a.sin_cos();
            // Rotation around X axis
            let rot_x = SMatrix::<f64, 3, 3>::new(1.0, 0.0, 0.0, 0.0, c, s, 0.0, -s, c);

            // Rotation around Y axis
            let rot_y = SMatrix::<f64, 3, 3>::new(c, 0.0, -s, 0.0, 1.0, 0.0, s, 0.0, c);

            // Compound rotation matrix
            let rot = rot_y * rot_x;
            let dx = (0..n_neighbors)
                .map(|_| {
                    rot * SVector::<f64, 3>::from_fn(|i, _| h[i] * (rng.random::<f64>() - 0.5))
                })
                .collect::<Vec<_>>();
            // the gradient is not exactly aligned with the mesh anisotropy
            let grad = SVector::<f64, 3>::new(1.2 / h[0], 2.3 / h[1], 3.4 / h[2]);

            for pc in [false, true] {
                let df = dx.iter().map(|dx| grad.dot(dx)).collect::<Vec<_>>();
                if let Ok(ls) = LeastSquaresGradient::new(order, weight, pc, dx.iter().copied()) {
                    let grad_1 = ls.gradient(df.iter().copied());
                    let err = (grad - grad_1).norm() / grad.norm();
                    println!("pc = {pc}, err = {err:.2e}");
                    assert!(err < tol);

                    let mut grad_2 = SVector::<f64, 3>::zeros();
                    for (g, df) in ls.gradient_weights().zip(df) {
                        grad_2 += df * g;
                    }
                    let err_2 = (grad_2 - grad_1).norm() / grad.norm();
                    // println!("pc = {pc}, err = {err_2:.2e}");
                    assert!(err_2 < tol);
                } else {
                    assert!(!pc);
                    println!("pc = {pc}, LeastSquaresGradient fails");
                }
            }
        };

        // isotropic
        println!("iso / linear");
        test(30, 1.0, 0, 1, 1e-14);
        test(30, 1.0, 1, 1, 1e-14);
        test(30, 1.0, 2, 1, 1e-14);

        // isotropic + quadratic
        println!("iso / quadratic");
        test(30, 1.0, 0, 2, 1e-14);
        test(30, 1.0, 1, 2, 1e-14);
        test(30, 1.0, 2, 2, 1e-14);

        // anisotropic
        println!("aniso / linear");
        test(30, 1e-6, 0, 1, 1e-10);
        test(30, 1e-6, 1, 1, 1e-10);
        test(30, 1e-6, 2, 1, 1e-10);

        // anisotropic
        println!("aniso / quadratic");
        test(30, 1e-6, 0, 2, 1e-10);
        test(30, 1e-6, 1, 2, 1e-10);
        test(30, 1e-6, 2, 2, 1e-10);
    }
}
