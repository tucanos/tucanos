use nalgebra::{SMatrix, SVector};

/// Solves the Newton system H * p = -g approximately using Truncated Conjugate Gradient.
fn truncated_cg<const N: usize>(
    grad: &SVector<f64, N>,
    hessian: &SMatrix<f64, N, N>,
    max_cg_iter: usize,
) -> SVector<f64, N> {
    let mut p = SVector::<f64, N>::zeros();
    let mut r = -grad; // Initial residual r = b - A*p = -grad - H*0
    let mut d = r;

    let grad_norm = grad.norm();
    // Forcing sequence tolerance forcing an inexact Newton step
    // when far from the solution, saving computational effort.
    let tol = 0.5_f64.min(grad_norm.sqrt()) * grad_norm;

    for _ in 0..max_cg_iter {
        if r.norm() < tol {
            break;
        }

        let hd = hessian * d;
        let curvature = d.dot(&hd);

        // Negative curvature detected (Hessian is not positive definite).
        // If this happens on step 1, fall back to Steepest Descent.
        // Otherwise, truncate and use the accumulated direction.
        if curvature <= 0.0 {
            if p.norm() == 0.0 {
                return -grad;
            }
            return p;
        }

        let r_sq_old = r.dot(&r);
        let alpha = r_sq_old / curvature;

        p += alpha * d;
        r -= alpha * hd;

        let r_sq_new = r.dot(&r);
        let beta = r_sq_new / r_sq_old;
        d = r + beta * d;
    }

    p
}

#[derive(Debug)]
#[allow(dead_code)]
pub enum ConvergenceStatus {
    Converged(usize),

    NotConverged,
}

/// Minimizes an N-dimensional function using the Newton-CG algorithm.
pub fn newton_cg_minimize<const N: usize, F, DF, DDF>(
    x0: SVector<f64, N>,
    f: F,
    df: DF,
    ddf: DDF,
    tol: f64,
    max_iter: usize,
) -> (SVector<f64, N>, ConvergenceStatus)
where
    F: Fn(&SVector<f64, N>) -> f64,
    DF: Fn(&SVector<f64, N>) -> SVector<f64, N>,
    DDF: Fn(&SVector<f64, N>) -> SMatrix<f64, N, N>,
{
    let mut x = x0;
    let n = x.len();
    let max_cg_iter = n; // CG typically converges in N iterations for N variables

    let c1 = 1e-4; // Armijo line search parameter
    let rho = 0.5; // Backtracking step contraction factor
    for iter in 0..max_iter {
        let grad = df(&x);
        if grad.norm() < tol {
            return (x, ConvergenceStatus::Converged(iter));
        }

        let hessian = ddf(&x);

        // 1. Compute approximate search direction via Truncated CG
        let step_dir = truncated_cg(&grad, &hessian, max_cg_iter);
        // let step_dir = -hessian.lu().solve(&grad).unwrap();

        // 2. Backtracking Line Search (Armijo condition)
        let f_val = f(&x);
        let grad_dot_dir = grad.dot(&step_dir);
        let mut alpha = 1.0;

        #[allow(clippy::while_float)]
        while f(&(x + alpha * step_dir)) > f_val + c1 * alpha * grad_dot_dir {
            alpha *= rho;
            if alpha < 1e-8 {
                // Line search failed to find sufficient decrease
                break;
            }
        }

        let step = alpha * step_dir;
        x += &step;

        // Convergence check
        if step.norm() < tol {
            return (x, ConvergenceStatus::Converged(iter + 1000));
        }
    }

    (x, ConvergenceStatus::NotConverged)
}
