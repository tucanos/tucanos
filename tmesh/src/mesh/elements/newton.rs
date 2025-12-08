use nalgebra::{SMatrix, SVector};

pub enum NewtonConvergenceStatus<const D: usize> {
    Converged(SVector<f64, D>),
    MaxItersReached(SVector<f64, D>),
    HessianNonInvertible,
}
/// Solves a multivariate minimization problem using the Newton method.
///
/// This function handles N-dimensional problems using nalgebra's DVector for the
/// state and gradient, and DMatrix for the Hessian.
///
/// # Arguments
/// * `initial_guess` - The starting point for the optimization (DVector of size N).
/// * `grad_fn` - The function that computes the gradient (DVector).
/// * `hess_fn` - The function that computes the Hessian matrix (DMatrix, NxN).
/// * `max_iters` - Maximum number of iterations.
/// * `tolerance` - Stop when the step size magnitude is less than this value.
///
/// # Returns
/// A Result containing the optimized DVector (the minimum x) or an error message.
pub fn newton_minimization<const D: usize>(
    initial_guess: SVector<f64, D>,
    obj_fn: impl Fn(&SVector<f64, D>) -> f64,
    grad_fn: impl Fn(&SVector<f64, D>) -> SVector<f64, D>,
    hess_fn: impl Fn(&SVector<f64, D>) -> SMatrix<f64, D, D>,
    max_iters: usize,
    tolerance: f64,
) -> NewtonConvergenceStatus<D> {
    let mut x_curr = initial_guess;

    for i in 0..max_iters {
        // 1. Compute Gradient and Hessian at current point
        let grad = grad_fn(&x_curr);
        let hess = hess_fn(&x_curr);
        println!(
            "Iter {}: x = ({x_curr:?}), f = {:.2e}",
            i + 1,
            obj_fn(&x_curr),
        );

        // Check if the gradient is near zero (already converged)
        if grad.norm() < tolerance {
            println!("Converged successfully (small gradient norm) in {i} iterations.");
            return NewtonConvergenceStatus::Converged(x_curr);
        }

        // 2. Compute the inverse of the Hessian
        // For Newton's method, we often solve H*p = -grad for the step 'p',
        // but for simplicity and smaller matrices, we use the inverse H^(-1).
        let Some(hess_inv) = hess.try_inverse() else {
            println!("Hessian is non-invertible at iteration {i}.");
            // If the Hessian is non-invertible, the method fails
            return NewtonConvergenceStatus::HessianNonInvertible;
        };

        // 3. Calculate the Newton Step: step = H^(-1) * grad
        let step = hess_inv * grad;
        let nrm = step.norm();
        let step = nrm.min(0.1) / nrm * step;

        // 4. Update the position: x_next = x_curr - step
        let x_next = x_curr - step;

        // 5. Check for convergence based on the step size
        let step_size = step.norm();
        if step_size < tolerance {
            println!("Converged successfully (small step size) in {i} iterations.");
            return NewtonConvergenceStatus::Converged(x_next);
        }

        // Update for next iteration
        x_curr = x_next;
    }

    println!("Max number of iterations ({max_iters}) reached");
    NewtonConvergenceStatus::MaxItersReached(x_curr)
}

#[cfg(test)]
mod tests {
    use nalgebra::{SMatrix, SVector};

    use crate::mesh::elements::newton::{NewtonConvergenceStatus, newton_minimization};

    /// The objective function: f(x1, x2) = x1^2 + 2x2^2 + 4x1 - 4x2 + 10.
    /// Minimum is at x = (-2, 1).
    fn example_objective(x: &SVector<f64, 2>) -> f64 {
        let x1 = x[0];
        let x2 = x[1];
        x1.powi(2) + 2.0 * x2.powi(2) + 4.0 * x1 - 4.0 * x2 + 10.0
    }

    /// The gradient of the example function:
    /// grad f = [ 2x1 + 4, 4x2 - 4 ]^T
    fn example_gradient(x: &SVector<f64, 2>) -> SVector<f64, 2> {
        let x1 = x[0];
        let x2 = x[1];
        SVector::from_column_slice(&[2.0 * x1 + 4.0, 4.0 * x2 - 4.0])
    }

    /// The Hessian of the example function (Constant Matrix):
    /// H = [[ 2, 0 ], [ 0, 4 ]]
    fn example_hessian(_x: &SVector<f64, 2>) -> SMatrix<f64, 2, 2> {
        SMatrix::from_column_slice(&[2.0, 0.0, 0.0, 4.0])
    }

    #[test]
    fn test_newton_minimization() {
        let initial_guess = SVector::zeros(); // Start at (0, 0)
        let max_iterations = 10;
        let tolerance = 1e-6;

        let result = newton_minimization(
            initial_guess,
            example_objective,
            example_gradient,
            example_hessian,
            max_iterations,
            tolerance,
        );

        let NewtonConvergenceStatus::Converged(x) = result else {
            panic!("Newton minimization did not converge.");
        };
        let final_val = example_objective(&x);

        assert!((x[0] + 2.0).abs() < 1e-6);
        assert!((x[1] - 1.0).abs() < 1e-6);
        assert!((final_val - 4.0).abs() < 1e-6);
    }
}
