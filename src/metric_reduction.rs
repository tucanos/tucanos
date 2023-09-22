use nalgebra::allocator::Allocator;
use nalgebra::{Const, DefaultAllocator, SMatrix};

/// Compute the simulaneous reduction of two metrics.
/// - It is computed as follows:
///   - Compute the EVD of $\mathcal M = \mathcal M_2^{-1/2} \mathcal M_1 \mathcal M_2^{-1/2} = P^T\Lambda P$
///   - Compute $M = \mathcal M_2^{-1/2} P = (e_0 | ... | e_d)$
///   - Compute $s_i = mav(e_i^T\mathcal M_1 e, e_i^T\mathcal M_2 e)$
///   - $\mathcal M_{1 \cap 2} = \mathcal M_2^{1/2} M diag(s) M^T \mathcal M_2^{1/2}$
/// $\mathcal M_{1 \cap 2}$ and $\det(\mathcal M_{1 \cap 2})$ are returned
pub fn simultaneous_reduction<const D: usize>(
    a: SMatrix<f64, D, D>,
    b: SMatrix<f64, D, D>,
) -> (SMatrix<f64, D, D>, f64)
where
    Const<D>: nalgebra::DimName + nalgebra::ToTypenum + nalgebra::DimSub<nalgebra::U1>,
    DefaultAllocator: Allocator<f64, Const<D>>
        + Allocator<f64, <Const<D> as nalgebra::DimSub<nalgebra::U1>>::Output>,
{
    let mut eig = b.symmetric_eigen();
    let mut det = eig.eigenvalues.iter().product::<f64>();
    eig.eigenvalues.iter_mut().for_each(|s| *s = 1.0 / s.sqrt());
    let tmp = eig.recompose();

    let p = tmp * a * tmp;
    let p = p.symmetric_eigen().eigenvectors;
    let tmp = tmp * p;

    let mut s = SMatrix::<f64, D, D>::zeros();
    let (n, _) = a.shape();
    for i in 0..n {
        let v = tmp.column(i);
        s[n * i + i] = f64::max(v.dot(&(a * v)), v.dot(&(b * v)));
        det *= f64::max(v.dot(&(a * v)), v.dot(&(b * v)));
    }

    eig.eigenvalues.iter_mut().for_each(|s| *s = 1.0 / *s);
    let tmp = eig.recompose();
    let tmp = p.transpose() * tmp;

    (tmp.transpose() * s * tmp, det)
}

/// Compute the step between two metrics $`\mathcal M_1`$ and $`\mathcal M_2`$, i.e.
/// the min and max of
/// ```math
/// \frac{\sqrt{e^T \mathcal M_2 e}}{\sqrt{e^T \mathcal M_1 e}}
/// ```
/// over all edges $`e`$.
pub fn step<const D: usize>(mat_a: SMatrix<f64, D, D>, mat_b: SMatrix<f64, D, D>) -> (f64, f64)
where
    Const<D>: nalgebra::ToTypenum,
    Const<D>: nalgebra::DimSub<nalgebra::U1>,
    DefaultAllocator: Allocator<f64, <Const<D> as nalgebra::DimSub<nalgebra::U1>>::Output>,
{
    let mut eig = mat_a.symmetric_eigen();
    eig.eigenvalues.iter_mut().for_each(|x| *x = 1.0 / x.sqrt());
    let tmp = eig.recompose();

    let s = tmp * mat_b * tmp;
    let eig2 = s.symmetric_eigen();
    (eig2.eigenvalues.min().sqrt(), eig2.eigenvalues.max().sqrt())
}

/// Bound the step between two metrics $`\mathcal M_1`$ and $`\mathcal M_2`$.
///
/// The goal is to find $`\mathcal M`$ as close as possible from $`\mathcal M_2`$ such that
/// ```math
/// 1/f \le \frac{\sqrt{e^T \mathcal M e}}{\sqrt{e^T \mathcal M_1 e}} \le f
/// ```
/// for any edge $`e`$.
///
/// Measuring the distance as the Froebenius norm $\|mathcal M_0^{-1/2} (\mathcal M - \mathcal M_2) \mathcal M_0^{-1/2}\|_F$,
/// the optimal $\mathcal M$ is computed as follows
/// - compute $`\mathcal N_1 := \mathcal M_0^{-1/2} \mathcal M_1 \mathcal M_0^{-1/2}`$,
/// - compute the eigenvalue decomposition $`Q D Q^T = \mathcal N_1`$,
/// - compute $`\mathcal N^\ast := Q \mathrm{diag}(\hat\lambda_i) Q^T`$ where $`\hat\lambda_i := \min\big(\max\big(\lambda_i, 1/f \big), f \big)`$,
/// - compute $`\mathcal M^\ast := \mathcal M_0^{1/2} \mathcal N^\ast \mathcal M_0^{1/2}`$.
///
pub fn control_step<const D: usize>(
    mat_a: SMatrix<f64, D, D>,
    mat_b: SMatrix<f64, D, D>,
    f: f64,
) -> Option<SMatrix<f64, D, D>>
where
    Const<D>: nalgebra::ToTypenum,
    Const<D>: nalgebra::DimSub<nalgebra::U1>,
    DefaultAllocator: Allocator<f64, <Const<D> as nalgebra::DimSub<nalgebra::U1>>::Output>,
{
    let f = f * f;

    let mut eig = mat_a.symmetric_eigen();
    eig.eigenvalues.iter_mut().for_each(|x| *x = 1.0 / x.sqrt());
    let tmp = eig.recompose();

    let s = tmp * mat_b * tmp;
    let mut eig2 = s.symmetric_eigen();
    if eig2.eigenvalues.iter().all(|&x| x > 1.0 / f && x < f) {
        return None;
    }
    eig2.eigenvalues
        .iter_mut()
        .for_each(|x| *x = x.min(f).max(1.0 / f));
    let s = eig2.recompose();

    eig.eigenvalues.iter_mut().for_each(|x| *x = 1.0 / *x);
    let tmp = eig.recompose();
    Some(tmp * s * tmp)
}

#[cfg(test)]
mod tests {
    use super::{control_step, simultaneous_reduction};
    use nalgebra::{SMatrix, SVector};

    #[test]
    fn test_simred_2d() {
        let mat_a = SMatrix::<f64, 2, 2>::new(0.05406506, -0.05927677, -0.05927677, 0.0803954);
        let mat_b = SMatrix::<f64, 2, 2>::new(0.20212283, 0.12064306, 0.12064306, 0.07613422);
        let mat_c_ref = SMatrix::<f64, 2, 2>::new(0.25017324, 0.05992738, 0.05992738, 0.15285352);
        let (mat_c, _) = simultaneous_reduction(mat_a, mat_b);

        assert!((mat_c - mat_c_ref).norm() < mat_c.norm() * 1e-6);

        let eps = 1e-8;

        for _ in 0..100 {
            let mat_r = SMatrix::<f64, 2, 2>::new_random();
            let mat_a = mat_r.transpose() * mat_r;
            let mat_r = SMatrix::<f64, 2, 2>::new_random();
            let mat_b = mat_r.transpose() * mat_r;

            let (mat_c, _) = simultaneous_reduction(mat_a, mat_b);

            for _ in 0..100 {
                let v = SVector::<f64, 2>::new_random();
                let v = v.normalize();
                let la2 = v.dot(&(mat_a * v));
                let lb2 = v.dot(&(mat_b * v));
                let lc2 = v.dot(&(mat_c * v));
                assert!(la2 > 0.0);
                assert!(lb2 > 0.0);
                assert!(lc2 > (1.0 - eps) * la2);
                assert!(lc2 > (1.0 - eps) * lb2);
            }
        }
    }

    #[test]
    fn test_simred_3d() {
        let eps = 1e-8;

        for _ in 0..100 {
            let mat_r = SMatrix::<f64, 3, 3>::new_random();
            let mat_a = mat_r.transpose() * mat_r;
            let mat_r = SMatrix::<f64, 3, 3>::new_random();
            let mat_b = mat_r.transpose() * mat_r;

            let (mat_c, _) = simultaneous_reduction(mat_a, mat_b);

            for _ in 0..100 {
                let v = SVector::<f64, 3>::new_random();
                let v = v.normalize();
                let la2 = v.dot(&(mat_a * v));
                let lb2 = v.dot(&(mat_b * v));
                let lc2 = v.dot(&(mat_c * v));
                assert!(la2 > 0.0);
                assert!(lb2 > 0.0);
                assert!(lc2 > (1.0 - eps) * la2);
                assert!(lc2 > (1.0 - eps) * lb2);
            }
        }
    }

    #[test]
    fn test_step_2d() {
        let eps = 1e-8;
        let f = 2.0;

        for _ in 0..100 {
            let mat_r = SMatrix::<f64, 2, 2>::new_random();
            let mat_a = mat_r.transpose() * mat_r;
            let mat_r = SMatrix::<f64, 2, 2>::new_random();
            let mat_b = mat_r.transpose() * mat_r;

            let mat_c = control_step(mat_a, mat_b, f).unwrap_or(mat_b);
            for _ in 0..100 {
                let v = SVector::<f64, 2>::new_random();
                let v = v.normalize();
                let la2 = v.dot(&(mat_a * v));
                let lc2 = v.dot(&(mat_c * v));
                assert!(f64::sqrt(lc2 / la2) > 1. / f - eps);
                assert!(f64::sqrt(la2 / lc2) < f + eps);
            }
        }
    }

    #[test]
    fn test_step_3d() {
        let eps = 1e-8;
        let f = 2.0;

        for _ in 0..100 {
            let mat_r = SMatrix::<f64, 3, 3>::new_random();
            let mat_a = mat_r.transpose() * mat_r;
            let mat_r = SMatrix::<f64, 3, 3>::new_random();
            let mat_b = mat_r.transpose() * mat_r;

            let mat_c = control_step(mat_a, mat_b, f).unwrap_or(mat_b);
            for _ in 0..100 {
                let v = SVector::<f64, 3>::new_random();
                let v = v.normalize();
                let la2 = v.dot(&(mat_a * v));
                let lc2 = v.dot(&(mat_c * v));
                assert!(f64::sqrt(lc2 / la2) > 1. / f - eps);
                assert!(f64::sqrt(la2 / lc2) < f + eps);
            }
        }
    }
}
