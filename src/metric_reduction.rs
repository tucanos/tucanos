use crate::linalg::lapack_generalized_symmetric_eigenvalues;
use nalgebra::allocator::Allocator;
use nalgebra::{Const, DefaultAllocator, SMatrix, SVector};

/// Compute the simulaneous reduction of two metrics.
/// Let $`\mathcal P = (e_0 | ... | e_d)`$ be the generalized eigenvectors of $`(\mathcal M_0, \mathcal M_1)`$:
/// ```math
/// \mathcal M_0 \mathcal P = \Lambda \mathcal M_1 \mathcal P
/// ```
/// then
/// ```math
/// \mathcal M_i = \mathcal P^{-1, T} \Lambda^{(i)} \mathcal P^{(-1)}
/// ```
/// with $`\Lambda^{(i)}_{jk} = e_j^T \mathcal M_i e_k \delta_{jk}`$
/// The intersection is then
/// ```math
/// \mathcal M_0 \cap \mathcal M_1 = \mathcal P^{-1, T} \Lambda^{(i,j)} \mathcal P^{(-1)}
/// ```
/// with $`\Lambda^{(i,j)}_{jk} = max(\Lambda^{(i)}_{jk}, \Lambda^{(j)}_{jk})`$.
///
/// NB: the generalized eigenvalue problem is solved using the `dsygv` function in Lapack
pub fn simultaneous_reduction<const D: usize>(
    mat_a: SMatrix<f64, D, D>,
    mat_b: SMatrix<f64, D, D>,
) -> SMatrix<f64, D, D> {
    let mut p = mat_a;
    let mut tmp = mat_b;
    let mut work = [0.0; 9]; // 9 >= 3*D as D <= 3

    lapack_generalized_symmetric_eigenvalues::<D>(p.as_mut_slice(), tmp.as_mut_slice(), &mut work);

    let mut s = SVector::<f64, D>::zeros();

    for j in 0..D {
        let v = p.column(j);
        let sa = v.dot(&(mat_a * v));
        let sb = v.dot(&(mat_b * v));
        s[j] = f64::max(f64::abs(sa), f64::abs(sb));
    }
    let s = SMatrix::<f64, D, D>::from_diagonal(&s);
    let p = p.try_inverse().unwrap();
    p.transpose() * s * p
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

    fn test_simred<const D: usize>() {
        let eps = 1e-8;

        for _ in 0..100 {
            let mat_r = SMatrix::<f64, D, D>::new_random();
            let mat_a = mat_r.transpose() * mat_r;
            let mat_r = SMatrix::<f64, D, D>::new_random();
            let mat_b = mat_r.transpose() * mat_r;

            let mat_c = simultaneous_reduction(mat_a, mat_b);

            for _ in 0..100 {
                let v = SVector::<f64, D>::new_random();
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
    fn test_simred_2d() {
        test_simred::<2>();
    }

    #[test]
    fn test_simred_3d() {
        test_simred::<3>();
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
