use nalgebra::{SMatrix, SVector};

use crate::linalg::lapack_generalized_symmetric_eigenvalues;

/// Compute the simulaneous reduction of two metric tensors
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

#[cfg(test)]
mod tests {
    use nalgebra::{SMatrix, SVector};

    use crate::metric_reduction::simultaneous_reduction;

    #[test]
    fn test_simred_2d() {
        let eps = 1e-8;

        for _ in 0..100 {
            let mat_r = SMatrix::<f64, 2, 2>::new_random();
            let mat_a = mat_r.transpose() * mat_r;
            let mat_r = SMatrix::<f64, 2, 2>::new_random();
            let mat_b = mat_r.transpose() * mat_r;

            let mat_c = simultaneous_reduction(mat_a, mat_b);

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

            let mat_c = simultaneous_reduction(mat_a, mat_b);

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
}
