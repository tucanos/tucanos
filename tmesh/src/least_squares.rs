use crate::{Error, Result};
use nalgebra::{Const, DMatrix, DVector, Dyn, OMatrix, SVector, QR};

pub struct LeastSquaresGradient<const D: usize> {
    qr: QR<f64, Dyn, Dyn>,
    r: DMatrix<f64>,
    weights: Vec<f64>,
}

impl<const D: usize> LeastSquaresGradient<D> {
    pub fn new<I: ExactSizeIterator<Item = SVector<f64, D>>>(
        weight_exp: i32,
        dx: I,
    ) -> Result<Self> {
        assert!(D == 2 || D == 3);
        let n_rows = 1 + dx.len();

        let mut mat = DMatrix::zeros(n_rows, D + 1);
        let mut weights = vec![0.0; n_rows - 1];

        let mut w_max = 0.0;
        for (irow, dp) in dx.enumerate() {
            let w = 1.0 / dp.norm().powi(weight_exp);
            weights[irow] = w;
            let irow = irow + 1;
            let mut row = OMatrix::<f64, Const<1>, Dyn>::zeros(D + 1);
            row[0] = w * 1.0;
            for i in 0..D {
                row[i + 1] = w * dp[i];
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
                "Poor conditionning in QR: {:2e}",
                maxd / mind
            )))
        } else {
            Ok(Self { qr, r, weights })
        }
    }

    pub fn gradient<I: ExactSizeIterator<Item = f64>>(&self, df: I) -> SVector<f64, D> {
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

        rhs.fixed_view::<D, 1>(1, 0).into()
    }

    #[allow(dead_code)]
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

#[cfg(test)]
mod tests {
    use super::LeastSquaresGradient;
    use nalgebra::SVector;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    #[test]
    fn test_ls_2d() {
        let n_neighbors = 10;
        let grad = SVector::<f64, 2>::new(1.2, 2.3);

        let mut rng = StdRng::seed_from_u64(1234);

        let dx = (0..n_neighbors)
            .map(|_| SVector::<f64, 2>::from_fn(|_, _| rng.random::<f64>() - 0.5))
            .collect::<Vec<_>>();
        let df = dx.iter().map(|dx| grad.dot(dx)).collect::<Vec<_>>();

        let ls = LeastSquaresGradient::new(2, dx.iter().cloned()).unwrap();
        let grad_1 = ls.gradient(df.iter().cloned());
        assert!((grad - grad_1).norm() < 1e-12);

        let mut grad_2 = SVector::<f64, 2>::zeros();
        for (g, df) in ls.gradient_weights().zip(df) {
            grad_2 += df * g;
        }

        assert!((grad - grad_2).norm() < 1e-12);
    }

    #[test]
    fn test_ls_3d() {
        let n_neighbors = 10;
        let grad = SVector::<f64, 3>::new(1.2, 2.3, 3.4);

        let mut rng = StdRng::seed_from_u64(1234);

        let dx = (0..n_neighbors)
            .map(|_| SVector::<f64, 3>::from_fn(|_, _| rng.random::<f64>() - 0.5))
            .collect::<Vec<_>>();
        let df = dx.iter().map(|dx| grad.dot(dx)).collect::<Vec<_>>();

        let ls = LeastSquaresGradient::new(2, dx.iter().cloned()).unwrap();
        let grad_1 = ls.gradient(df.iter().cloned());
        assert!((grad - grad_1).norm() < 1e-12);

        let mut grad_2 = SVector::<f64, 3>::zeros();
        for (g, df) in ls.gradient_weights().zip(df) {
            grad_2 += df * g;
        }

        assert!((grad - grad_2).norm() < 1e-12);
    }
}
