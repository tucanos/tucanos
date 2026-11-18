#[cfg(feature = "fast-math-approx")]
const LN2: f64 = std::f64::consts::LN_2;
#[cfg(feature = "fast-math-approx")]
const INV_LN2: f64 = std::f64::consts::LOG2_E;

#[inline]
#[must_use]
pub fn fast_ln(x: f64) -> f64 {
    #[cfg(feature = "fast-math-approx")]
    {
        if x.is_nan() || x < 0.0 {
            return f64::NAN;
        }
        if x == 0.0 {
            return f64::NEG_INFINITY;
        }
        if x.is_infinite() {
            return f64::INFINITY;
        }

        let bits = x.to_bits();
        let mut e = ((bits >> 52) & 0x7ff) as i32 - 1023;
        let mant_bits = (bits & 0x000f_ffff_ffff_ffff) | (1023_u64 << 52);
        let mut m = f64::from_bits(mant_bits);

        // Keep mantissa close to 1 so the odd polynomial in y converges quickly.
        if m < std::f64::consts::FRAC_1_SQRT_2 {
            m *= 2.0;
            e -= 1;
        }

        let y = (m - 1.0) / (m + 1.0);
        let y2 = y * y;
        let p = 2.0
            + y2 * (0.666_666_666_666_666_6
                + y2 * (0.4 + y2 * (0.285_714_285_714_285_7 + y2 * 0.222_222_222_222_222_2)));

        f64::from(e) * LN2 + y * p
    }
    #[cfg(not(feature = "fast-math-approx"))]
    {
        libm::log(x)
    }
}

#[inline]
#[must_use]
pub fn fast_exp(x: f64) -> f64 {
    #[cfg(feature = "fast-math-approx")]
    {
        if x.is_nan() {
            return f64::NAN;
        }
        if x > 709.0 {
            return f64::INFINITY;
        }
        if x < -745.0 {
            return 0.0;
        }

        let n = (x * INV_LN2).round() as i32;
        let r = x - f64::from(n) * LN2;

        let r2 = r * r;
        let r3 = r2 * r;
        let r4 = r3 * r;
        let r5 = r4 * r;
        let r6 = r5 * r;

        let er = 1.0
            + r
            + 0.5 * r2
            + 0.166_666_666_666_666_66 * r3
            + 0.041_666_666_666_666_664 * r4
            + 0.008_333_333_333_333_333 * r5
            + 0.001_388_888_888_888_889 * r6;

        let k = n + 1023;
        if k <= 0 {
            return 0.0;
        }
        if k >= 0x7ff {
            return f64::INFINITY;
        }

        let scale = f64::from_bits((k as u64) << 52);
        er * scale
    }
    #[cfg(not(feature = "fast-math-approx"))]
    {
        libm::exp(x)
    }
}
