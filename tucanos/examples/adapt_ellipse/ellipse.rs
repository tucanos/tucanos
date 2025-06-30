pub struct EllipseProjection {
    a: f64,
    b: f64,
}

impl EllipseProjection {
    const TOL: f64 = 1e-12;

    pub const fn new(a: f64, b: f64) -> Self {
        Self { a, b }
    }

    fn f(&self, p: f64, x: f64, y: f64) -> f64 {
        2.0 * self.a * (x - self.a * p.cos()) * p.sin()
            - 2.0 * self.b * (y - self.b * p.sin()) * p.cos()
    }

    fn solve(&self, x: f64, y: f64) -> f64 {
        let (mut start, mut end) = (0.0, 0.5 * std::f64::consts::PI);
        let (mut f_start, mut f_end) = (self.f(start, x, y), self.f(end, x, y));

        #[allow(clippy::while_float)]
        while (end - start).abs() > Self::TOL {
            let mid = 0.5 * (start + end);
            let f_mid = self.f(mid, x, y);
            if f_mid * f_start > 0.0 {
                start = mid;
                f_start = f_mid;
            } else if f_mid * f_end > 0.0 {
                end = mid;
                f_end = f_mid;
            } else if f_mid.abs() < f64::EPSILON {
                break;
            } else {
                unreachable!("{x} {y} {start} {f_start} {end} {f_end} {f_mid}");
            }
        }
        0.5 * (start + end)
    }

    pub fn project(&self, x: f64, y: f64) -> (f64, f64) {
        let sgn_x = x.signum();
        let x = x.abs();
        let sgn_y = y.signum();
        let y = y.abs();

        let t = self.solve(x, y);
        (self.a * t.cos() * sgn_x, self.b * t.sin() * sgn_y)
    }

    pub fn normal(&self, x: f64, y: f64) -> (f64, f64) {
        let sgn_x = x.signum();
        let x = x.abs();
        let sgn_y = y.signum();
        let y = y.abs();

        let t = self.solve(x, y);
        let x1 = self.a * t.cos() * sgn_x;
        let y1 = self.b * t.sin() * sgn_y;

        let nx = 1. / self.a.powi(2) * x1;
        let ny = 1. / self.b.powi(2) * y1;
        let n = nx.hypot(ny);
        (nx / n, ny / n)
    }

    pub fn is_in(&self, x: f64, y: f64) -> bool {
        (x * x) / (self.a * self.a) + (y * y) / (self.b * self.b) < 1.0
    }
}
