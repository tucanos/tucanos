#[cfg(any(
    feature = "accelerate",
    feature = "netlib",
    feature = "intel-mkl",
    feature = "openblas"
))]
extern crate lapack_src;

/// Return x so that ||A.x-B|| is minimum, using Lapack QR factorization
/// Return None if the problem is too ill-conditioned
pub fn lapack_qr_least_squares<const N: usize>(
    a: &mut [f64],
    b: &mut [f64],
    work: &mut [f64],
) -> Option<[f64; N]> {
    let mut info = 0;
    let n = N as i32;
    let nrows = b.len();
    let nr32 = nrows as i32;
    let lwork = work.len() as i32;
    let mut tau = [0.; N];
    unsafe {
        // QR factorization
        lapack::dgeqrf(nr32, n, a, nr32, &mut tau, work, lwork, &mut info);
    }
    assert!(info == 0, "lapack::dgeqrf info={info}");
    // check for ill-conditioned system
    let (mind, maxd) = (0..N)
        .map(|i| a[i + nrows * i].abs())
        .map(|x| (x, x))
        .reduce(|(mi1, ma1), (mi2, ma2)| (mi1.min(mi2), ma1.max(ma2)))
        .unwrap();
    if maxd / mind > 1e8 {
        return None;
    }
    unsafe {
        // b <- Q^t.b
        lapack::dormqr(
            b'L', b'T', nr32, 1, n, a, nr32, &tau, b, nr32, work, lwork, &mut info,
        );
    }
    assert!(info == 0, "lapack::dormqr info={info}");
    unsafe {
        // b <- R^-1.b
        lapack::dtrtrs(b'U', b'N', b'N', n, 1, a, nr32, b, nr32, &mut info);
    }
    assert!(info == 0, "lapack::dtrtrs info={info}");
    Some(b[0..N].try_into().unwrap())
}

/// Compute the generalized eigenvalues and eigenvectors for two NxN symmetric matrices using Lapack
/// a is modified and contains the eigenvectors
pub fn lapack_generalized_symmetric_eigenvalues<const N: usize>(
    a: &mut [f64],
    b: &mut [f64],
    work: &mut [f64],
) -> [f64; N] {
    assert_eq!(a.len(), N * N);
    assert_eq!(b.len(), N * N);
    assert!(work.len() >= usize::max(1, 3 * N - 1));

    let itype = [1];
    let mut info = 0;
    let n = N as i32;
    let lwork = work.len() as i32;
    let mut eigenvalues = [0.; N];
    unsafe {
        lapack::dsygv(
            &itype,
            b'V',
            b'L',
            n,
            a,
            n,
            b,
            n,
            eigenvalues.as_mut_slice(),
            work,
            lwork,
            &mut info,
        );
    }
    assert!(info == 0, "lapack::dsygv info={info}");

    eigenvalues
}
