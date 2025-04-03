use crate::{
    mesh::{Elem, Point, SimplexMesh},
    metric::Metric,
    Idx, Result,
};
use log::{debug, warn};
use nalgebra::{allocator::Allocator, Const, DefaultAllocator};
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};

impl<const D: usize, E: Elem> SimplexMesh<D, E> {
    /// Compute the gradation on an edge
    fn edge_gradation<M: Metric<D>>(m0: &M, m1: &M, e: &Point<D>) -> f64 {
        let l0 = m0.length(e);
        let l1 = m1.length(e);
        let a = l0 / l1;
        let l = if f64::abs(a - 1.0) < 1e-3 {
            l0
        } else {
            l0 * f64::ln(a) / (a - 1.0)
        };
        f64::max(a, 1.0 / a).powf(1. / l)
    }

    /// Compute the maximum metric gradation and the fraction of edges with a gradation
    /// higher than an threshold
    pub fn gradation<M: Metric<D>>(&self, m: &[M], target: f64) -> Result<(f64, f64)> {
        let edges = self.get_edges()?;

        let (count, max_gradation) = edges
            .par_iter()
            .map(|&e| {
                let m0 = &m[e[0] as usize];
                let m1 = &m[e[1] as usize];
                let e = self.vert(e[1]) - self.vert(e[0]);
                Self::edge_gradation(m0, m1, &e)
            })
            .fold(
                || (0_usize, 0.0_f64),
                |mut a, b| {
                    if b > target {
                        a.0 += 1;
                    }
                    a.1 = a.1.max(b);
                    a
                },
            )
            .reduce(|| (0_usize, 0.0), |a, b| (a.0 + b.0, a.1.max(b.1)));

        Ok((max_gradation, count as f64 / edges.len() as f64))
    }

    /// Enforce a maximum gradiation on a metric field
    /// Algorithm taken from "Size gradation control of anisotropic meshes", F. Alauzet, 2010 and
    /// "Feature-based and goal-oriented anisotropic mesh adaptation for RANS
    /// applications in aeronautics and aerospace", F. Alauzet & L. Frazza, 2021
    ///
    /// and modified for parallel implementation
    pub fn apply_metric_gradation<M: Metric<D>>(
        &self,
        m: &mut [M],
        beta: f64,
        t: f64,
        max_iter: Idx,
    ) -> Result<Idx> {
        debug!(
            "Apply metric gradation (beta = {beta}, max_iter = {max_iter})"
        );

        let v2v = self.get_vertex_to_vertices()?;

        let mut n = 0;
        for _ in 0..max_iter {
            let tmp = m.to_vec();

            n = m
                .par_iter_mut()
                .enumerate()
                .map(|(i_vert, m_new)| {
                    let neighbors = v2v.row(i_vert as Idx);
                    let v0 = self.vert(i_vert as Idx);
                    let mut fixed = false;
                    for &i_neigh in neighbors {
                        let e = self.vert(i_neigh) - v0;
                        let m = &tmp[i_neigh as usize];
                        let g = Self::edge_gradation(m_new, m, &e);
                        if g < 1.01 * beta {
                            continue;
                        }
                        fixed = true;
                        let m_spanned = m.span(&e, beta, t);
                        *m_new = m_new.intersect(&m_spanned);
                    }
                    fixed
                })
                .filter(|&x| x)
                .count();

            if n == 0 {
                break;
            }
        }

        if n > 0 {
            let (c_max, frac_large_gradation) = self.gradation(m, beta)?;
            warn!(
                "gradation: target not achieved: max gradation: {:.2}, {:.2e}% of edges have a gradation > {}",
                c_max,
                frac_large_gradation * 100.0,
                beta
            );
        }

        Ok(n as Idx)
    }

    /// Extend a metric defined on some of the vertices to the whole domain assuming a
    /// gradation
    /// - metric: the metric field at every vertex
    /// - flg: a flag defined at every vertex indicating whether the metric is to be used at
    ///   that vertex
    /// - beta: the mesh gradation
    pub fn extend_metric<M: Metric<D>>(
        &self,
        metric: &mut [M],
        flg: &mut [bool],
        beta: f64,
        t: f64,
    ) -> Result<()>
    where
        Const<D>: nalgebra::ToTypenum + nalgebra::DimSub<nalgebra::U1>,
        DefaultAllocator: Allocator<<Const<D> as nalgebra::DimSub<nalgebra::U1>>::Output>,
    {
        debug!("Extend the metric into the domain using gradation = {beta}");

        let n_verts = self.n_verts() as usize;

        let mut to_fix = flg.iter().filter(|&&x| !x).count();
        debug!("{to_fix} / {n_verts} internal vertices to fix");

        let v2v = self.get_vertex_to_vertices()?;

        let mut n_iter = 0;
        loop {
            let mut fixed = vec![false; n_verts];
            let mut tmp = vec![M::default(); n_verts];
            fixed
                .par_iter_mut()
                .zip(tmp.par_iter_mut())
                .enumerate()
                .for_each(|(i_vert, (is_fixed, m_new))| {
                    if flg[i_vert] {
                        return;
                    }
                    let pt = self.vert(i_vert as Idx);
                    let neighbors = v2v.row(i_vert as Idx);
                    let mut valid_neighbors =
                        neighbors.iter().copied().filter(|&i| flg[i as usize]);
                    if let Some(i) = valid_neighbors.next() {
                        let m_i = metric[i as usize];
                        let pt_i = self.vert(i);
                        let e = pt - pt_i;
                        *m_new = m_i.span(&e, beta, t);
                        for i in valid_neighbors {
                            let m_i = metric[i as usize];
                            let pt_i = self.vert(i);
                            let e = pt - pt_i;
                            let m_i_spanned = m_i.span(&e, beta, t);
                            *m_new = m_new.intersect(&m_i_spanned);
                        }
                        *is_fixed = true;
                    }
                });

            flg.par_iter_mut()
                .zip(metric.par_iter_mut())
                .zip(fixed.par_iter().zip(tmp.par_iter()))
                .for_each(|((f, m), (is_fixed, m_new))| {
                    if *is_fixed {
                        *f = true;
                        *m = *m_new;
                    }
                });
            to_fix = flg.par_iter().filter(|&&x| !x).count();
            if to_fix == 0 {
                break;
            } else if !fixed.par_iter().copied().any(|x| x) {
                // No element was fixed
                warn!(
                    "stop at iteration {}, {} elements cannot be fixed",
                    n_iter + 1,
                    to_fix
                );
                break;
            }
            n_iter += 1;

            debug!(
                "iteration {n_iter}: {to_fix} / {n_verts} vertices remain to be fixed"
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        mesh::{
            test_meshes::{test_mesh_2d, test_mesh_3d},
            Point, SimplexMesh, Tetrahedron,
        },
        metric::{AnisoMetric, AnisoMetric3d, IsoMetric, Metric},
        Idx, Result,
    };
    use nalgebra::SMatrix;
    use rand::{rngs::StdRng, Rng, SeedableRng};
    #[test]
    fn test_gradation_iso() {
        let beta = 1.5;
        let e = Point::<3>::new(1.0, 0.0, 0.0);
        let m0 = IsoMetric::<3>::from(0.01);
        for h in [0.02, 0.1, 1.0, 10.0, 100.0] {
            let m1 = IsoMetric::<3>::from(h);
            let m = m0.span(&e, beta, 1.0);
            let m1 = m1.intersect(&m);
            let g = SimplexMesh::<3, Tetrahedron>::edge_gradation(&m0, &m1, &e);
            assert!(g < 1.01 * beta, "{g} > {beta}");
        }

        for h in [0.02, 0.1, 1.0, 10.0, 100.0] {
            let m1 = IsoMetric::<3>::from(h);
            let g = SimplexMesh::<3, Tetrahedron>::edge_gradation(&m0, &m1, &e);
            let beta = (0.5 * g).max(1.1);
            let m = m0.span(&e, beta, 1.0);
            let m1 = m1.intersect(&m);
            let g = SimplexMesh::<3, Tetrahedron>::edge_gradation(&m0, &m1, &e);
            assert!(g < 1.01 * beta, "{g} > {beta}");
        }
    }

    #[test]
    fn test_gradation_aniso() {
        let mut rng = StdRng::seed_from_u64(0);

        let e = Point::<3>::new(1.0, 0.0, 0.0);

        let beta = 1.5;
        let t = 0.125;
        for _ in 0..100 {
            let mat = SMatrix::<f64, 3, 3>::from_fn(|_, _| rng.random());
            let mat = mat.transpose() * mat;
            let m0 = AnisoMetric3d::from_mat(mat);

            let mat = 0.1 * SMatrix::<f64, 3, 3>::from_fn(|_, _| rng.random());
            let mat = mat.transpose() * mat;
            let m1 = AnisoMetric3d::from_mat(mat);
            let before = SimplexMesh::<3, Tetrahedron>::edge_gradation(&m0, &m1, &e);
            let m = m0.span(&e, beta, t);
            let m1 = m1.intersect(&m);
            let after = SimplexMesh::<3, Tetrahedron>::edge_gradation(&m0, &m1, &e);
            // println!("{before} {after} {}", after < before);
            assert!(after < before, "{after} > {before}");
        }

        let t = 0.0;
        for _ in 0..100 {
            let mat = SMatrix::<f64, 3, 3>::from_fn(|_, _| rng.random());
            let mat = mat.transpose() * mat;
            let m0 = AnisoMetric3d::from_mat(mat);

            let mat = 0.1 * SMatrix::<f64, 3, 3>::from_fn(|_, _| rng.random());
            let mat = mat.transpose() * mat;
            let m1 = AnisoMetric3d::from_mat(mat);
            let m = m0.span(&e, beta, t);
            let m1 = m1.intersect(&m);
            let after = SimplexMesh::<3, Tetrahedron>::edge_gradation(&m0, &m1, &e);
            assert!(after < 1.01 * beta, "{after} > {beta}");
        }
    }

    #[test]
    fn test_gradation_2d() {
        let mut mesh = test_mesh_2d().split();
        mesh.compute_edges();

        let mut m: Vec<_> = (0..mesh.n_verts())
            .map(|_| IsoMetric::<2>::from(0.1))
            .collect();
        m[0] = IsoMetric::<2>::from(0.0001);

        let beta = 1.2;
        let t = 1.0;
        let (c_max, frac_large_c) = mesh.gradation(&m, beta).unwrap();
        assert!(c_max > beta);
        assert!(frac_large_c > 0.0);

        mesh.compute_vertex_to_vertices();
        let n = mesh.apply_metric_gradation(&mut m, beta, t, 10).unwrap();
        assert_eq!(n, 0);

        let (c_max, frac_large_c) = mesh.gradation(&m, beta).unwrap();
        assert!(c_max < 1.02 * beta);
        assert!(frac_large_c < 1e-12);

        let edges = mesh.get_edges().unwrap();
        for &e in edges {
            let i0 = e[0] as usize;
            let i1 = e[1] as usize;

            let e = mesh.vert(i1 as Idx) - mesh.vert(i0 as Idx);

            let l = m[i0].length(&e);
            let rmax = 1.0 + l * f64::ln(beta);
            assert!(m[i1].h() < 1.0001 * m[i0].h() * rmax);

            let l = m[i1].length(&e);
            let rmax = 1.0 + l * f64::ln(beta);
            assert!(m[i0].h() < 1.0001 * m[i1].h() * rmax);
        }
    }

    #[test]
    fn test_extend() -> Result<()> {
        let mut mesh = test_mesh_3d().split().split().split();

        let mut m = vec![IsoMetric::<3>::default(); mesh.n_verts() as usize];
        let mut flg = vec![false; mesh.n_verts() as usize];

        for (f, t) in mesh.faces().zip(mesh.ftags()) {
            if t == 1 {
                for i in f {
                    flg[i as usize] = true;
                    m[i as usize] = IsoMetric::<3>::from(0.01);
                }
            }
        }

        mesh.compute_vertex_to_vertices();

        mesh.extend_metric(&mut m, &mut flg, 1.0, 1.0)?;

        for x in m {
            assert!(f64::abs(x.h() - 0.01) < 1e-6);
        }

        Ok(())
    }
}
