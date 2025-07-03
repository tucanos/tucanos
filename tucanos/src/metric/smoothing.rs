use crate::{
    Result,
    mesh::{Elem, SimplexMesh},
    metric::Metric,
};
use log::debug;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

impl<const D: usize, E: Elem> SimplexMesh<D, E> {
    /// Smooth a metric field to avoid numerical artifacts
    /// For each mesh vertex $`i`$, a set a suitable neighbors $`N(i)`$ is built as
    /// a subset of the neighbors of $`i`$ ($`i`$ is included) ignoring the vertices with the metrics with
    /// the smallest and largest metric volume.
    /// The smoothed metric field is then computed as the average (i.e. interpolation
    /// with equal weights) of the metrics in $`N(i)`$
    /// , the metric is replaced by the average
    /// on its neighbors ignoring the metrics with the minimum and maximum volumes
    /// TODO: doc
    pub fn smooth_metric<M: Metric<D>>(&self, m: &[M]) -> Result<Vec<M>> {
        debug!("Apply metric smoothing");
        let n = self.n_verts() as usize;
        assert_eq!(m.len(), n);

        let v2v = self.get_vertex_to_vertices()?;

        let mut res = vec![M::default(); n];

        res.par_iter_mut()
            .enumerate()
            .for_each(|(i_vert, m_smooth)| {
                let m_v = &m[i_vert];
                let vol = m_v.vol();
                let mut min_vol = vol;
                let mut max_vol = vol;
                let mut min_idx = 0;
                let mut max_idx = 0;
                let neighbors = v2v.row(i_vert);
                for i_neigh in neighbors {
                    let m_n = &m[*i_neigh];
                    let vol = m_n.vol();
                    if vol < min_vol {
                        min_vol = vol;
                        min_idx = i_neigh + 1;
                    } else if vol > max_vol {
                        max_vol = vol;
                        max_idx = i_neigh + 1;
                    }
                }

                let mut weights = Vec::new();
                let mut metrics = Vec::new();

                let n = if min_idx == max_idx {
                    neighbors.len()
                } else {
                    neighbors.len() - 1
                };
                let w = 1. / n as f64;

                if min_idx != 0 && max_idx != 0 {
                    weights.push(w);
                    metrics.push(&m[i_vert]);
                }
                for i_neigh in neighbors {
                    if min_idx != i_neigh + 1 && max_idx != i_neigh + 1 {
                        weights.push(w);
                        metrics.push(&m[*i_neigh]);
                    }
                }

                *m_smooth = M::interpolate(weights.iter().copied().zip(metrics.iter().copied()));
            });

        Ok(res)
    }
}

#[cfg(test)]
mod tests {
    use tmesh::mesh::Mesh;

    use crate::{
        mesh::Point,
        mesh::test_meshes::{test_mesh_2d, test_mesh_3d},
        metric::{AnisoMetric2d, AnisoMetric3d, IsoMetric, Metric},
        min_iter,
    };

    #[test]
    fn test_smooth_2d() {
        let mut mesh = test_mesh_2d().split().split();
        mesh.compute_vertex_to_vertices();

        let mut m: Vec<_> = (0..mesh.n_verts())
            .map(|_| IsoMetric::<2>::from(0.1))
            .collect();

        m[2] = IsoMetric::<2>::from(0.01);
        m[5] = IsoMetric::<2>::from(1.);

        let m = mesh.smooth_metric(&m).unwrap();

        let vmin = min_iter(m.iter().map(Metric::vol));
        let vmax = min_iter(m.iter().map(Metric::vol));

        assert!(f64::abs(vmin - 0.01) < 1e-6);
        assert!(f64::abs(vmax - 0.01) < 1e-6);
    }

    #[test]
    fn test_smooth_2d_aniso() {
        let mut mesh = test_mesh_2d().split().split();
        mesh.compute_volumes();

        mesh.compute_vertex_to_vertices();

        let v0 = Point::<2>::new(0.5, 0.);
        let v1 = Point::<2>::new(0.0, 4.0);

        let mut m: Vec<_> = (0..mesh.n_verts())
            .map(|_| AnisoMetric2d::from_sizes(&v0, &v1))
            .collect();

        let v0 = Point::<2>::new(0.05, 0.);
        let v1 = Point::<2>::new(0.0, 4.0);
        m[2] = AnisoMetric2d::from_sizes(&v0, &v1);

        let v0 = Point::<2>::new(0.5, 0.);
        let v1 = Point::<2>::new(0.0, 40.0);
        m[5] = AnisoMetric2d::from_sizes(&v0, &v1);

        let m = mesh.smooth_metric(&m).unwrap();

        let vmin = min_iter(m.iter().map(Metric::vol));
        let vmax = min_iter(m.iter().map(Metric::vol));

        assert!(f64::abs(vmin - 2.0) < 1e-6);
        assert!(f64::abs(vmax - 2.0) < 1e-6);
    }

    #[test]
    fn test_smooth_3d() {
        let mut mesh = test_mesh_3d().split().split();
        mesh.compute_vertex_to_vertices();

        let mut m: Vec<_> = (0..mesh.n_verts())
            .map(|_| IsoMetric::<3>::from(0.1))
            .collect();

        m[2] = IsoMetric::<3>::from(0.01);
        m[5] = IsoMetric::<3>::from(1.);

        let m = mesh.smooth_metric(&m).unwrap();

        let vmin = min_iter(m.iter().map(Metric::vol));
        let vmax = min_iter(m.iter().map(Metric::vol));

        assert!(f64::abs(vmin - 0.001) < 1e-6);
        assert!(f64::abs(vmax - 0.001) < 1e-6);
    }

    #[test]
    fn test_smooth_3d_aniso() {
        let mut mesh = test_mesh_3d().split().split();
        mesh.compute_volumes();

        mesh.compute_vertex_to_vertices();

        let v0 = Point::<3>::new(0.5, 0.0, 0.0);
        let v1 = Point::<3>::new(0.0, 4.0, 0.0);
        let v2 = Point::<3>::new(0.0, 0.0, 0.1);

        let mut m: Vec<_> = (0..mesh.n_verts())
            .map(|_| AnisoMetric3d::from_sizes(&v0, &v1, &v2))
            .collect();

        let v0 = Point::<3>::new(0.05, 0.0, 0.0);
        let v1 = Point::<3>::new(0.0, 4.0, 0.0);
        let v2 = Point::<3>::new(0.0, 0.0, 0.1);
        m[2] = AnisoMetric3d::from_sizes(&v0, &v1, &v2);

        let v0 = Point::<3>::new(0.5, 0.0, 0.0);
        let v1 = Point::<3>::new(0.0, 4.0, 0.0);
        let v2 = Point::<3>::new(0.0, 0.0, 1.0);
        m[5] = AnisoMetric3d::from_sizes(&v0, &v1, &v2);

        let m = mesh.smooth_metric(&m).unwrap();

        let vmin = min_iter(m.iter().map(Metric::vol));
        let vmax = min_iter(m.iter().map(Metric::vol));

        assert!(f64::abs(vmin - 0.2) < 1e-6);
        assert!(f64::abs(vmax - 0.2) < 1e-6);
    }
}
