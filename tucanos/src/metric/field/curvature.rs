use crate::{
    Result, Tag,
    geometry::MeshedGeometry,
    metric::{AnisoMetric2d, AnisoMetric3d, MetricField},
};
use log::debug;
use rustc_hash::FxHashSet;
use tmesh::{
    Vert2d,
    graph::CSRGraph,
    mesh::{Edge, GenericMesh, Idx, Mesh, SubMesh, Tetrahedron, Triangle},
};

impl<'a, T: Idx, M: Mesh<3, Tetrahedron<T>>> MetricField<'a, 3, Tetrahedron<T>, M, AnisoMetric3d> {
    /// Compute an anisotropic metric based on the boundary curvature
    /// - geom : the geometry on which the curvature is computed
    /// - r_h: the curvature radius to element size ratio
    /// - beta: the mesh gradation
    /// - h_n: the normal size, defined at the boundary vertices
    ///   if <0, the min of the tangential sizes is used
    #[allow(clippy::too_many_arguments)]
    pub fn curvature_metric_3d<T2: Idx>(
        msh: &'a M,
        v2v: &'a CSRGraph,
        geom: &MeshedGeometry<3, Triangle<T2>, impl Mesh<3, Triangle<T2>>>,
        r_h: f64,
        beta: f64,
        t: f64,
        h_min: Option<f64>,
        h_max: Option<f64>,
        h_n: Option<&[f64]>,
        h_n_tags: Option<&[Tag]>,
    ) -> Result<Self> {
        debug!("Compute the curvature metric with r/h = {r_h} and gradation = {beta}");

        let (bdy, boundary_vertex_ids) = msh.boundary::<GenericMesh<3, Triangle<T>>>();
        let bdy_tags: FxHashSet<Tag> = bdy.etags().collect();

        // Initialize the metric field
        let m = AnisoMetric3d::default();
        let n_verts = msh.n_verts();
        let mut curvature_metric = vec![m; n_verts];
        let mut flg = vec![false; n_verts];

        // Set the metric at the boundary vertices
        for tag in bdy_tags {
            let ids = SubMesh::new(&bdy, |t| t == tag).parent_vert_ids;
            let use_h_n = h_n_tags.is_some_and(|h_n_tags| h_n_tags.contains(&tag));
            ids.iter()
                .map(|&i| (i, boundary_vertex_ids[i]))
                .for_each(|(i_bdy_vert, i_vert)| {
                    let pt = msh.vert(i_vert);
                    let (mut u, v) = geom.curvature(&pt, tag);
                    let mut v = v.unwrap();
                    let mut hu = 1. / (r_h * u.norm());
                    let mut hv = 1. / (r_h * v.norm());
                    let mut hn = f64::min(hu, hv);
                    if use_h_n && let Some(h_n) = h_n {
                        assert!(h_n[i_bdy_vert] > 0.0);
                        hn = h_n[i_bdy_vert].min(hn);
                    }
                    if let Some(h_min) = h_min {
                        hu = hu.max(h_min);
                        hv = hv.max(h_min);
                        hn = hn.max(h_min);
                    }
                    if let Some(h_max) = h_max {
                        hu = hu.min(h_max);
                        hv = hv.min(h_max);
                        hn = hn.min(h_max);
                    }
                    u.normalize_mut();
                    v.normalize_mut();
                    let n = hn * u.cross(&v);
                    u *= hu;
                    v *= hv;

                    curvature_metric[i_vert] = AnisoMetric3d::from_sizes(&n, &u, &v);
                    flg[i_vert] = true;
                });
        }

        let mut res = Self::new(msh, curvature_metric);
        res.extend_from_boundary(v2v, &mut flg, beta, t)?;

        Ok(res)
    }
}

impl<'a, T: Idx, M: Mesh<2, Triangle<T>>> MetricField<'a, 2, Triangle<T>, M, AnisoMetric2d> {
    /// Compute an anisotropic metric based on the boundary curvature
    /// - geom : the geometry on which the curvature is computed
    /// - r_h: the curvature radius to element size ratio
    /// - beta: the mesh gradation
    /// - h_n: the normal size, defined at the boundary vertices
    ///   if <0, the min of the tangential sizes is used
    #[allow(clippy::too_many_arguments)]
    pub fn curvature_metric_2d<T2: Idx>(
        msh: &'a M,
        v2v: &'a CSRGraph,
        geom: &MeshedGeometry<2, Edge<T2>, impl Mesh<2, Edge<T2>>>,
        r_h: f64,
        beta: f64,
        t: f64,
        h_min: Option<f64>,
        h_max: Option<f64>,
        h_n: Option<&[f64]>,
        h_n_tags: Option<&[Tag]>,
    ) -> Result<Self> {
        debug!("Compute the curvature metric with r/h = {r_h} and gradation = {beta}");

        let (bdy, boundary_vertex_ids) = msh.boundary::<GenericMesh<2, Edge<T>>>();
        let bdy_tags: FxHashSet<Tag> = bdy.etags().collect();

        // Initialize the metric field
        let m = AnisoMetric2d::default();
        let n_verts = msh.n_verts();
        let mut curvature_metric = vec![m; n_verts];
        let mut flg = vec![false; n_verts];

        // Set the metric at the boundary vertices
        for tag in bdy_tags {
            let ids = SubMesh::new(&bdy, |t| t == tag).parent_vert_ids;
            let use_h_n = h_n_tags.is_some_and(|h_n_tags| h_n_tags.contains(&tag));
            ids.iter()
                .map(|&i| (i, boundary_vertex_ids[i]))
                .for_each(|(i_bdy_vert, i_vert)| {
                    let pt = msh.vert(i_vert);
                    let (mut u, _) = geom.curvature(&pt, tag);
                    let mut hu = 1. / (r_h * u.norm());
                    let mut hn = hu;
                    if use_h_n && let Some(h_n) = h_n {
                        assert!(h_n[i_bdy_vert] > 0.0);
                        hn = h_n[i_bdy_vert].min(hn);
                    }
                    if let Some(h_min) = h_min {
                        hu = hu.max(h_min);
                        hn = hn.max(h_min);
                    }
                    if let Some(h_max) = h_max {
                        hu = hu.min(h_max);
                        hn = hn.min(h_max);
                    }
                    u.normalize_mut();

                    let n = hn * Vert2d::new(-u[1], u[0]);
                    u *= hu;

                    curvature_metric[i_vert] = AnisoMetric2d::from_sizes(&n, &u);
                    flg[i_vert] = true;
                });
        }

        let mut res = Self::new(msh, curvature_metric);
        res.extend_from_boundary(v2v, &mut flg, beta, t)?;

        Ok(res)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        ANISO_MAX, Result,
        geometry::MeshedGeometry,
        mesh::{MeshTopology, test_meshes::test_mesh_3d},
        metric::{AnisoMetric3d, Metric, MetricField},
    };
    use nalgebra::SVector;
    use tmesh::{
        Vert3d, assert_delta,
        mesh::{BoundaryMesh3d, Mesh},
    };

    #[test]
    #[allow(clippy::too_many_lines)]
    fn test_curvature() -> Result<()> {
        // build a cylinder mesh
        let (r_in, r_out) = (0.1, 0.5);
        let mut mesh = test_mesh_3d().split().split().split();

        mesh.verts_mut().for_each(|p| {
            let r = r_in + (r_out - r_in) * p[0];
            let theta = 3.0 * p[1];
            let z = p[2];
            let x = r * f64::cos(theta);
            let y = r * f64::sin(theta);
            *p = Vert3d::new(x, y, z);
        });

        // build the geometry
        let (bdy, bdy_ids) = mesh.boundary::<BoundaryMesh3d>();

        // tag vertices on the interior & exterior cylinders
        let mut bdy_flg = vec![0; bdy.n_verts()];
        let (tag_in, tag_out) = (6, 5);
        bdy.elems().zip(bdy.etags()).for_each(|(f, t)| {
            if t == tag_in {
                f.into_iter().for_each(|i| bdy_flg[i] = 1);
            }
            if t == tag_out {
                f.into_iter().for_each(|i| bdy_flg[i] = 2);
            }
        });
        bdy.elems().zip(bdy.etags()).for_each(|(f, t)| {
            if t != tag_in && t != tag_out {
                f.into_iter().for_each(|i| bdy_flg[i] = 0);
            }
        });

        let topo = MeshTopology::new(&mesh);

        let geom = MeshedGeometry::new(&mesh, &topo, bdy.clone())?;

        // curvature metric (no prescribes normal size)
        let v2v = mesh.vertex_to_vertices();
        let m_curv = MetricField::curvature_metric_3d(
            &mesh, &v2v, &geom, 4.0, 2.0, 1.0, None, None, None, None,
        )?;
        let m_curv = m_curv.metric();

        for (i_bdy_vert, &i_vert) in bdy_ids.iter().enumerate() {
            if bdy_flg[i_bdy_vert] == 1 {
                let m = m_curv[i_vert];
                let s = m.sizes();
                assert_delta!(s[0], r_in / 4.0, r_in * 0.1);
                assert_delta!(s[1], r_in / 4.0, r_in * 0.1);
                assert!(s[2] > 1000.0);
            }
            if bdy_flg[i_bdy_vert] == 2 {
                let m = m_curv[i_vert];
                let s = m.sizes();
                assert_delta!(s[0], r_out / 4.0, r_out * 0.1);
                assert_delta!(s[1], r_out / 4.0, r_out * 0.1);
                assert!(s[2] > 1000.0);
            }
        }

        // curvature metric (prescribed normal size on the inner cylinder)
        let h_0 = 1e-3;
        let mut h_n = vec![-1.0; bdy.n_verts()];
        bdy.elems().zip(bdy.etags()).for_each(|(f, t)| {
            if t == tag_in {
                f.into_iter().for_each(|i| h_n[i] = h_0);
            }
        });

        let v2v = mesh.vertex_to_vertices();
        let m_curv = MetricField::curvature_metric_3d(
            &mesh,
            &v2v,
            &geom,
            4.0,
            2.0,
            1.0,
            None,
            None,
            Some(&h_n),
            Some(&[tag_in]),
        )?;
        let m = m_curv.metric();
        for (i_bdy_vert, &i_vert) in bdy_ids.iter().enumerate() {
            if bdy_flg[i_bdy_vert] == 1 {
                let m = m[i_vert];
                let s = m.sizes();
                assert!(f64::abs(s[0] - h_0) < 1e-8);
                assert!(f64::abs(s[1] - r_in / 4.0) < r_in * 0.1);
                assert!(s[2] > f64::min(1000., 0.99 * h_0 * ANISO_MAX)); // bounded by ANISO_MAX
            }
            if bdy_flg[i_bdy_vert] == 2 {
                let m = m[i_vert];
                let s = m.sizes();
                assert!(f64::abs(s[0] - r_out / 4.0) < r_out * 0.1);
                assert!(f64::abs(s[1] - r_out / 4.0) < r_out * 0.1);
                assert!(s[2] > 1000.0);
            }
        }

        // test metric
        let mfunc = |_p| {
            let v0 = Vert3d::new(0.2, 0., 0.);
            let v1 = Vert3d::new(0.0, 0.1, 0.);
            let v2 = Vert3d::new(0., 0.0, 1e-4);
            AnisoMetric3d::from_sizes(&v0, &v1, &v2)
        };
        let m: Vec<_> = mesh.verts().map(mfunc).collect();
        let mut m = MetricField::new(&mesh, m);

        // Complexity
        m.scale((1e-2, 0.2), 10000, Some(&m_curv), None, None, 20)?;
        let c = m.complexity(0.0, 1.0);
        assert!(f64::abs(c - 10000.) < 1000.);

        let m = m.metric();
        let m_curv = m_curv.metric();
        for (i_vert, &m) in m.iter().enumerate() {
            let s = m.sizes();
            assert!(s[0] > 0.99 * 1e-2 && s[0] < 1.01 * 0.2);
            assert!(s[1] > 0.99 * 1e-2 && s[1] < 1.01 * 0.2);
            assert!(s[2] > 0.99 * 1e-2 && s[2] < 1.01 * 0.2);
            for _ in 0..100 {
                let v = SVector::<f64, 3>::new_random();
                let v = v.normalize();
                let l_m = m.length(&v);
                let mut m_curv = m_curv[i_vert];
                m_curv.scale_with_bounds(1.0, 1e-2, 0.2);
                let l_m_curv = m_curv.length(&v);
                assert!(l_m_curv < 1.01 * l_m);
            }
        }
        Ok(())
    }
}
