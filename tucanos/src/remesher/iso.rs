use tmesh::{
    Tag,
    mesh::{GenericMesh, Mesh, Simplex},
};

use crate::{
    Dim, Result,
    geometry::{Geometry, MeshedGeometry},
    mesh::MeshTopology,
    metric::{ImpliedMetric, Metric, MetricField},
    remesher::{
        CollapseParams, Remesher, RemesherParams, RemeshingStep, SmoothParams, SmoothingMethod,
    },
};

struct IsoGeometry<const D: usize, M: Mesh<D>> {
    geom: MeshedGeometry<D, M>,
}

impl<const D: usize, M: Mesh<D>> Geometry<D> for IsoGeometry<D, M> {
    fn check(&self, _topo: &crate::mesh::Topology) -> Result<()> {
        Ok(())
    }

    fn project(&self, pt: &mut tmesh::Vertex<D>, tag: &crate::TopoTag) -> f64 {
        if *tag == (<M::C as Simplex>::DIM as Dim, Tag::MAX) {
            0.0
        } else {
            self.geom.project(pt, tag)
        }
    }

    fn angle(&self, pt: &tmesh::Vertex<D>, n: &tmesh::Vertex<D>, tag: &crate::TopoTag) -> f64 {
        if *tag == (<M::C as Simplex>::DIM as Dim, Tag::MAX) {
            0.0
        } else {
            self.geom.angle(pt, n, tag)
        }
    }
}

#[allow(clippy::type_complexity)]
pub fn remesh_isosurface<T, const D: usize, M: Mesh<D>>(
    msh: &M,
    f: &[f64],
) -> Result<(GenericMesh<D, M::C>, GenericMesh<D, M::C>)>
where
    T: Metric<D> + Send + Sync + Default + Clone,
    <<M as Mesh<D>>::C as Simplex>::GEOM<D>: ImpliedMetric<T>,
{
    assert_eq!(msh.n_verts(), f.len());

    let m = MetricField::implied_metric(msh);
    let mut m = m.metric().to_vec();

    let (mut split_msh, split_edgs) = msh.split_isosurface::<GenericMesh<D, M::C>>(f);
    let n = msh.n_verts();
    let n2 = split_msh.n_verts();
    m.resize(n2, T::default());
    for (e, idx) in &split_edgs {
        let i_vert: usize = (*idx).try_into().unwrap();
        let i0 = e.get(0);
        let i1 = e.get(1);
        debug_assert!(i_vert >= n && i_vert < n2);
        debug_assert!(i0 < n && i1 < n);
        m[i_vert] = T::interpolate([(0.5, &m[i0]), (0.5, &m[i1])].into_iter());
    }

    split_msh.etags_mut().for_each(|t| *t += 2);
    let min_tag = split_msh.ftags().min().unwrap_or(0) - 1;
    split_msh.ftags_mut().for_each(|t| {
        if *t != Tag::MAX {
            *t -= min_tag;
        }
    });

    split_msh.check(&split_msh.all_faces())?;

    let topo = &MeshTopology::new(&split_msh);
    let (mut bdy, _) = split_msh.boundary::<GenericMesh<D, <M::C as Simplex>::FACE>>();
    bdy.fix().unwrap();

    let geom = MeshedGeometry::new(&bdy)?;
    let geom = IsoGeometry { geom };
    let mut remesher = Remesher::new(&split_msh, topo, &m, &geom)?;

    let collapse = RemeshingStep::Collapse(CollapseParams {
        l: 0.5,
        max_iter: 3,
        max_l_rel: f64::MAX,
        max_l_abs: f64::MAX,
        min_q_rel: 1e-6,
        min_q_abs: 1e-6,
        max_angle: 25.0,
    });
    let smooth = RemeshingStep::Smooth(SmoothParams {
        n_iter: 3,
        max_angle: 25.0,
        method: SmoothingMethod::Laplacian,
        relax: vec![1.0, 0.5, 0.25],
        keep_local_minima: false,
    });

    let mut steps = Vec::new();
    for _ in 0..3 {
        steps.push(collapse.clone());
        steps.push(smooth.clone());
    }

    // let steps = vec![collapse, smooth, collapse, smooth];
    let params = RemesherParams {
        steps,
        debug: false,
    };
    remesher.remesh(&params, &geom)?;

    let mut out = remesher.to_mesh(false);

    out.etags_mut().for_each(|t| *t -= 2);
    out.ftags_mut().for_each(|t| {
        if *t != Tag::MAX {
            *t += min_tag;
        }
    });

    Ok((split_msh, out))
}

#[cfg(test)]
mod tests {
    use tmesh::mesh::{Mesh, Mesh2d, Mesh3d, box_mesh, rectangle_mesh};

    use crate::remesher::remesh_isosurface;

    #[test]
    fn test_3d() {
        let msh = box_mesh::<Mesh3d>(1.0, 10, 1.0, 10, 1.0, 10).random_shuffle();

        let f = msh
            .verts()
            .map(|p| {
                let r0 = (p[0] - 0.5).hypot(p[1] - 0.5);
                let r1 = (p[0]).hypot(p[1]);
                (r0 - 0.25) * (r1 - 0.25)
            })
            .collect::<Vec<f64>>();

        let (split_msh, res) = remesh_isosurface(&msh, &f).unwrap();
        split_msh.check(&split_msh.all_faces()).unwrap();
        res.check(&res.all_faces()).unwrap();
        // split_msh.write_meshb("iso_split.meshb").unwrap();
        // res.write_meshb("iso_remeshed.meshb").unwrap();
    }

    #[test]
    fn test_2d() {
        let msh = rectangle_mesh::<Mesh2d>(1.0, 10, 1.0, 10).random_shuffle();

        let f = msh
            .verts()
            .map(|p| {
                let r0 = (p[0] - 0.5).hypot(p[1] - 0.5);
                let r1 = (p[0]).hypot(p[1]);
                (r0 - 0.25) * (r1 - 0.25)
            })
            .collect::<Vec<f64>>();

        let (split_msh, res) = remesh_isosurface(&msh, &f).unwrap();
        split_msh.check(&split_msh.all_faces()).unwrap();
        res.check(&res.all_faces()).unwrap();
        // split_msh.write_meshb("iso_split_2d.meshb").unwrap();
        // res.write_meshb("iso_remeshed_2d.meshb").unwrap();
    }
}
