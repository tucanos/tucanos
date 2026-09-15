use tmesh::{
    Tag,
    mesh::{GenericMesh, Mesh, Simplex},
};

use crate::{
    Result,
    geometry::MeshedGeometry,
    mesh::MeshTopology,
    metric::{ImpliedMetric, Metric, MetricField},
    remesher::{Remesher, RemesherParams},
};

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
    let mut remesher = Remesher::new(&split_msh, topo, &m, &geom)?;

    let params = RemesherParams::default();
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
    use tmesh::mesh::{Mesh, Mesh3d, box_mesh};

    use crate::remesher::remesh_isosurface;

    #[test]
    fn test_3d() {
        let msh: Mesh3d = box_mesh::<Mesh3d>(1.0, 10, 1.0, 10, 1.0, 10).random_shuffle();

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
        use tmesh::mesh::Mesh2d;
        let mut msh =
            Mesh2d::from_meshb("/midterm/FP/garnaud_x/CODA/tucanos/ice/mesh.meshb").unwrap();
        msh.fix().unwrap();
        let (f, m) = Mesh2d::read_solb("/midterm/FP/garnaud_x/CODA/tucanos/ice/sdf.solb").unwrap();
        assert_eq!(m, 1);

        let (split_msh, res) = remesh_isosurface(&msh, &f).unwrap();
        split_msh.check(&split_msh.all_faces()).unwrap();
        res.check(&res.all_faces()).unwrap();
        // split_msh.write_meshb("iso_split_2d.meshb").unwrap();
        // res.write_meshb("iso_remeshed_2d.meshb").unwrap();
    }
}
