//! Extrude 2d triangle meshes to 3d as 1 layer of prisms
use crate::{
    Error, Result, Tag, Vert2d, Vert3d,
    dual::{DualMesh2d, PolyMesh, PolyMeshType, SimplePolyMesh, merge_polylines},
    io::{VTUEncoding, VTUFile},
    mesh::{Mesh, Mesh2d, Prism, Quadrangle, Triangle},
};

/// Extrusion of a `Mesh2d` along `z`
pub struct ExtrudedMesh2d {
    verts: Vec<Vert3d>,
    prisms: Vec<Prism>,
    prism_tags: Vec<Tag>,
    tris: Vec<Triangle>,
    tri_tags: Vec<Tag>,
    quads: Vec<Quadrangle>,
    quad_tags: Vec<Tag>,
}

impl ExtrudedMesh2d {
    /// Create a new mesh from coordinates, connectivities and tags
    #[must_use]
    pub const fn new(
        verts: Vec<Vert3d>,
        prisms: Vec<Prism>,
        prism_tags: Vec<Tag>,
        tris: Vec<Triangle>,
        tri_tags: Vec<Tag>,
        quads: Vec<Quadrangle>,
        quad_tags: Vec<Tag>,
    ) -> Self {
        Self {
            verts,
            prisms,
            prism_tags,
            tris,
            tri_tags,
            quads,
            quad_tags,
        }
    }

    /// Extrude a `Mesh2d` by a distance `h` along direction `z`
    #[must_use]
    pub fn from_mesh2d(msh: &Mesh2d, h: f64) -> Self {
        assert!(h > 0.0);

        let n = msh.n_verts();
        let mut verts = msh
            .verts()
            .map(|v| Vert3d::new(v[0], v[1], 0.0))
            .collect::<Vec<_>>();
        verts.extend(msh.verts().map(|v| Vert3d::new(v[0], v[1], h)));

        let elems = msh
            .elems()
            .map(|tri| [tri[0], tri[1], tri[2], tri[0] + n, tri[1] + n, tri[2] + n])
            .collect::<Vec<_>>();
        let etags = msh.etags().collect::<Vec<_>>();

        let mut tris = msh.elems().collect::<Vec<_>>();
        tris.extend(msh.elems().map(|tri| [tri[0] + n, tri[2] + n, tri[1] + n]));
        let mut tri_tags = vec![Tag::MAX; msh.n_elems()];
        tri_tags.resize(2 * msh.n_elems(), Tag::MAX - 1);

        let quads = msh
            .faces()
            .map(|edg| [edg[0], edg[1], edg[1] + n, edg[0] + n])
            .collect::<Vec<_>>();
        let quad_tags = msh.ftags().collect::<Vec<_>>();

        Self {
            verts,
            prisms: elems,
            prism_tags: etags,
            tris,
            tri_tags,
            quads,
            quad_tags,
        }
    }

    /// Get a `Mesh2d` from the z=0 face
    pub fn to_mesh2d(&self) -> Result<Mesh2d> {
        let n = self.verts.len() / 2;

        let mut ok = true;
        let verts = self
            .verts
            .iter()
            .take(n)
            .map(|v| {
                if v[2].abs() > 1e-12 {
                    ok = false;
                }
                Vert2d::new(v[0], v[1])
            })
            .collect::<Vec<_>>();
        if !ok {
            return Err(Error::from("Unable to convert to Mesh2d"));
        }

        let elems = self
            .prisms
            .iter()
            .map(|p| [p[0], p[1], p[2]])
            .collect::<Vec<_>>();
        let etags = self.prism_tags.clone();

        let faces = self.quads.iter().map(|p| [p[0], p[1]]).collect::<Vec<_>>();
        let ftags = self.quad_tags.clone();

        Ok(Mesh2d::new(&verts, &elems, &etags, &faces, &ftags))
    }

    /// Number of vertices
    #[must_use]
    pub const fn n_verts(&self) -> usize {
        self.verts.len()
    }

    /// Sequential iterator over the vertices
    #[must_use]
    pub fn verts(&self) -> impl ExactSizeIterator<Item = Vert3d> + '_ {
        self.verts.iter().copied()
    }

    /// Number of prisms
    #[must_use]
    pub const fn n_prisms(&self) -> usize {
        self.prisms.len()
    }

    /// Sequential iterator over the prisms
    #[must_use]
    pub fn prisms(&self) -> impl ExactSizeIterator<Item = &Prism> + '_ {
        self.prisms.iter()
    }

    /// Sequential iterator over the prism tags
    #[must_use]
    pub fn prism_tags(&self) -> impl ExactSizeIterator<Item = Tag> + '_ {
        self.prism_tags.iter().copied()
    }

    /// Number of triangles
    #[must_use]
    pub const fn n_tris(&self) -> usize {
        self.tris.len()
    }

    /// Sequential iterator over the triangles
    #[must_use]
    pub fn tris(&self) -> impl ExactSizeIterator<Item = &Triangle> + '_ {
        self.tris.iter()
    }

    /// Sequential iterator over the triangle tags
    #[must_use]
    pub fn tri_tags(&self) -> impl ExactSizeIterator<Item = Tag> + '_ {
        self.tri_tags.iter().copied()
    }

    /// Number of quadrangles
    #[must_use]
    pub const fn n_quads(&self) -> usize {
        self.quads.len()
    }

    /// Sequential iterator over the quadrangles
    #[must_use]
    pub fn quads(&self) -> impl ExactSizeIterator<Item = &Quadrangle> + '_ {
        self.quads.iter()
    }

    /// Sequential iterator over the quandrangle tags
    #[must_use]
    pub fn quad_tags(&self) -> impl ExactSizeIterator<Item = Tag> + '_ {
        self.quad_tags.iter().copied()
    }

    /// Write the mesh in a `.vtu` file
    pub fn write_vtk(&self, file_name: &str) -> Result<()> {
        let vtu = VTUFile::from_extruded_mesh(self, VTUEncoding::Binary);

        vtu.export(file_name)?;

        Ok(())
    }
}

impl Mesh2d {
    /// Extrude the mesh by a distance `h` along direction `z`
    #[must_use]
    pub fn extrude(&self, h: f64) -> ExtrudedMesh2d {
        ExtrudedMesh2d::from_mesh2d(self, h)
    }
}

impl DualMesh2d {
    /// Extrude the mesh by a distance `h` along direction `z`
    #[must_use]
    pub fn extrude(&self, h: f64) -> SimplePolyMesh<3> {
        let mut res = SimplePolyMesh::<3>::empty(PolyMeshType::Polyhedra);

        let n = self.n_verts();
        for v in self.verts() {
            res.insert_vert(Vert3d::new(v[0], v[1], 0.0));
        }
        for v in self.verts() {
            res.insert_vert(Vert3d::new(v[0], v[1], h));
        }
        for (f, t) in self.faces().zip(self.ftags()) {
            assert_eq!(f.len(), 2);
            res.insert_face(&[f[0], f[1], f[1] + n, f[0] + n], t);
        }

        for (e, t) in self.elems().zip(self.etags()) {
            let tmp = e
                .iter()
                .map(|&(i, orient)| {
                    let mut f = self.face(i).to_vec();
                    if !orient {
                        f.reverse();
                    }
                    f
                })
                .collect::<Vec<_>>();

            let polylines = tmp.iter().map(std::vec::Vec::as_slice).collect::<Vec<_>>();

            let polygons = merge_polylines(&polylines);
            assert_eq!(polygons.len(), 1);

            let polygon = &polygons[0];
            let i0 = res.insert_face(polygon, Tag::MAX);
            let new_polygon = polygon.iter().rev().map(|&i| i + n).collect::<Vec<_>>();
            let i1 = res.insert_face(&new_polygon, Tag::MAX - 1);

            let mut new_e = e.to_vec();
            new_e.push((i0, true));
            new_e.push((i1, true));

            res.insert_elem(&new_e, t);
        }
        res
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        dual::{DualMesh, DualMesh2d, DualType},
        mesh::{Mesh, Mesh2d, rectangle_mesh},
    };

    use super::ExtrudedMesh2d;

    #[test]
    fn test_extrude_mesh() {
        let msh = rectangle_mesh::<Mesh2d>(1.0, 10, 2.0, 15);
        let extruded = ExtrudedMesh2d::from_mesh2d(&msh, 1.0);
        // extruded.write_vtk("extruded.vtu").unwrap();

        let msh2 = extruded.to_mesh2d().unwrap();
        msh.check_equals(&msh2, 1e-12).unwrap();
    }

    #[test]
    fn test_extrude_dual() {
        let msh = rectangle_mesh::<Mesh2d>(1.0, 10, 2.0, 15);
        let dual = DualMesh2d::new(&msh, DualType::Median);
        let _extruded = dual.extrude(1.0);
        // extruded.write_vtk("extruded_self.vtu").unwrap();
    }
}
