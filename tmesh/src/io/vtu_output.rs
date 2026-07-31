use std::{
    io::{BufWriter, Result},
    path::Path,
};

use crate::{
    Vertex,
    dual::{PolyMesh, PolyMeshType, merge_polylines},
    extruded::ExtrudedMesh2d,
    mesh::{Idx, Mesh, Simplex},
};
use rustc_hash::{FxBuildHasher, FxHashSet};
use tucanos_vtkio::{Scalar, UnstructuredGridWriter};

fn add_points<'a, const D: usize, IT>(writer: &mut UnstructuredGridWriter<'a>, verts: IT)
where
    IT: Iterator<Item = Vertex<D>> + 'a,
{
    match D {
        3 => writer.add_points(verts.map(|v| v.data.0[0])),
        2 => writer.add_points(verts.map(|x| [x[0], x[1], 0.0])),
        _ => unimplemented!(),
    }
}
/// VTU file writer
pub struct VTUFile<'a>(UnstructuredGridWriter<'a>);

impl<'a> VTUFile<'a> {
    /// Create a vtu Mesh writer
    pub fn from_mesh<const D: usize, M: Mesh<D>>(mesh: &'a M) -> Self {
        let mut w = UnstructuredGridWriter::default();
        w.set_num_points(mesh.n_verts());
        let n = mesh.n_elems();
        w.set_num_cells(n);
        add_points(&mut w, mesh.verts());
        w.add_cell_data("tags", 1, mesh.etags());
        let offsets = (0..n).map(|i| <M::C as Simplex>::N_VERTS * (i + 1));
        let cell_type: u8 = match <M::C as Simplex>::order() {
            1 => match <M::C as Simplex>::N_VERTS {
                4 => 10,
                3 => 5,
                2 => 3,
                _ => unimplemented!(),
            },
            2 => match <M::C as Simplex>::N_VERTS {
                10 => 24,
                6 => 22,
                3 => 21,
                _ => unimplemented!(),
            },
            _ => unimplemented!(),
        };
        let types = (0..n).map(move |_i| cell_type);
        w.add_cells(
            <M::C as Simplex>::N_VERTS * n,
            mesh.elems().flatten(),
            offsets,
            types,
        );
        Self(w)
    }

    /// Create a vtu ExtrudedMesh2d writer
    #[must_use]
    pub fn from_extruded_mesh(mesh: &'a ExtrudedMesh2d<impl Idx>) -> Self {
        let mut w = UnstructuredGridWriter::default();
        w.set_num_points(mesh.n_verts());
        let n = mesh.n_prisms();
        w.set_num_cells(n);
        add_points(&mut w, mesh.verts());
        w.add_cell_data("tags", 1, mesh.prism_tags());
        let prisms = mesh.prisms();
        let connectivity = prisms.copied().flatten();
        let offsets = (0..n).map(|i| 6 * (i + 1));
        let types = (0..n).map(|_i| 13_u8);
        w.add_cells(6 * n, connectivity, offsets, types);
        Self(w)
    }

    /// Create a vtu PolyMesh writer
    pub fn from_poly_mesh<const D: usize, M: PolyMesh<D>>(mesh: &'a M) -> Self {
        let mut w = UnstructuredGridWriter::default();
        w.set_num_points(mesh.n_verts());
        w.set_num_cells(mesh.n_elems());
        add_points(&mut w, mesh.verts());
        w.add_cell_data("tags", 1, mesh.etags());
        let (connectivity, offsets) = Self::poly_connectivity(mesh);
        let n = mesh.n_elems();
        let cell_type: u8 = match mesh.poly_type() {
            PolyMeshType::Polylines => 4,
            PolyMeshType::Polygons => 7,
            PolyMeshType::Polyhedra => 42,
        };
        let types = (0..n).map(move |_| cell_type);
        w.add_cells(connectivity.len(), connectivity, offsets, types);
        if matches!(mesh.poly_type(), PolyMeshType::Polyhedra) {
            let mut faces = Vec::new();
            let mut faceoffsets = Vec::new();

            for e in mesh.elems() {
                faces.push(e.len());
                for (i_face, orient) in e {
                    let mut f = mesh.face(i_face).collect::<Vec<_>>();
                    if !orient {
                        f.reverse();
                    }
                    faces.push(f.len());
                    faces.extend_from_slice(&f);
                }
                faceoffsets.push(faces.len());
            }
            w.add_polyhedron_faces(faces.len(), faces, faceoffsets.len(), faceoffsets);
        }
        Self(w)
    }

    fn poly_connectivity<const D: usize, M: PolyMesh<D>>(mesh: &M) -> (Vec<usize>, Vec<usize>) {
        let mut connectivity = Vec::new();
        let mut offsets = Vec::new();
        for e in mesh.elems() {
            match mesh.poly_type() {
                PolyMeshType::Polylines => todo!(),
                PolyMeshType::Polygons => {
                    // copy faces to reorient if needed
                    let mut tmp_ptr = Vec::new();
                    tmp_ptr.push(0);
                    let mut tmp = Vec::new();
                    let n_faces = e.len();
                    for (i_face, orient) in e {
                        let mut face = mesh.face(i_face).collect::<Vec<_>>();
                        if !orient {
                            face.reverse();
                        }
                        tmp.extend_from_slice(&face);
                        tmp_ptr.push(tmp.len());
                    }
                    let faces = (0..n_faces)
                        .map(|i| {
                            let start = tmp_ptr[i];
                            let end = tmp_ptr[i + 1];
                            &tmp[start..end]
                        })
                        .collect::<Vec<_>>();

                    let polygons = merge_polylines(&faces);
                    assert_eq!(polygons.len(), 1, "faces = {faces:?}");
                    connectivity.extend_from_slice(&polygons[0]);
                }
                PolyMeshType::Polyhedra => {
                    let mut tmp = FxHashSet::with_hasher(FxBuildHasher);
                    for (i_face, _) in e {
                        let face = mesh.face(i_face);
                        for i_vert in face {
                            tmp.insert(i_vert);
                        }
                    }
                    connectivity.extend(tmp.iter().copied());
                }
            }
            offsets.push(connectivity.len());
        }
        (connectivity, offsets)
    }

    pub fn export<P: AsRef<Path>>(self, file_name: P) -> Result<()> {
        let f = std::fs::File::create(file_name)?;
        let mut writer = BufWriter::new(f);
        self.0.write(&mut writer)
    }

    pub fn add_cell_data<T, IT>(&mut self, label: &str, num_components: usize, values: IT)
    where
        T: Scalar + 'a,
        IT: IntoIterator<Item = T> + 'a,
    {
        self.0.add_cell_data(label, num_components, values);
    }

    pub fn add_point_data<T, IT>(&mut self, label: &str, num_components: usize, values: IT)
    where
        T: Scalar + 'a,
        IT: IntoIterator<Item = T> + 'a,
    {
        self.0.add_point_data(label, num_components, values);
    }
}

#[cfg(test)]
mod tests {
    use super::VTUFile;
    use crate::mesh::{Mesh2d, rectangle_mesh};

    #[test]
    fn test_write_triangles() {
        let msh: Mesh2d = rectangle_mesh(1.0, 10, 2.0, 15);
        VTUFile::from_mesh(&msh).export("toto.vtu").unwrap();
    }
}
