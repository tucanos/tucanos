use crate::{
    Tag, Vertex,
    dual::{PolyMesh, PolyMeshType, merge_polylines},
    extruded::ExtrudedMesh2d,
    mesh::{Idx, Mesh, Prism, Simplex},
};
use rustc_hash::{FxBuildHasher, FxHashSet};
use std::io::{BufWriter, Result, Write};

/// VTU file writer
pub struct VTUFile {
    number_of_points: usize,
    number_of_cells: usize,
    points: DataArray,
    cells: Vec<DataArray>,
    cell_data: Vec<DataArray>,
    point_data: Vec<DataArray>,
}

#[derive(Clone, Copy)]
enum Version {
    /// File version for best compatibility
    V0_1,
    /// File version only compatible with VTK >= 9.4, Paraview >= 6
    #[allow(dead_code)]
    V2_3,
}

impl VTUFile {
    /// Create a vtu Mesh writer
    pub fn from_mesh<const D: usize, M: Mesh<D>>(mesh: &M) -> Self {
        Self {
            number_of_points: mesh.n_verts(),
            number_of_cells: mesh.n_elems(),
            points: DataArray::from_verts(mesh.verts()),
            cells: DataArray::from_elems(mesh),
            cell_data: vec![DataArray::from_etags(mesh.etags())],
            point_data: vec![],
        }
    }

    /// Create a vtu ExtrudedMesh2d writer
    #[must_use]
    pub fn from_extruded_mesh(mesh: &ExtrudedMesh2d<impl Idx>) -> Self {
        Self {
            number_of_points: mesh.n_verts(),
            number_of_cells: mesh.n_prisms(),
            points: DataArray::from_verts(mesh.verts()),
            cells: DataArray::from_prisms(mesh.prisms()),
            cell_data: vec![DataArray::from_etags(mesh.prism_tags())],
            point_data: vec![],
        }
    }

    /// Create a vtu PolyMesh writer
    pub fn from_poly_mesh<const D: usize, M: PolyMesh<D>>(mesh: &M) -> Self {
        Self {
            number_of_points: mesh.n_verts(),
            number_of_cells: mesh.n_elems(),
            points: DataArray::from_verts(mesh.verts()),
            cells: DataArray::from_poly(mesh, Version::V0_1),
            cell_data: vec![DataArray::from_etags(mesh.etags())],
            point_data: vec![],
        }
    }

    /// Add cell data
    pub fn add_cell_data<T: ScalarData>(
        &mut self,
        name: &str,
        number_of_components: usize,
        data: impl Iterator<Item = T>,
    ) {
        self.cell_data
            .push(DataArray::new(name, number_of_components, data));
    }

    /// Add point data
    pub fn add_point_data<T: ScalarData>(
        &mut self,
        name: &str,
        number_of_components: usize,
        data: impl Iterator<Item = T>,
    ) {
        self.point_data
            .push(DataArray::new(name, number_of_components, data));
    }

    /// Write the file
    pub fn export(&self, file_name: &str) -> Result<()> {
        let f = std::fs::File::create(file_name)?;
        let mut writer = BufWriter::new(f);
        writeln!(
            writer,
            concat!(
                r#"<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian""#,
                r#" header_type="UInt64">
  <UnstructuredGrid>
    <Piece NumberOfPoints="{}" NumberOfCells="{}">"#
            ),
            self.number_of_points, self.number_of_cells
        )?;
        let mut offset = 0;
        let mut write_section = |name: &str, arrays: &[DataArray]| -> Result<()> {
            if arrays.is_empty() {
                return Ok(());
            }
            writeln!(writer, "      <{name}>")?;
            for a in arrays {
                writeln!(writer, "        {}", a.to_xml_tag(offset))?;
                offset += size_of::<u64>() + a.data.len();
            }
            writeln!(writer, "      </{name}>")
        };

        write_section("Points", std::slice::from_ref(&self.points))?;
        write_section("Cells", &self.cells)?;
        write_section("CellData", &self.cell_data)?;
        write_section("PointData", &self.point_data)?;
        write!(
            writer,
            "    </Piece>\n  </UnstructuredGrid>\n  <AppendedData encoding=\"raw\">\n   _"
        )?;
        for a in std::iter::once(&self.points)
            .chain(&self.cells)
            .chain(&self.cell_data)
            .chain(&self.point_data)
        {
            a.write(&mut writer)?;
        }
        Ok(())
    }
}

struct DataArray {
    data_type: String,
    name: String,
    number_of_components: usize,
    data: Vec<u8>,
}

pub trait ScalarData: Sized {
    const TYPE_NAME: &'static str;
    type Bytes: IntoIterator<Item = u8>;
    fn to_le_bytes(self) -> Self::Bytes;
}

impl ScalarData for f64 {
    const TYPE_NAME: &'static str = "Float64";
    type Bytes = [u8; 8];
    fn to_le_bytes(self) -> Self::Bytes {
        self.to_le_bytes()
    }
}

#[cfg(target_pointer_width = "64")]
impl ScalarData for usize {
    const TYPE_NAME: &'static str = "UInt64";
    type Bytes = [u8; 8];

    fn to_le_bytes(self) -> Self::Bytes {
        self.to_le_bytes()
    }
}

#[cfg(feature = "64bit-tags")]
impl ScalarData for i64 {
    const TYPE_NAME: &'static str = "Int64";
    type Bytes = [u8; 8];
    fn to_le_bytes(self) -> Self::Bytes {
        self.to_le_bytes()
    }
}

#[cfg(feature = "32bit-tags")]
impl ScalarData for i32 {
    const TYPE_NAME: &'static str = "Int32";
    type Bytes = [u8; 4];
    fn to_le_bytes(self) -> Self::Bytes {
        self.to_le_bytes()
    }
}

#[cfg(not(any(feature = "32bit-tags", feature = "64bit-tags")))]
impl ScalarData for i16 {
    const TYPE_NAME: &'static str = "Int16";
    type Bytes = [u8; 2];
    fn to_le_bytes(self) -> Self::Bytes {
        self.to_le_bytes()
    }
}

impl ScalarData for u8 {
    const TYPE_NAME: &'static str = "UInt8";
    type Bytes = [Self; 1];
    fn to_le_bytes(self) -> Self::Bytes {
        self.to_le_bytes()
    }
}

impl DataArray {
    pub fn new<T>(
        name: &str,
        number_of_components: usize,
        data: impl IntoIterator<Item = T>,
    ) -> Self
    where
        T: ScalarData,
    {
        Self {
            data_type: T::TYPE_NAME.to_string(),
            name: name.to_string(),
            number_of_components,
            data: data.into_iter().flat_map(T::to_le_bytes).collect(),
        }
    }
    pub fn from_etags(data: impl ExactSizeIterator<Item = Tag>) -> Self {
        Self::new("tags", 1, data)
    }
    pub fn from_prisms<'a>(
        prisms: impl ExactSizeIterator<Item = &'a Prism<impl Idx>>,
    ) -> Vec<Self> {
        let n = prisms.len();
        let connectivity = Self::new("connectivity", 1, prisms.copied().flatten());
        let data = (0..n).map(|i| 6 * (i + 1));
        let offsets = Self::new("offsets", 1, data);
        let data = (0..n).map(|_i| 13_u8);
        let types = Self::new("types", 1, data);
        vec![connectivity, offsets, types]
    }

    pub fn from_elems<const D: usize, M: Mesh<D>>(mesh: &M) -> Vec<Self> {
        let n = mesh.n_elems();

        let connectivity = Self::new("connectivity", 1, mesh.elems().flatten());

        let data = (0..n).map(|i| <M::C as Simplex>::N_VERTS * (i + 1));
        let offsets = Self::new("offsets", 1, data);

        let cell_type: u8 = match <M::C as Simplex>::order() {
            1 => match <M::C as Simplex>::N_VERTS {
                4 => 10,
                3 => 5,
                2 => 3,
                _ => unimplemented!(),
            },
            2 => match <M::C as Simplex>::N_VERTS {
                6 => 22,
                3 => 21,
                _ => unimplemented!(),
            },
            _ => unimplemented!(),
        };

        let data = (0..n).map(|_i| cell_type);
        let types = Self::new("types", 1, data);

        vec![connectivity, offsets, types]
    }

    pub fn from_verts<const D: usize>(data: impl ExactSizeIterator<Item = Vertex<D>>) -> Self {
        let name = "Points";
        match D {
            3 => Self::new(name, 3, data.flat_map(|x| [x[0], x[1], x[2]])),
            2 => Self::new(name, 3, data.flat_map(|x| [x[0], x[1], 0.0])),
            _ => unimplemented!(),
        }
    }

    /// Generates the XML header tag for the Appended mode.
    pub fn to_xml_tag(&self, offset: usize) -> String {
        format!(
            concat!(
                r#"<DataArray type="{}" Name="{}" NumberOfComponents="{}" "#,
                r#"format="appended" offset="{}"/>"#
            ),
            self.data_type, self.name, self.number_of_components, offset
        )
    }

    pub fn write<W: Write>(&self, writer: &mut W) -> Result<()> {
        writer.write_all(&self.data.len().to_le_bytes())?;
        writer.write_all(&self.data)
    }

    fn poly_connectivity<const D: usize, M: PolyMesh<D>>(mesh: &M) -> (Self, Self) {
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
                    assert_eq!(polygons.len(), 1);
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
        let connectivity = Self::new("connectivity", 1, connectivity);
        let offsets = Self::new("offsets", 1, offsets);
        (connectivity, offsets)
    }

    pub fn from_poly<const D: usize, M: PolyMesh<D>>(mesh: &M, version: Version) -> Vec<Self> {
        let n = mesh.n_elems();
        let (connectivity, offsets) = Self::poly_connectivity(mesh);
        let cell_type: u8 = match mesh.poly_type() {
            PolyMeshType::Polylines => 4,
            PolyMeshType::Polygons => 7,
            PolyMeshType::Polyhedra => 42,
        };

        let types = Self::new("types", 1, (0..n).map(|_| cell_type));

        let mut data_array = vec![connectivity, offsets, types];

        if matches!(mesh.poly_type(), PolyMeshType::Polyhedra) {
            match version {
                Version::V0_1 => {
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

                    data_array.push(Self::new("faces", 1, faces));
                    data_array.push(Self::new("faceoffsets", 1, faceoffsets));
                }
                Version::V2_3 => {
                    let mut polyhedron_offsets = Vec::with_capacity(n);
                    let mut offset = 0;
                    for e in mesh.elems() {
                        offset += e.len();
                        polyhedron_offsets.push(offset);
                    }
                    let polyhedron_to_faces = Self::new(
                        "polyhedron_to_faces",
                        1,
                        mesh.elems().flat_map(|x| x.map(|(x, _)| x)),
                    );

                    let polyhedron_offsets = Self::new("polyhedron_offsets", 1, polyhedron_offsets);

                    let n = mesh.n_faces();
                    let mut face_offsets = Vec::with_capacity(n);
                    let mut offset = 0;
                    for e in mesh.faces() {
                        offset += e.len();
                        face_offsets.push(offset);
                    }

                    data_array.push(Self::new("face_connectivity", 1, mesh.faces().flatten()));
                    data_array.push(Self::new("face_offsets", 1, face_offsets));
                    data_array.push(polyhedron_to_faces);
                    data_array.push(polyhedron_offsets);
                }
            }
        }

        data_array
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
