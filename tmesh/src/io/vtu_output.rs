use crate::{
    Result, Tag, Vertex,
    dual::{PolyMesh, PolyMeshType, merge_polylines},
    extruded::ExtrudedMesh2d,
    mesh::{Idx, Mesh, Prism, Simplex},
};
use base64::Engine as _;
use quick_xml::se::to_utf8_io_writer;
use rustc_hash::{FxBuildHasher, FxHashSet};
use serde::Serialize;
use std::io::{BufWriter, Write};

/// Encoding for vtk files
#[derive(Clone, Copy)]
#[allow(dead_code)]
pub enum VTUEncoding {
    /// Ascii
    Ascii,
    /// Binary
    Binary,
}

#[derive(Serialize)]
#[serde(rename = "VTKFile", rename_all = "PascalCase")]
/// VTU file writer
pub struct VTUFile {
    #[serde(rename = "@type")]
    grid_type: String,
    #[serde(rename = "@version")]
    version: f64,
    #[serde(rename = "@header_type")]
    header_type: String,
    #[serde(rename = "@byte_order")]
    byte_order: String,
    unstructured_grid: UnstructuredGrid,
    #[serde(skip_serializing)]
    encoding: VTUEncoding,
}

impl VTUFile {
    /// Create a vtu Mesh writer
    pub fn from_mesh<const D: usize, M: Mesh<D>>(mesh: &M, encoding: VTUEncoding) -> Self {
        Self {
            grid_type: "UnstructuredGrid".to_string(),
            version: 0.1,
            header_type: "UInt32".to_string(),
            byte_order: "LittleEndian".to_string(),
            unstructured_grid: UnstructuredGrid {
                piece: Piece {
                    number_of_points: mesh.n_verts(),
                    number_of_cells: mesh.n_elems(),
                    points: Points::from_verts(mesh.verts(), encoding),
                    cells: Cells::from_elems(mesh, encoding),
                    cell_data: CellData::from_etags(mesh.etags(), encoding),
                    point_data: PointData::new(),
                },
            },
            encoding,
        }
    }

    /// Create a vtu ExtrudedMesh2d writer
    #[must_use]
    pub fn from_extruded_mesh(mesh: &ExtrudedMesh2d<impl Idx>, encoding: VTUEncoding) -> Self {
        Self {
            grid_type: "UnstructuredGrid".to_string(),
            version: 0.1,
            header_type: "UInt32".to_string(),
            byte_order: "LittleEndian".to_string(),
            unstructured_grid: UnstructuredGrid {
                piece: Piece {
                    number_of_points: mesh.n_verts(),
                    number_of_cells: mesh.n_prisms(),
                    points: Points::from_verts(mesh.verts(), encoding),
                    cells: Cells::from_prisms(mesh.prisms(), encoding),
                    cell_data: CellData::from_etags(mesh.prism_tags(), encoding),
                    point_data: PointData::new(),
                },
            },
            encoding,
        }
    }

    /// Create a vtu PolyMesh writer
    pub fn from_poly_mesh<const D: usize, M: PolyMesh<D>>(mesh: &M, encoding: VTUEncoding) -> Self {
        Self {
            grid_type: "UnstructuredGrid".to_string(),
            version: 0.1,
            header_type: "UInt32".to_string(),
            byte_order: "LittleEndian".to_string(),
            unstructured_grid: UnstructuredGrid {
                piece: Piece {
                    number_of_points: mesh.n_verts(),
                    number_of_cells: mesh.n_elems(),
                    points: Points::from_verts(mesh.verts(), encoding),
                    cells: Cells::from_poly(mesh, encoding, 0.1),
                    cell_data: CellData::from_etags(mesh.etags(), encoding),
                    point_data: PointData::new(),
                },
            },
            encoding,
        }
    }

    /// Add cell data
    pub fn add_cell_data(
        &mut self,
        name: &str,
        number_of_components: usize,
        data: impl Iterator<Item = f64>,
    ) {
        self.unstructured_grid
            .piece
            .cell_data
            .data_array
            .push(DataArray::new_f64(
                name,
                number_of_components,
                self.unstructured_grid.piece.number_of_cells,
                data,
                self.encoding,
            ));
    }

    /// Add point data
    pub fn add_point_data(
        &mut self,
        name: &str,
        number_of_components: usize,
        data: impl Iterator<Item = f64>,
    ) {
        self.unstructured_grid
            .piece
            .point_data
            .data_array
            .push(DataArray::new_f64(
                name,
                number_of_components,
                self.unstructured_grid.piece.number_of_points,
                data,
                self.encoding,
            ));
    }

    /// Write the file
    pub fn export(&self, file_name: &str) -> Result<()> {
        let f = std::fs::File::create(file_name)?;
        let mut writer = BufWriter::new(f);
        writeln!(writer, "<?xml version=\"1.0\"?>")?;
        to_utf8_io_writer(&mut writer, self)?;
        Ok(())
    }
}

#[derive(Serialize)]
#[serde(rename_all = "PascalCase")]
struct UnstructuredGrid {
    piece: Piece,
}

#[derive(Serialize)]
#[serde(rename_all = "PascalCase")]
struct Piece {
    #[serde(rename = "@NumberOfPoints")]
    number_of_points: usize,
    #[serde(rename = "@NumberOfCells")]
    number_of_cells: usize,
    points: Points,
    cells: Cells,
    cell_data: CellData,
    point_data: PointData,
}

#[derive(Serialize)]
#[serde(rename_all = "PascalCase")]
struct Points {
    data_array: DataArray,
}

impl Points {
    fn from_verts<const D: usize>(
        data: impl ExactSizeIterator<Item = Vertex<D>>,
        encoding: VTUEncoding,
    ) -> Self {
        let name = "Points";
        Self {
            data_array: match D {
                3 => DataArray::new_f64(
                    name,
                    3,
                    3 * data.len(),
                    data.flat_map(|x| [x[0], x[1], x[2]]),
                    encoding,
                ),
                2 => DataArray::new_f64(
                    name,
                    3,
                    3 * data.len(),
                    data.flat_map(|x| [x[0], x[1], 0.0]),
                    encoding,
                ),
                _ => unimplemented!(),
            },
        }
    }
}

#[derive(Serialize)]
struct DataArray {
    #[serde(rename = "@type")]
    data_type: String,
    #[serde(rename = "@Name")]
    name: String,
    #[serde(rename = "@format")]
    format: String,
    #[serde(rename = "@NumberOfComponents")]
    number_of_components: usize,
    #[serde(rename = "$text")]
    data: String,
}

fn encode<T>(len: usize, data: impl Iterator<Item = u8>) -> String {
    let capacity = size_of::<u32>() + len * size_of::<T>();

    let mut out = Vec::with_capacity(capacity);
    let header = ((len * size_of::<T>()) as u32).to_le_bytes();
    out.extend_from_slice(&header);
    out.extend(data);
    assert_eq!(out.len(), capacity);
    base64::prelude::BASE64_STANDARD.encode(out)
}

impl DataArray {
    fn new_f64(
        name: &str,
        number_of_components: usize,
        len: usize,
        data: impl Iterator<Item = f64>,
        encoding: VTUEncoding,
    ) -> Self {
        use std::fmt::Write;
        let (format, data) = match encoding {
            VTUEncoding::Ascii => (
                "ascii".to_string(),
                data.fold(String::new(), |mut output, b| {
                    let _ = write!(output, "{b} ");
                    output
                }),
            ),
            VTUEncoding::Binary => (
                "binary".to_string(),
                encode::<f64>(len, data.flat_map(f64::to_le_bytes)),
            ),
        };

        Self {
            data_type: "Float64".to_string(),
            name: name.to_string(),
            format,
            number_of_components,
            data,
        }
    }

    fn new_i64(
        name: &str,
        number_of_components: usize,
        len: usize,
        data: impl Iterator<Item = i64>,
        encoding: VTUEncoding,
    ) -> Self {
        use std::fmt::Write;
        let (format, data) = match encoding {
            VTUEncoding::Ascii => (
                "ascii".to_string(),
                data.fold(String::new(), |mut output, b| {
                    let _ = write!(output, "{b} ");
                    output
                }),
            ),
            VTUEncoding::Binary => (
                "binary".to_string(),
                encode::<i64>(len, data.flat_map(i64::to_le_bytes)),
            ),
        };

        Self {
            data_type: "Int64".to_string(),
            name: name.to_string(),
            format,
            number_of_components,
            data,
        }
    }

    #[allow(dead_code)]
    fn new_i32(
        name: &str,
        number_of_components: usize,
        len: usize,
        data: impl Iterator<Item = i32>,
        encoding: VTUEncoding,
    ) -> Self {
        use std::fmt::Write;
        let (format, data) = match encoding {
            VTUEncoding::Ascii => (
                "ascii".to_string(),
                data.fold(String::new(), |mut output, b| {
                    let _ = write!(output, "{b} ");
                    output
                }),
            ),
            VTUEncoding::Binary => (
                "binary".to_string(),
                encode::<i32>(len, data.flat_map(i32::to_le_bytes)),
            ),
        };

        Self {
            data_type: "Int32".to_string(),
            name: name.to_string(),
            format,
            number_of_components,
            data,
        }
    }

    #[allow(dead_code)]
    fn new_i16(
        name: &str,
        number_of_components: usize,
        len: usize,
        data: impl Iterator<Item = i16>,
        encoding: VTUEncoding,
    ) -> Self {
        use std::fmt::Write;
        let (format, data) = match encoding {
            VTUEncoding::Ascii => (
                "ascii".to_string(),
                data.fold(String::new(), |mut output, b| {
                    let _ = write!(output, "{b} ");
                    output
                }),
            ),
            VTUEncoding::Binary => (
                "binary".to_string(),
                encode::<i16>(len, data.flat_map(i16::to_le_bytes)),
            ),
        };

        Self {
            data_type: "Int16".to_string(),
            name: name.to_string(),
            format,
            number_of_components,
            data,
        }
    }

    fn new_u8(
        name: &str,
        number_of_components: usize,
        len: usize,
        data: impl Iterator<Item = u8>,
        encoding: VTUEncoding,
    ) -> Self {
        use std::fmt::Write;
        let (format, data) = match encoding {
            VTUEncoding::Ascii => (
                "ascii".to_string(),
                data.fold(String::new(), |mut output, b| {
                    let _ = write!(output, "{b} ");
                    output
                }),
            ),
            VTUEncoding::Binary => (
                "binary".to_string(),
                encode::<u8>(len, data.flat_map(u8::to_le_bytes)),
            ),
        };

        Self {
            data_type: "UInt8".to_string(),
            name: name.to_string(),
            format,
            number_of_components,
            data,
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "PascalCase")]
struct Cells {
    data_array: Vec<DataArray>,
}

impl Cells {
    fn from_elems<const D: usize, M: Mesh<D>>(mesh: &M, encoding: VTUEncoding) -> Self {
        let n = mesh.n_elems();

        let connectivity = DataArray::new_i64(
            "connectivity",
            1,
            <M::C as Simplex>::N_VERTS * n,
            mesh.elems().flatten().map(|x| x as i64),
            encoding,
        );

        let data = (0..n).map(|i| (<M::C as Simplex>::N_VERTS * (i + 1)) as i64);
        let offsets = DataArray::new_i64("offsets", 1, data.len(), data, encoding);

        let cell_type: u8 = match M::C::N_VERTS {
            4 => 10,
            3 => 5,
            2 => 4,
            _ => unreachable!(),
        };

        let data = (0..n).map(|_i| cell_type);
        let types = DataArray::new_u8("types", 1, data.len(), data, encoding);

        Self {
            data_array: vec![connectivity, offsets, types],
        }
    }

    fn from_prisms<'a>(
        prisms: impl ExactSizeIterator<Item = &'a Prism<impl Idx>>,
        encoding: VTUEncoding,
    ) -> Self {
        let n = prisms.len();

        let connectivity = DataArray::new_i64(
            "connectivity",
            1,
            6 * n,
            prisms.copied().flatten().map(|x| x as i64),
            encoding,
        );

        let data = (0..n).map(|i| (6 * (i + 1)) as i64);
        let offsets = DataArray::new_i64("offsets", 1, data.len(), data, encoding);

        let cell_type = 13_u8;

        let data = (0..n).map(|_i| cell_type);
        let types = DataArray::new_u8("types", 1, data.len(), data, encoding);

        Self {
            data_array: vec![connectivity, offsets, types],
        }
    }

    #[allow(clippy::too_many_lines)]
    fn from_poly<const D: usize, M: PolyMesh<D>>(
        mesh: &M,
        encoding: VTUEncoding,
        version: f64,
    ) -> Self {
        let n = mesh.n_elems();

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
        let connectivity = DataArray::new_i64(
            "connectivity",
            1,
            *offsets.last().unwrap(),
            connectivity.iter().map(|&x| x as i64),
            encoding,
        );

        let offsets =
            DataArray::new_i64("offsets", 1, n, offsets.iter().map(|&i| i as i64), encoding);

        let cell_type = match mesh.poly_type() {
            PolyMeshType::Polylines => 4,
            PolyMeshType::Polygons => 7,
            PolyMeshType::Polyhedra => 42,
        };

        let types = DataArray::new_u8("types", 1, n, (0..n).map(|_| cell_type), encoding);

        let mut data_array = vec![connectivity, offsets, types];

        if matches!(mesh.poly_type(), PolyMeshType::Polyhedra) {
            if (version - 0.1).abs() < 1e-6 {
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

                let faces = DataArray::new_i64(
                    "faces",
                    1,
                    faces.len(),
                    faces.iter().map(|&i| i as i64),
                    encoding,
                );

                let faceoffsets = DataArray::new_i64(
                    "faceoffsets",
                    1,
                    n,
                    faceoffsets.iter().map(|&i| i as i64),
                    encoding,
                );
                data_array.push(faces);
                data_array.push(faceoffsets);
            } else if (version - 2.3).abs() < 1e-6 {
                let mut polyhedron_offsets = Vec::with_capacity(n);
                let mut offset = 0;
                for e in mesh.elems() {
                    offset += e.len();
                    polyhedron_offsets.push(offset);
                }
                let polyhedron_to_faces = DataArray::new_i64(
                    "polyhedron_to_faces",
                    1,
                    *polyhedron_offsets.last().unwrap(),
                    mesh.elems().flat_map(|x| x.map(|(x, _)| x as i64)),
                    encoding,
                );

                let polyhedron_offsets = DataArray::new_i64(
                    "polyhedron_offsets",
                    1,
                    n,
                    polyhedron_offsets.iter().map(|&i| i as i64),
                    encoding,
                );

                let n = mesh.n_faces();
                let mut face_offsets = Vec::with_capacity(n);
                let mut offset = 0;
                for e in mesh.faces() {
                    offset += e.len();
                    face_offsets.push(offset);
                }
                let face_connectivity = DataArray::new_i64(
                    "face_connectivity",
                    1,
                    *face_offsets.last().unwrap(),
                    mesh.faces().flat_map(|x| x.map(|x| x as i64)),
                    encoding,
                );

                let face_offsets = DataArray::new_i64(
                    "face_offsets",
                    1,
                    n,
                    face_offsets.iter().map(|&i| i as i64),
                    encoding,
                );
                data_array.push(face_connectivity);
                data_array.push(face_offsets);
                data_array.push(polyhedron_to_faces);
                data_array.push(polyhedron_offsets);
            } else {
                unimplemented!();
            }
        }

        Self { data_array }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "PascalCase")]
struct CellData {
    data_array: Vec<DataArray>,
}

impl CellData {
    fn from_etags(data: impl ExactSizeIterator<Item = Tag>, encoding: VTUEncoding) -> Self {
        #[cfg(feature = "64bit-tags")]
        let tags = DataArray::new_i64("tags", 1, data.len(), data, encoding);
        #[cfg(feature = "32bit-tags")]
        let tags = DataArray::new_i32("tags", 1, data.len(), data, encoding);
        #[cfg(not(any(feature = "32bit-tags", feature = "64bit-tags")))]
        let tags = DataArray::new_i16("tags", 1, data.len(), data, encoding);

        Self {
            data_array: vec![tags],
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "PascalCase")]
struct PointData {
    data_array: Vec<DataArray>,
}

impl PointData {
    const fn new() -> Self {
        Self {
            data_array: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{VTUEncoding, VTUFile};
    use crate::mesh::{Mesh2d, rectangle_mesh};

    #[test]
    fn test_write_triangles() {
        let msh: Mesh2d = rectangle_mesh(1.0, 10, 2.0, 15);
        let writer = VTUFile::from_mesh(&msh, VTUEncoding::Binary);

        writer.export("toto.vtu").unwrap();
    }
}
