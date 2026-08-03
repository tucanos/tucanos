//! Generic zero-copy serializer for VTK XML files in Appended binary format.
//!
//! Provides traits and abstractions for formatting structured or unstructured
//! geometric primitives into raw binary VTK XML structures (`.vtp`).
use std::{
    collections::BTreeMap,
    io::{Result, Write},
    mem::size_of,
};

/// Trait implemented by specific VTK file structures (e.g., `PolyData`, `UnstructuredGrid`)
trait FileType {
    const NAME: &str;
    fn write_piece_attributes(&self, writer: &mut impl Write) -> Result<()>;
}

#[derive(Default)]
struct PolyData {
    points: usize,
    lines: usize,
    verts: usize,
}

impl FileType for PolyData {
    const NAME: &str = "PolyData";

    fn write_piece_attributes(&self, writer: &mut impl Write) -> Result<()> {
        write!(
            writer,
            r#"NumberOfPoints="{}" NumberOfVerts="{}" NumberOfLines="{}""#,
            self.points, self.verts, self.lines
        )
    }
}

/// Helper to write VTK `PolyData` in appended format.
#[derive(Default)]
pub struct PolyDataWriter<'a>(AppendedWriter<'a, PolyData>);

impl<'a> PolyDataWriter<'a> {
    /// Sets the total number of points in the dataset.
    pub const fn set_num_points(&mut self, n: usize) {
        self.0.file_type.points = n;
    }

    /// Sets the total number of line cells in the dataset.
    pub const fn set_num_lines(&mut self, n: usize) {
        self.0.file_type.lines = n;
    }

    /// Sets the total number of vertex cells in the dataset.
    pub const fn set_num_verts(&mut self, n: usize) {
        self.0.file_type.verts = n;
    }

    /// Adds 3D point coordinates to the `.vtp` dataset.
    ///
    /// # Arguments
    /// * `iterator` - Iterator yielding coordinate values for all points.
    ///   The total number of scalar items must equal `3 * set_num_points`.
    pub fn add_points<T, IT>(&mut self, iterator: IT)
    where
        T: Scalar + 'a,
        IT: IntoIterator<Item = T> + 'a,
    {
        self.0.sections.insert(
            "Points",
            vec![DataArray::new(
                "Points",
                3,
                self.0.file_type.points,
                iterator,
            )],
        );
    }
    fn add_elems<T, ITO, ITC>(
        &mut self,
        num_conn: usize,
        connectivity: ITC,
        num: usize,
        offsets: ITO,
        section: &'a str,
    ) where
        T: Scalar + 'a,
        ITC: IntoIterator<Item = T> + 'a,
        ITO: IntoIterator<Item = T> + 'a,
    {
        self.0.sections.insert(
            section,
            vec![
                DataArray::new("connectivity", 1, num_conn, connectivity),
                DataArray::new("offsets", 1, num, offsets),
            ],
        );
    }

    /// Adds vertex elements (point cells) to the `.vtp` file.
    ///
    /// # Arguments
    /// * `num_conn` - Total number of connectivity indices across all vertex cells.
    /// * `connectivity` - Iterator over point indices referenced by the vertices.
    /// * `offsets` - Iterator over cumulative end-index offsets for each vertex cell.
    pub fn add_verts<T, ITO, ITC>(&mut self, num_conn: usize, connectivity: ITC, offsets: ITO)
    where
        T: Scalar + 'a,
        ITC: IntoIterator<Item = T> + 'a,
        ITO: IntoIterator<Item = T> + 'a,
    {
        self.add_elems(
            num_conn,
            connectivity,
            self.0.file_type.verts,
            offsets,
            "Verts",
        );
    }

    /// Adds polyline elements to the `.vtp` file.
    ///
    /// Supports multi-segment polylines using explicit connectivity offsets.
    ///
    /// # Arguments
    /// * `num_conn` - Total number of connectivity indices across all line cells.
    /// * `connectivity` - Iterator over point indices forming the lines.
    /// * `offsets` - Iterator yielding cumulative end-index offsets for each polyline cell.
    pub fn add_lines<T, ITO, ITC>(&mut self, num_conn: usize, connectivity: ITC, offsets: ITO)
    where
        T: Scalar + 'a,
        ITC: IntoIterator<Item = T> + 'a,
        ITO: IntoIterator<Item = T> + 'a,
    {
        self.add_elems(
            num_conn,
            connectivity,
            self.0.file_type.lines,
            offsets,
            "Lines",
        );
    }

    /// Adds a `CellData` field array to the dataset.
    ///
    /// In VTK PolyData (`.vtp`), cell attributes for all cell types are concatenated
    /// into a single contiguous array. The values in `values` **must** be ordered
    /// sequentially by cell type in the following order:
    ///
    /// 1. **Verts** (Vertices / PolyVertices)
    /// 2. **Lines** (Lines / PolyLines)
    /// 3. **Polys** (Triangles, Quads, Polygons)
    /// 4. **Strips** (Triangle Strips)
    ///
    /// Within each cell type, data points must match the insertion order of the cells.
    /// If `num_components > 1` (e.g. 3 for 3D vectors), components for a single cell
    /// are packed contiguously before moving to the next cell.
    ///
    /// # Arguments
    ///
    /// * `label` - Name of the field array
    /// * `num_components` - Number of components per cell (e.g., `1` for scalar, `3` for 3D vector).
    /// * `values` - Iterator yielding scalar values for all cells combined.
    pub fn add_cell_data<T, IT>(&mut self, label: &str, num_components: usize, values: IT)
    where
        T: Scalar + 'a,
        IT: IntoIterator<Item = T> + 'a,
    {
        let num_cells = self.num_cells();
        self.0
            .sections
            .entry("CellData")
            .or_default()
            .push(DataArray::new(label, num_components, num_cells, values));
    }

    const fn num_cells(&self) -> usize {
        self.0.file_type.verts + self.0.file_type.lines
    }

    pub fn write(self, writer: &mut impl Write) -> Result<()> {
        self.0.write(writer)
    }
}

#[derive(Default)]
pub struct UnstructuredGridWriter<'a>(AppendedWriter<'a, UnstructuredGrid>);

impl<'a> UnstructuredGridWriter<'a> {
    pub const fn set_num_points(&mut self, n: usize) {
        self.0.file_type.number_of_points = n;
    }
    pub const fn set_num_cells(&mut self, n: usize) {
        self.0.file_type.number_of_cells = n;
    }
    pub fn add_points<T, IT>(&mut self, iterator: IT)
    where
        T: Scalar + 'a,
        IT: IntoIterator<Item = T> + 'a,
    {
        self.0.sections.insert(
            "Points",
            vec![DataArray::new(
                "Points",
                3,
                self.0.file_type.number_of_points,
                iterator,
            )],
        );
    }
    pub fn add_cells<TC, ITC, TO, ITO, TT, ITT>(
        &mut self,
        num_conn: usize,
        connectivity: ITC,
        offsets: ITO,
        types: ITT,
    ) where
        TC: Scalar + 'a,
        TO: Scalar + 'a,
        TT: Scalar + 'a,
        ITC: IntoIterator<Item = TC> + 'a,
        ITO: IntoIterator<Item = TO> + 'a,
        ITT: IntoIterator<Item = TT> + 'a,
    {
        self.0.sections.insert(
            "Cells",
            vec![
                DataArray::new("connectivity", 1, num_conn, connectivity),
                DataArray::new("offsets", 1, self.0.file_type.number_of_cells, offsets),
                DataArray::new("types", 1, self.0.file_type.number_of_cells, types),
            ],
        );
    }

    /// Add polyhedron faces. Must be called after `add_cells`.
    pub fn add_polyhedron_faces<T, ITF, ITO>(
        &mut self,
        num_conn: usize,
        faces: ITF,
        num_faces: usize,
        faceoffsets: ITO,
    ) where
        T: Scalar + 'a,
        ITF: IntoIterator<Item = T> + 'a,
        ITO: IntoIterator<Item = T> + 'a,
    {
        let s = self
            .0
            .sections
            .get_mut("Cells")
            .expect("add_polyhedron_faces must be called after add_cells");
        s.push(DataArray::new("faces", 1, num_conn, faces));
        s.push(DataArray::new("faceoffsets", 1, num_faces, faceoffsets));
    }

    pub fn add_polyhedron_faces_v23<T, ITFC, ITFO>(
        &mut self,
        num_conn: usize,
        face_connectivity: ITFC,
        num_faces: usize,
        face_offsets: ITFO,
    ) where
        T: Scalar + 'a,
        ITFC: IntoIterator<Item = T> + 'a,
        ITFO: IntoIterator<Item = T> + 'a,
    {
        self.0.version = "2.3";
        let s = self
            .0
            .sections
            .get_mut("Cells")
            .expect("add_polyhedron_faces must be called after add_cells");
        s.push(DataArray::new(
            "face_connectivity",
            1,
            num_conn,
            face_connectivity,
        ));
        s.push(DataArray::new("face_offsets", 1, num_faces, face_offsets));
    }

    pub fn add_polyhedron_face_map_v23<T, ITPF, ITPO>(
        &mut self,
        num_p_faces: usize,
        polyhedron_to_faces: ITPF,
        polyhedron_offsets: ITPO,
    ) where
        T: Scalar + 'a,
        ITPF: IntoIterator<Item = T> + 'a,
        ITPO: IntoIterator<Item = T> + 'a,
    {
        self.0.version = "2.3";
        let num_cells = self.0.file_type.number_of_cells;
        let s = self
            .0
            .sections
            .get_mut("Cells")
            .expect("add_polyhedron_faces must be called after add_cells");
        s.push(DataArray::new(
            "polyhedron_to_faces",
            1,
            num_p_faces,
            polyhedron_to_faces,
        ));
        s.push(DataArray::new(
            "polyhedron_offsets",
            1,
            num_cells,
            polyhedron_offsets,
        ));
    }

    pub fn add_cell_data<T, IT>(&mut self, label: &str, num_components: usize, values: IT)
    where
        T: Scalar + 'a,
        IT: IntoIterator<Item = T> + 'a,
    {
        let num_cells = self.num_cells();
        self.0
            .sections
            .entry("CellData")
            .or_default()
            .push(DataArray::new(label, num_components, num_cells, values));
    }

    pub fn add_point_data<T, IT>(&mut self, label: &str, num_components: usize, values: IT)
    where
        T: Scalar + 'a,
        IT: IntoIterator<Item = T> + 'a,
    {
        let d = DataArray::new(
            label,
            num_components,
            self.0.file_type.number_of_points,
            values,
        );
        self.0.sections.entry("PointData").or_default().push(d);
    }

    const fn num_cells(&self) -> usize {
        self.0.file_type.number_of_cells
    }

    pub fn write(self, writer: &mut impl Write) -> Result<()> {
        self.0.write(writer)
    }
}

#[derive(Default)]
pub struct UnstructuredGrid {
    number_of_points: usize,
    number_of_cells: usize,
}

impl FileType for UnstructuredGrid {
    const NAME: &str = "UnstructuredGrid";

    fn write_piece_attributes(&self, writer: &mut impl Write) -> Result<()> {
        write!(
            writer,
            r#"NumberOfPoints="{}" NumberOfCells="{}""#,
            self.number_of_points, self.number_of_cells
        )
    }
}

struct AppendedWriter<'a, T: FileType> {
    file_type: T,
    version: &'static str,
    sections: BTreeMap<&'a str, Vec<DataArray<'a>>>,
}

impl<T: FileType + Default> Default for AppendedWriter<'_, T> {
    fn default() -> Self {
        Self {
            file_type: T::default(),
            version: "1.0",
            sections: BTreeMap::default(),
        }
    }
}

impl<'a, T: FileType> AppendedWriter<'a, T> {
    fn write(mut self, writer: &mut impl Write) -> Result<()> {
        let typ = T::NAME;
        let endianness = if cfg!(target_endian = "little") {
            "LittleEndian"
        } else {
            "BigEndian"
        };
        write!(
            writer,
            r#"<VTKFile type="{typ}" version="{}" byte_order="{endianness}""#,
            self.version
        )?;
        writeln!(
            writer,
            r#" header_type="UInt32">
  <{typ}>
    <Piece "#
        )?;
        self.file_type.write_piece_attributes(writer)?;
        writeln!(writer, ">")?;

        let mut offset = 0;

        let mut write_section = |name: &str, arrays: &[DataArray<'a>]| -> Result<()> {
            if !arrays.is_empty() {
                writeln!(writer, "      <{name}>")?;
                for a in arrays {
                    writeln!(writer, "        {}", a.to_xml_tag(offset))?;
                    offset += size_of::<u32>() + a.byte_len;
                }
                writeln!(writer, "      </{name}>")?;
            }
            Ok(())
        };

        for (name, arrays) in &self.sections {
            write_section(name, arrays)?;
        }
        write!(
            writer,
            "    </Piece>\n  </{typ}>\n  <AppendedData encoding=\"raw\">\n   _"
        )?;
        for section in self.sections.values_mut() {
            for array in section {
                array.write(writer)?;
            }
        }
        writeln!(writer, "\n  </AppendedData>\n</VTKFile>")
    }
}

struct DataArray<'a> {
    data_type: &'static str,
    name: String,
    number_of_components: usize,
    byte_len: usize,
    data: Box<dyn WriteableIter + 'a>,
}

pub trait Scalar: Sized + Copy {
    const TYPE_NAME: &'static str;
    fn write_ne_bytes(&self, writer: &mut dyn Write) -> Result<()>;
}

#[cfg(target_pointer_width = "64")]
impl Scalar for usize {
    const TYPE_NAME: &'static str = "UInt64";
    fn write_ne_bytes(&self, writer: &mut dyn Write) -> Result<()> {
        writer.write_all(&Self::to_ne_bytes(*self))
    }
}

impl Scalar for i64 {
    const TYPE_NAME: &'static str = "Int64";
    fn write_ne_bytes(&self, writer: &mut dyn Write) -> Result<()> {
        writer.write_all(&Self::to_ne_bytes(*self))
    }
}

impl Scalar for i32 {
    const TYPE_NAME: &'static str = "Int32";
    fn write_ne_bytes(&self, writer: &mut dyn Write) -> Result<()> {
        writer.write_all(&Self::to_ne_bytes(*self))
    }
}

impl Scalar for i16 {
    const TYPE_NAME: &'static str = "Int16";
    fn write_ne_bytes(&self, writer: &mut dyn Write) -> Result<()> {
        writer.write_all(&Self::to_ne_bytes(*self))
    }
}

impl Scalar for u8 {
    const TYPE_NAME: &'static str = "UInt8";
    fn write_ne_bytes(&self, writer: &mut dyn Write) -> Result<()> {
        writer.write_all(&Self::to_ne_bytes(*self))
    }
}

impl Scalar for i8 {
    const TYPE_NAME: &'static str = "Int8";
    fn write_ne_bytes(&self, writer: &mut dyn Write) -> Result<()> {
        writer.write_all(&Self::to_ne_bytes(*self))
    }
}

impl Scalar for u32 {
    const TYPE_NAME: &'static str = "UInt32";
    fn write_ne_bytes(&self, writer: &mut dyn Write) -> Result<()> {
        writer.write_all(&Self::to_ne_bytes(*self))
    }
}

impl Scalar for f32 {
    const TYPE_NAME: &'static str = "Float32";
    fn write_ne_bytes(&self, writer: &mut dyn Write) -> Result<()> {
        writer.write_all(&Self::to_ne_bytes(*self))
    }
}

impl Scalar for f64 {
    const TYPE_NAME: &'static str = "Float64";
    fn write_ne_bytes(&self, writer: &mut dyn Write) -> Result<()> {
        writer.write_all(&Self::to_ne_bytes(*self))
    }
}

impl<const D: usize> Scalar for [f32; D] {
    const TYPE_NAME: &'static str = "Float32";

    fn write_ne_bytes(&self, writer: &mut dyn Write) -> Result<()> {
        for val in self {
            writer.write_all(&val.to_ne_bytes())?;
        }
        Ok(())
    }
}

impl<const D: usize> Scalar for [f64; D] {
    const TYPE_NAME: &'static str = "Float64";

    fn write_ne_bytes(&self, writer: &mut dyn Write) -> Result<()> {
        for val in self {
            writer.write_all(&val.to_ne_bytes())?;
        }
        Ok(())
    }
}

impl<T: Scalar> Scalar for &T {
    const TYPE_NAME: &'static str = T::TYPE_NAME;
    fn write_ne_bytes(&self, writer: &mut dyn Write) -> Result<()> {
        (*self).write_ne_bytes(writer)
    }
}

trait WriteableIter {
    fn write_to(&mut self, writer: &mut dyn Write) -> Result<()>;
}

impl<I, T> WriteableIter for I
where
    I: Iterator<Item = T>,
    T: Scalar,
{
    fn write_to(&mut self, writer: &mut dyn Write) -> Result<()> {
        for item in self {
            item.write_ne_bytes(writer)?;
        }
        Ok(())
    }
}

impl<'a> DataArray<'a> {
    fn new<IT>(name: &str, number_of_components: usize, len: usize, data: IT) -> Self
    where
        IT: IntoIterator + 'a,
        IT::Item: Scalar,
    {
        Self {
            data_type: <IT::Item as Scalar>::TYPE_NAME,
            name: name.to_string(),
            number_of_components,
            data: Box::new(data.into_iter()),
            byte_len: len * std::mem::size_of::<IT::Item>(),
        }
    }

    #[must_use]
    fn to_xml_tag(&self, offset: usize) -> String {
        format!(
            concat!(
                r#"<DataArray type="{}" Name="{}" NumberOfComponents="{}" "#,
                r#"format="appended" offset="{}"/>"#
            ),
            self.data_type, self.name, self.number_of_components, offset
        )
    }

    fn write<W: Write>(&mut self, writer: &mut W) -> Result<()> {
        let len = self.byte_len as u32;
        writer.write_all(&len.to_ne_bytes())?;
        self.data.write_to(writer)
    }
}
